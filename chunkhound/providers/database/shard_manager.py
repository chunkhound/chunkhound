"""Shard lifecycle manager for ChunkHound.

Coordinates shard operations including search, insert, delete, and maintenance.
Uses derived state architecture - metrics computed from DuckDB and USearch files.
"""

import gc
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans  # type: ignore[import-untyped]

from chunkhound.core.config.sharding_config import ShardingConfig, bytes_per_vector
from chunkhound.providers.database import usearch_wrapper
from chunkhound.providers.database.shard_state import ShardState, get_shard_state
from chunkhound.providers.database.usearch_wrapper import SearchResult
from chunkhound.utils.windows_constants import fsync_path

# Sample size for k-means centroid discovery during shard splits.
# Larger values improve clustering quality but increase memory usage.
KMEANS_SAMPLE_SIZE = 512

# Heartbeat intervals for progress signaling during long operations.
# Different operations have different batch sizes and performance characteristics.
HEARTBEAT_INTERVAL_INSERT_ROUTING = 100  # Per-vector centroid routing (fine-grained)
HEARTBEAT_INTERVAL_MERGE_ROUTING = 10_000  # Bulk reassignment (coarse-grained)


def _parse_uuid(value: Any) -> UUID:
    """Parse UUID from DuckDB result (handles both UUID objects and strings)."""
    return value if isinstance(value, UUID) else UUID(value)


@contextmanager
def _transaction(conn: Any) -> Generator[None, None, None]:
    """Execute a block within an explicit DuckDB transaction.

    DuckDB auto-commits each statement by default. This context manager
    ensures atomicity for multi-statement operations like split/merge.

    Args:
        conn: DuckDB connection

    Yields:
        None

    Raises:
        Any exception from the wrapped block (after rollback)
    """
    conn.execute("BEGIN TRANSACTION")
    try:
        yield
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


class ShardManager:
    """Coordinates shard lifecycle operations.

    Manages shard search, insert, delete, and maintenance operations.
    Maintains centroid cache for shard routing decisions.
    """

    def __init__(
        self,
        db_provider: Any,
        shard_dir: Path,
        config: ShardingConfig,
        heartbeat_callback: Callable[[], None] | None = None,
    ) -> None:
        """Initialize shard manager.

        Args:
            db_provider: Database provider for DuckDB operations
            shard_dir: Directory containing .usearch shard files
            config: Sharding configuration with thresholds
            heartbeat_callback: Optional callback to signal progress during
                long-running operations (prevents timeout)
        """
        self.db = db_provider
        self.shard_dir = shard_dir
        self.config = config
        self.centroids: dict[UUID, np.ndarray] = {}
        self._heartbeat = heartbeat_callback or (lambda: None)

    def _calculate_max_concurrent_shards(self, dims: int, quantization: str) -> int:
        """Calculate max concurrent shards to stay under memory budget.

        Args:
            dims: Vector dimensionality
            quantization: Quantization type (i8, f16, f32, f64)

        Returns:
            Number of shards that can be loaded concurrently within memory budget
        """
        bpv = bytes_per_vector(dims, quantization, self.config.hnsw_connectivity)
        shard_memory = self.config.split_threshold * bpv

        if shard_memory == 0:
            return self.config.max_concurrent_shards  # fallback to config

        max_shards = self.config.memory_budget_bytes // shard_memory
        return max(1, max_shards)  # floor at 1, no artificial cap

    def search(
        self,
        query: list[float],
        k: int,
        dims: int,
        provider: str,
        model: str,
        conn: Any,
    ) -> list[SearchResult]:
        """Search across shards for nearest neighbors.

        Uses centroid-based filtering to reduce search space:
        1. Filter shards by centroid similarity threshold
        2. Batch search relevant shards via usearch_wrapper.multi_search
        3. Return top k results sorted by distance

        Args:
            query: Query embedding vector
            k: Number of nearest neighbors to return
            dims: Embedding dimensions
            provider: Embedding provider name
            model: Embedding model name
            conn: Database connection (required - passed from executor)

        Returns:
            List of SearchResult sorted by distance
        """
        query_array = np.array(query, dtype=np.float32)

        result = conn.execute(
            """
            SELECT shard_id, quantization FROM vector_shards
            WHERE dims = ? AND provider = ? AND model = ?
            """,
            [dims, provider, model],
        ).fetchall()

        if not result:
            return []

        # Get quantization for memory calculation (use first shard's, all should match)
        quantization = result[0][1] if result else self.config.default_quantization

        # Filter shards by centroid similarity
        relevant_shards: list[tuple[UUID, Path]] = []
        for i, row in enumerate(result):
            # Heartbeat every 100 shards to prevent timeout during large shard filtering
            if i > 0 and i % 100 == 0:
                self._heartbeat()

            # DuckDB returns UUID objects directly
            shard_id = _parse_uuid(row[0])
            file_path = self._shard_path(shard_id)  # Always derive path

            # Check file exists
            if not file_path.exists():
                continue

            # Check centroid similarity
            centroid = self.centroids.get(shard_id)
            if centroid is not None:
                # Compute cosine similarity
                query_norm = np.linalg.norm(query_array)
                centroid_norm = np.linalg.norm(centroid)
                if query_norm > 0 and centroid_norm > 0:
                    similarity = np.dot(query_array, centroid) / (
                        query_norm * centroid_norm
                    )
                    if similarity < self.config.shard_similarity_threshold:
                        continue  # Skip distant shard

            relevant_shards.append((shard_id, file_path))

        if not relevant_shards:
            return []

        # Batch search across relevant shards (limit by memory budget)
        all_results: list[SearchResult] = []
        max_concurrent = self._calculate_max_concurrent_shards(dims, quantization)

        for i in range(0, len(relevant_shards), max_concurrent):
            # Heartbeat before each batch to signal search progress
            self._heartbeat()

            batch = relevant_shards[i : i + max_concurrent]
            # Use Path objects for type compatibility with usearch_wrapper
            paths: list[Path | str] = [shard[1] for shard in batch]

            try:
                internal_k = k * self.config.overfetch_multiplier
                batch_results = usearch_wrapper.multi_search(
                    paths, query_array, internal_k
                )
                all_results.extend(batch_results)
            except Exception as e:
                logger.warning(f"Batch search failed: {e}")
                continue

        # Sort by distance and return top k
        all_results.sort(key=lambda r: r.distance)
        return all_results[:k]

    def insert_embeddings(
        self,
        embeddings: list[dict],
        dims: int,
        provider: str,
        model: str,
        conn: Any,
    ) -> tuple[bool, bool]:
        """Insert embeddings with per-vector centroid routing.

        Routes each embedding to its nearest shard based on centroid similarity.
        If no shards exist, creates a new shard.
        May trigger shard splits if thresholds exceeded.

        Args:
            embeddings: List of dicts with keys: id, embedding (vector)
            dims: Embedding dimensions
            provider: Embedding provider name
            model: Embedding model name
            conn: Database connection (required - passed from executor)

        Returns:
            Tuple of (success, needs_fix_pass) flags.
            needs_fix_pass is True if a new shard was created or
            if shard count exceeds split threshold.
        """
        if not embeddings:
            return (True, False)

        try:
            table_name = f"embeddings_{dims}"
            targets, was_created = self._get_target_shards(dims, provider, model, conn)

            if not targets:
                return (False, False)

            # PHASE 1: Precompute per-vector shard assignments
            assignments: dict[UUID, list[int]] = {t[0]: [] for t in targets}
            vectors_by_shard: dict[UUID, dict[int, list[float]]] = {
                t[0]: {} for t in targets
            }

            # Single shard or no centroids: all to first target
            if len(targets) == 1 or targets[0][1] is None:
                target_id = targets[0][0]
                for emb in embeddings:
                    emb_id = emb.get("id")
                    vector = emb.get("embedding")
                    if emb_id is not None:
                        assignments[target_id].append(emb_id)
                        if vector is not None:
                            vectors_by_shard[target_id][emb_id] = vector
            else:
                # Signal start of per-vector routing (potentially long-running)
                self._heartbeat()

                # PER-VECTOR ROUTING: assign each embedding to nearest centroid
                for idx, emb in enumerate(embeddings):
                    emb_id = emb.get("id")
                    vector = emb.get("embedding")
                    if emb_id is None:
                        continue

                    # Heartbeat every 100 vectors to prevent timeout.
                    # Insert uses per-vector centroid distance (CPU-intensive),
                    # so more frequent heartbeats needed for large batches (1000+).
                    if idx > 0 and idx % HEARTBEAT_INTERVAL_INSERT_ROUTING == 0:
                        self._heartbeat()

                    if vector is None:
                        # No vector: assign to first shard
                        assignments[targets[0][0]].append(emb_id)
                        continue

                    vec_array = np.array(vector, dtype=np.float32)
                    vec_norm = np.linalg.norm(vec_array)

                    # Find nearest shard by cosine distance
                    best_target = targets[0][0]
                    best_distance = float("inf")

                    for target_id, centroid in targets:
                        if centroid is None:
                            continue
                        centroid_norm = np.linalg.norm(centroid)
                        if vec_norm > 0 and centroid_norm > 0:
                            dist = 1.0 - np.dot(vec_array, centroid) / (
                                vec_norm * centroid_norm
                            )
                        else:
                            dist = 1.0
                        if dist < best_distance:
                            best_distance = dist
                            best_target = target_id

                    assignments[best_target].append(emb_id)
                    vectors_by_shard[best_target][emb_id] = vector

            # PHASE 2: Batch UPDATE per shard (one SQL per shard)
            for target_id, emb_ids in assignments.items():
                if not emb_ids:
                    continue
                placeholders = ",".join(["?"] * len(emb_ids))
                conn.execute(
                    f"UPDATE {table_name} SET shard_id = ? "
                    f"WHERE id IN ({placeholders})",
                    [str(target_id)] + emb_ids,
                )

            # PHASE 3: Incremental HNSW update per affected shard
            needs_fix_pass = was_created
            for target_id, vectors_dict in vectors_by_shard.items():
                if not vectors_dict:
                    continue

                # Signal progress before blocking HNSW index.add() operation
                self._heartbeat()

                file_path = self._shard_path(target_id)

                try:
                    if file_path.exists():
                        # Open existing index for incremental add
                        index = usearch_wrapper.open_writable(file_path)
                    else:
                        # Create new index for shard without file
                        quant_result = conn.execute(
                            "SELECT quantization FROM vector_shards WHERE shard_id = ?",
                            [str(target_id)],
                        ).fetchone()
                        quantization = (
                            quant_result[0]
                            if quant_result
                            else self.config.default_quantization
                        )

                        index = usearch_wrapper.create(
                            dims,
                            quantization,
                            connectivity=self.config.hnsw_connectivity,
                            expansion_add=self.config.hnsw_expansion_add,
                            expansion_search=self.config.hnsw_expansion_search,
                        )
                        self.shard_dir.mkdir(parents=True, exist_ok=True)

                    add_keys = np.array(list(vectors_dict.keys()), dtype=np.uint64)
                    add_vectors = np.array(
                        list(vectors_dict.values()), dtype=np.float32
                    )
                    index.add(add_keys, add_vectors)

                    # Atomic save
                    tmp_path = file_path.with_suffix(".usearch.tmp")
                    index.save(str(tmp_path))
                    fsync_path(tmp_path)
                    tmp_path.replace(file_path)

                    # Explicit cleanup to release RAM
                    del index

                    logger.debug(
                        f"Added {len(vectors_dict)} vectors to shard {target_id}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to add vectors to shard {target_id}: {e}")
                    needs_fix_pass = True

                # Check split threshold per shard
                count = self._get_shard_count(target_id, dims, conn)
                if count >= self.config.split_threshold:
                    needs_fix_pass = True

            return (True, needs_fix_pass)

        except Exception as e:
            logger.error(f"Failed to insert embeddings: {e}")
            return (False, False)

    def _get_target_shards(
        self,
        dims: int,
        provider: str,
        model: str,
        conn: Any,
    ) -> tuple[list[tuple[UUID, np.ndarray | None]], bool]:
        """Get candidate target shards for per-vector routing.

        Returns list of shards with their cached centroids for routing decisions.
        Creates a new shard if none exist for dims/provider/model.

        Args:
            dims: Embedding dimensions
            provider: Embedding provider name
            model: Embedding model name
            conn: Database connection

        Returns:
            Tuple of (list of (shard_id, centroid or None), was_new_shard_created)
        """
        result = conn.execute(
            """
            SELECT shard_id FROM vector_shards
            WHERE dims = ? AND provider = ? AND model = ?
            """,
            [dims, provider, model],
        ).fetchall()

        if not result:
            # Create new shard
            new_shard_id = uuid4()
            try:
                conn.execute(
                    """
                    INSERT INTO vector_shards
                        (shard_id, dims, provider, model, quantization)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        str(new_shard_id),
                        dims,
                        provider,
                        model,
                        self.config.default_quantization,
                    ],
                )
                logger.info(
                    f"Created new shard {new_shard_id} for {dims}D {provider}/{model}"
                )
                return ([(new_shard_id, None)], True)
            except Exception as e:
                logger.error(f"Failed to create shard: {e}")
                return ([], False)

        # Build list with cached centroids
        targets: list[tuple[UUID, np.ndarray | None]] = []
        for row in result:
            shard_id = _parse_uuid(row[0])
            centroid = self.centroids.get(shard_id)
            targets.append((shard_id, centroid))

        # If no centroids available, only return first shard
        if all(c is None for _, c in targets):
            return ([targets[0]], False)

        # Filter to only shards with centroids for routing
        return ([t for t in targets if t[1] is not None], False)

    def _get_shard_count(self, shard_id: UUID, dims: int, conn: Any) -> int:
        """Get count of embeddings in a shard."""
        table_name = f"embeddings_{dims}"
        result = conn.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE shard_id = ?",
            [str(shard_id)],
        ).fetchone()
        return result[0] if result else 0

    def delete_embedding(
        self,
        embedding_id: int,
        dims: int,
        conn: Any,
    ) -> tuple[bool, bool]:
        """Delete embedding from its shard.

        Hard DELETE from DuckDB and tombstone in USearch index.
        Tombstoning (compact=False) marks the vector as deleted without
        restructuring the HNSW graph, keeping deletion O(1).

        Args:
            embedding_id: ID of embedding to delete
            dims: Embedding dimensions
            conn: Database connection (required - passed from executor)

        Returns:
            Tuple of (success, needs_fix_pass) flags
        """
        table_name = f"embeddings_{dims}"

        try:
            # Get shard_id before deleting
            result = conn.execute(
                f"SELECT shard_id FROM {table_name} WHERE id = ?",
                [embedding_id],
            ).fetchone()

            if result is None:
                # Embedding not found
                return (False, False)

            shard_id_str = result[0]
            shard_id = UUID(shard_id_str) if shard_id_str else None

            # Hard DELETE from DuckDB
            conn.execute(
                f"DELETE FROM {table_name} WHERE id = ?",
                [embedding_id],
            )

            # Remove from USearch index with tombstone (no compaction)
            needs_fix_pass = False
            if shard_id is not None:
                file_path = self._shard_path(shard_id)
                if file_path.exists():
                    try:
                        index = usearch_wrapper.open_writable(file_path)
                        index.remove([embedding_id], compact=False)

                        # Atomic save
                        tmp_path = file_path.with_suffix(".usearch.tmp")
                        index.save(str(tmp_path))
                        fsync_path(tmp_path)
                        tmp_path.replace(file_path)

                        # Explicit cleanup to release RAM
                        del index

                        logger.debug(
                            f"Tombstoned embedding {embedding_id} in shard {shard_id}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to tombstone embedding {embedding_id} "
                            f"in shard {shard_id}: {e}, will trigger fix_pass"
                        )
                        needs_fix_pass = True

                # Check if below merge threshold or empty
                db_count = self._get_shard_count(shard_id, dims, conn)
                if db_count < self.config.merge_threshold or db_count == 0:
                    needs_fix_pass = True

            return (True, needs_fix_pass)

        except Exception as e:
            logger.error(f"Failed to delete embedding {embedding_id}: {e}")
            return (False, False)

    def fix_pass(
        self,
        conn: Any,
        check_quality: bool = True,
    ) -> None:
        """Run maintenance pass on all shards.

        Uses LIRE-style convergence loop:
        1. Cleanup orphaned and temp files
        2. For each shard: rebuild or incremental sync as needed
        3. Repeat until no changes occur
        4. Populate centroid cache

        Args:
            conn: Database connection (required - passed from executor)
            check_quality: If True, measure self-recall for each shard
        """
        logger.info("Starting fix_pass maintenance")
        self._heartbeat()

        # Cleanup phase
        orphaned_count = self._cleanup_orphaned_files(conn)
        temp_count = self._cleanup_temp_files()
        if orphaned_count or temp_count:
            logger.info(
                f"Cleanup: removed {orphaned_count} orphaned, {temp_count} temp files"
            )

        # LIRE-style convergence loop - repeat until no changes
        max_iterations = 100  # Safety limit per spec
        iteration = 0

        # Track rebuild attempts per shard to prevent infinite loops
        # if recall never improves (e.g., degenerate identical vectors)
        rebuild_attempts: dict[UUID, int] = {}
        max_rebuild_attempts = 3

        while iteration < max_iterations:
            iteration += 1
            changes_made = 0
            self._heartbeat()

            # Get all shards from DB
            shards = self._list_shards(conn)
            if not shards:
                logger.debug("No shards found, skipping fix_pass")
                break

            for shard in shards:
                self._heartbeat()
                shard_id = UUID(shard["shard_id"])
                dims = shard["dims"]
                file_path = self._shard_path(shard_id)

                # Get current state
                try:
                    state = get_shard_state(
                        shard_id=shard_id,
                        db_connection=conn,
                        file_path=file_path,
                        dims=dims,
                        measure_quality=check_quality,
                    )
                except FileNotFoundError:
                    # Missing file requires rebuild (normal for newly created shards)
                    logger.info(f"Shard {shard_id} missing file, rebuilding")
                    if self._rebuild_index_from_duckdb(shard, conn):
                        changes_made += 1
                    continue
                except (ValueError, RuntimeError) as e:
                    # Corrupted file requires rebuild
                    logger.warning(f"Shard {shard_id} corrupted: {e}, rebuilding")
                    if file_path.exists():
                        file_path.unlink()
                    if self._rebuild_index_from_duckdb(shard, conn):
                        changes_made += 1
                    continue

                # Check if rebuild needed
                if self._needs_rebuild(shard, state):
                    attempts = rebuild_attempts.get(shard_id, 0)
                    if attempts >= max_rebuild_attempts:
                        logger.warning(
                            f"Shard {shard_id} rebuilt {attempts} times "
                            "without recall improvement, skipping"
                        )
                        continue
                    rebuild_attempts[shard_id] = attempts + 1
                    reason = self._rebuild_reason(shard, state)
                    logger.info(f"Shard {shard_id} needs rebuild: {reason}")
                    if self._rebuild_index_from_duckdb(shard, conn):
                        changes_made += 1
                    continue

                # Try incremental sync for small deltas
                if self._try_incremental_sync(shard, state, conn):
                    changes_made += 1
                    continue
                elif state.index_live != state.db_count:
                    # Incremental sync failed but mismatch exists - fall back to rebuild
                    logger.warning(
                        f"Incremental sync failed for shard {shard_id}, rebuilding"
                    )
                    if self._rebuild_index_from_duckdb(shard, conn):
                        changes_made += 1
                    continue

                # Structural triggers per spec: empty first, then split, then merge
                # Check for empty shard: remove record and file
                if state.db_count == 0:
                    logger.info(f"Removing empty shard {shard_id}")
                    if self._safe_delete_empty_shard(shard_id, dims, conn):
                        changes_made += 1
                    continue

                # Check for split: db_count >= split_threshold
                if state.db_count >= self.config.split_threshold:
                    if self._split_shard(shard, conn):
                        changes_made += 1
                        continue

                # Check for merge: db_count < merge_threshold and can merge
                if (
                    state.db_count < self.config.merge_threshold
                    and self._can_merge(shard, conn)
                ):
                    if self._merge_shard(shard, conn):
                        changes_made += 1
                        continue

                # Explicit memory cleanup after processing shard
                if self.config.enable_aggressive_gc:
                    gc.collect()
                    logger.debug(f"Memory cleanup after shard {shard_id}")

            if changes_made == 0:
                logger.debug(f"Fix_pass converged after {iteration} iteration(s)")
                break

        if iteration >= max_iterations:
            logger.warning(f"Fix_pass reached max iterations ({max_iterations})")

        # Final: populate centroid cache for routing
        self._heartbeat()
        self._populate_centroid_cache(conn)
        logger.info("Fix_pass complete")

    def _cleanup_orphaned_files(self, conn: Any) -> int:
        """Delete .usearch files without corresponding DB records.

        Args:
            conn: Database connection

        Returns:
            Number of orphaned files deleted
        """
        if not self.shard_dir.exists():
            return 0

        # Get all shard IDs from DB
        db_shard_ids = set()
        for shard in self._list_shards(conn):
            db_shard_ids.add(shard["shard_id"])

        # Find .usearch files without DB records
        deleted = 0
        for usearch_file in self.shard_dir.glob("*.usearch"):
            # Extract shard_id from filename (format: {shard_id}.usearch)
            shard_id_str = usearch_file.stem
            if shard_id_str not in db_shard_ids:
                logger.debug(f"Removing orphaned shard file: {usearch_file}")
                usearch_file.unlink()
                deleted += 1

        return deleted

    def _cleanup_temp_files(self) -> int:
        """Delete .usearch.tmp temporary files.

        Returns:
            Number of temp files deleted
        """
        if not self.shard_dir.exists():
            return 0

        deleted = 0
        for tmp_file in self.shard_dir.glob("*.usearch.tmp"):
            logger.debug(f"Removing temp file: {tmp_file}")
            tmp_file.unlink()
            deleted += 1

        return deleted

    def _safe_delete_empty_shard(
        self, shard_id: UUID, dims: int, conn: Any
    ) -> bool:
        """Atomically delete shard record, file, and cache if no embeddings reference it.

        Uses NOT EXISTS subquery to guard against stale MVCC snapshots that
        may occur after system sleep/wake cycles. The DELETE is atomic within
        a single statement, ensuring consistent snapshot for both check and delete.

        Args:
            shard_id: Shard UUID to delete
            dims: Embedding dimensions (for table name)
            conn: Database connection

        Returns:
            True if shard was deleted, False if embeddings still reference it
        """
        table_name = f"embeddings_{dims}"
        conn.execute(
            f"""
            DELETE FROM vector_shards
            WHERE shard_id = ?
            AND NOT EXISTS (SELECT 1 FROM {table_name} WHERE shard_id = ?)
            """,
            [str(shard_id), str(shard_id)],
        )
        # Check if deletion occurred
        result = conn.execute(
            "SELECT 1 FROM vector_shards WHERE shard_id = ?",
            [str(shard_id)],
        ).fetchone()
        deleted = result is None
        if deleted:
            # Cleanup file and cache (consolidated from call sites)
            file_path = self._shard_path(shard_id)
            if file_path.exists():
                file_path.unlink()
            self.centroids.pop(shard_id, None)
        else:
            logger.warning(
                f"Shard {shard_id} delete skipped: embeddings exist "
                "(possible stale MVCC snapshot)"
            )
        return deleted

    def _needs_rebuild(self, shard: dict[str, Any], state: ShardState) -> bool:
        """Check if shard needs full rebuild.

        Rebuild conditions:
        - index_live != db_count with large delta (>10% of index_live)
        - self_recall < quality_threshold (default 0.95)
        - tombstone_ratio >= compaction_threshold (default 0.20)

        Args:
            shard: Shard record from DB
            state: Current derived shard state

        Returns:
            True if rebuild is required
        """
        # Large count mismatch
        if state.index_live > 0:
            delta = abs(state.index_live - state.db_count)
            if delta / state.index_live > self.config.incremental_sync_threshold:
                return True
        elif state.db_count > 0:
            # Index empty but DB has records
            return True

        # Quality degradation
        if state.self_recall < self.config.quality_threshold:
            return True

        # High tombstone ratio
        if state.tombstone_ratio >= self.config.compaction_threshold:
            return True

        return False

    def _rebuild_reason(self, shard: dict[str, Any], state: ShardState) -> str:
        """Get human-readable reason for rebuild decision."""
        reasons = []

        if state.index_live > 0:
            delta = abs(state.index_live - state.db_count)
            if delta / state.index_live > self.config.incremental_sync_threshold:
                reasons.append(f"count delta {delta} ({delta/state.index_live:.0%})")
        elif state.db_count > 0:
            reasons.append(f"empty index, {state.db_count} in DB")

        if state.self_recall < self.config.quality_threshold:
            reasons.append(f"low recall {state.self_recall:.2f}")

        if state.tombstone_ratio >= self.config.compaction_threshold:
            reasons.append(f"tombstone ratio {state.tombstone_ratio:.0%}")

        return ", ".join(reasons) if reasons else "unknown"

    def _try_incremental_sync(
        self, shard: dict[str, Any], state: ShardState, conn: Any
    ) -> bool:
        """Try to sync shard incrementally by adding/removing individual vectors.

        Only attempts if delta is small (<10% of index_live).

        Args:
            shard: Shard record from DB
            state: Current derived shard state
            conn: Database connection

        Returns:
            True if changes were made
        """
        # Calculate delta
        delta = state.db_count - state.index_live

        if delta == 0:
            return False  # Already in sync - no changes needed

        # Check if delta is small enough for incremental sync
        if state.index_live > 0:
            if abs(delta) / state.index_live > self.config.incremental_sync_threshold:
                return False  # Too large, needs rebuild

        shard_id = UUID(shard["shard_id"])
        dims = shard["dims"]
        file_path = self._shard_path(shard_id)

        # Get current index keys efficiently using slice (avoids O(n) iterator)
        index = usearch_wrapper.open_view(file_path)
        index_keys = set(int(k) for k in cast(np.ndarray, index.keys[:]))

        # Get embedding IDs from DB for this shard
        db_ids = self._get_shard_embedding_ids(shard_id, dims, conn)
        db_keys = set(db_ids.keys())

        # Find differences
        to_add = db_keys - index_keys  # In DB but not in index
        to_remove = index_keys - db_keys  # In index but not in DB

        if not to_add and not to_remove:
            return False

        logger.debug(
            f"Shard {shard_id}: incremental sync "
            f"{len(to_add)} add, {len(to_remove)} remove"
        )

        # Load index for modification (not as view)
        index = usearch_wrapper.open_writable(file_path)

        # Remove deleted vectors (tombstone only, don't compact)
        if to_remove:
            index.remove(list(to_remove), compact=False)

        # Add new vectors (batched for performance)
        if to_add:
            add_keys = np.array(list(to_add), dtype=np.uint64)
            add_vectors = np.array([db_ids[k] for k in to_add], dtype=np.float32)
            index.add(add_keys, add_vectors)

        # Save atomically via temp file with fsync for mmap visibility
        tmp_path = file_path.with_suffix(".usearch.tmp")
        index.save(str(tmp_path))
        fsync_path(tmp_path)  # Flush data before rename
        tmp_path.replace(file_path)

        # Explicit cleanup to release RAM
        del index

        # Force GC after sync to release memory
        if self.config.enable_aggressive_gc:
            gc.collect()

        return True

    def _rebuild_index_from_duckdb(self, shard: dict[str, Any], conn: Any) -> bool:
        """Build fresh USearch index from DuckDB embeddings.

        Args:
            shard: Shard record from DB with shard_id, dims, quantization
            conn: Database connection

        Returns:
            True if index rebuilt or empty shard deleted successfully,
            False if empty shard deletion failed (stale MVCC snapshot)
        """
        shard_id = UUID(shard["shard_id"])
        dims = shard["dims"]
        quantization = shard.get("quantization", self.config.default_quantization)
        file_path = self._shard_path(shard_id)

        self._heartbeat()

        # Get all embeddings - uses fetchall() for immediate materialization
        embeddings = self._get_shard_embedding_ids(shard_id, dims, conn)

        if not embeddings:
            # DuckDB shows shard is empty - delete the shard now
            # (fix_pass can't reach deletion code due to continue after rebuild call)
            logger.info(f"Shard {shard_id} has no embeddings, deleting empty shard")
            return self._safe_delete_empty_shard(shard_id, dims, conn)

        logger.info(f"Rebuilding shard {shard_id} with {len(embeddings)} vectors")

        self._heartbeat()

        # Create new index
        index = usearch_wrapper.create(
            dims,
            quantization,
            connectivity=self.config.hnsw_connectivity,
            expansion_add=self.config.hnsw_expansion_add,
            expansion_search=self.config.hnsw_expansion_search,
        )

        self._heartbeat()

        # Batch add all vectors
        keys = np.array(list(embeddings.keys()), dtype=np.uint64)
        vectors = np.array(list(embeddings.values()), dtype=np.float32)
        index.add(keys, vectors)

        # Ensure shard directory exists
        self.shard_dir.mkdir(parents=True, exist_ok=True)

        self._heartbeat()

        # Save atomically via temp file with fsync for mmap visibility
        tmp_path = file_path.with_suffix(".usearch.tmp")
        index.save(str(tmp_path))
        fsync_path(tmp_path)  # Flush data before rename
        tmp_path.replace(file_path)

        # Explicit cleanup to release RAM
        del index
        del embeddings  # Release the dict

        # Force GC after rebuild
        if self.config.enable_aggressive_gc:
            gc.collect()

        logger.debug(f"Shard {shard_id} rebuilt: {len(keys)} vectors")
        return True  # Rebuild succeeded

    def _populate_centroid_cache(self, conn: Any) -> None:
        """Compute medoid for each shard and cache for routing.

        Args:
            conn: Database connection

        Populates self.centroids with shard_id -> medoid_vector mapping.
        """
        self.centroids.clear()

        for shard in self._list_shards(conn):
            shard_id = UUID(shard["shard_id"])
            file_path = self._shard_path(shard_id)

            if not file_path.exists():
                continue

            try:
                index = usearch_wrapper.open_view(file_path)
                if len(index) == 0:
                    continue

                _, medoid = usearch_wrapper.get_medoid(index)
                self.centroids[shard_id] = medoid
                logger.debug(f"Cached centroid for shard {shard_id}")

            except Exception as e:
                logger.warning(f"Failed to compute centroid for shard {shard_id}: {e}")

        logger.info(f"Populated {len(self.centroids)} shard centroids")

    def _shard_path(self, shard_id: UUID) -> Path:
        """Get file path for shard."""
        return self.shard_dir / f"{shard_id}.usearch"

    def _list_shards(self, conn: Any) -> list[dict[str, Any]]:
        """List all shards from vector_shards table.

        Args:
            conn: Database connection

        Note: file_path is derived via _shard_path(), not stored in DB
              per portability constraint (spec I14: Path Independence)
        """
        result = conn.execute("""
            SELECT shard_id, dims, provider, model, quantization
            FROM vector_shards
        """).fetchall()

        return [
            {
                "shard_id": str(row[0]),
                "dims": row[1],
                "provider": row[2],
                "model": row[3],
                "quantization": row[4] or self.config.default_quantization,
            }
            for row in result
        ]

    def _get_shard_embedding_ids(
        self, shard_id: UUID, dims: int, conn: Any
    ) -> dict[int, list[float]]:
        """Get embedding ID -> vector mapping for a shard.

        Uses fetchall() for immediate materialization to avoid cursor
        snapshot isolation issues when called within active transactions.

        Args:
            shard_id: Shard UUID
            dims: Embedding dimensions (determines table name)
            conn: Database connection

        Returns:
            Dict mapping embedding ID to vector
        """
        table_name = f"embeddings_{dims}"
        result = conn.execute(
            f"SELECT id, embedding FROM {table_name} WHERE shard_id = ? ORDER BY id",
            [str(shard_id)],
        ).fetchall()

        return {row[0]: list(row[1]) for row in result}

    def _can_merge(self, shard: dict[str, Any], conn: Any) -> bool:
        """Check if shard can be merged with another.

        A shard can be merged if there exists at least one other shard
        with the same dims, provider, and model.

        Args:
            shard: Shard record from DB
            conn: Database connection

        Returns:
            True if merge is possible
        """
        # Find other shards with same dims/provider/model
        result = conn.execute(
            """
            SELECT COUNT(*) FROM vector_shards
            WHERE dims = ? AND provider = ? AND model = ? AND shard_id != ?
            """,
            [shard["dims"], shard["provider"], shard["model"], shard["shard_id"]],
        ).fetchone()

        return result[0] > 0 if result else False

    def _kmeans_fallback(
        self, shard: dict[str, Any], conn: Any, n_clusters: int = 2
    ) -> dict[int, int]:
        """Sample-then-assign clustering using sklearn KMeans.

        Instead of loading all embeddings into memory:
        1. Sample ~512 random vectors from DuckDB
        2. Run KMeans on samples to find centroids
        3. Assign ALL vectors via DuckDB SQL using array_cosine_distance

        Args:
            shard: Shard record from DB
            conn: Database connection
            n_clusters: Number of clusters to create

        Returns:
            Dict mapping embedding_id -> cluster_label (0 to n_clusters-1)
        """
        shard_id = UUID(shard["shard_id"])
        dims = shard["dims"]
        table_name = f"embeddings_{dims}"

        self._heartbeat()

        # Phase 1: Sample random vectors for centroid discovery
        # Minimum 50 samples for reliable k=2 clustering (industry: 20-30 per cluster)
        sample_size = min(KMEANS_SAMPLE_SIZE, max(50, self.config.split_threshold // 5))
        samples = conn.execute(
            f"SELECT id, embedding FROM {table_name} "
            f"WHERE shard_id = ? ORDER BY RANDOM() LIMIT ?",
            [str(shard_id), sample_size],
        ).fetchall()

        if not samples:
            return {}

        # Adjust n_clusters if we have fewer samples
        actual_clusters = min(n_clusters, len(samples))
        if actual_clusters < 2:
            # Can't cluster with fewer than 2 points - assign all to cluster 0
            result = conn.execute(
                f"SELECT id FROM {table_name} WHERE shard_id = ?",
                [str(shard_id)],
            ).fetchall()
            return {row[0]: 0 for row in result}

        self._heartbeat()

        # Phase 2: Run KMeans only on samples to find centroids
        vectors = np.array([list(row[1]) for row in samples], dtype=np.float32)
        kmeans = KMeans(n_clusters=actual_clusters, n_init="auto")
        kmeans.fit(vectors)
        centroids = kmeans.cluster_centers_

        self._heartbeat()

        # Phase 3: Assign ALL vectors using DuckDB SQL with array_cosine_distance
        # Build CASE expression to find nearest centroid for each vector
        embedding_type = f"FLOAT[{dims}]"
        centroid_params = [centroids[i].tolist() for i in range(actual_clusters)]

        # For 2 clusters, use simple CASE WHEN
        if actual_clusters == 2:
            query = f"""
                SELECT id,
                    CASE WHEN array_cosine_distance(embedding, ?::{embedding_type})
                              <= array_cosine_distance(embedding, ?::{embedding_type})
                         THEN 0 ELSE 1 END AS cluster_label
                FROM {table_name}
                WHERE shard_id = ?
            """
            params = [centroid_params[0], centroid_params[1], str(shard_id)]
        else:
            # For n clusters, build a nested CASE or use LEAST with index tracking
            # Generate distance columns and find argmin
            dist_exprs = [
                f"array_cosine_distance(embedding, ?::{embedding_type}) AS d{i}"
                for i in range(actual_clusters)
            ]
            case_parts = []
            for i in range(actual_clusters):
                conditions = [f"d{i} <= d{j}" for j in range(actual_clusters) if j != i]
                case_parts.append(f"WHEN {' AND '.join(conditions)} THEN {i}")

            query = f"""
                SELECT id,
                    CASE {' '.join(case_parts)} END AS cluster_label
                FROM (
                    SELECT id, embedding, {', '.join(dist_exprs)}
                    FROM {table_name}
                    WHERE shard_id = ?
                ) sub
            """
            params = centroid_params + [str(shard_id)]

        result = conn.execute(query, params).fetchall()
        return {row[0]: row[1] for row in result}

    def _split_shard(self, shard: dict[str, Any], conn: Any) -> bool:
        """Split a shard when db_count >= split_threshold.

        Uses K-means clustering to create exactly 2 balanced child shards.

        DuckDB transaction:
        - Create child shards in vector_shards
        - Reassign embeddings shard_id to children
        - Delete parent shard record

        Child indexes are built by convergence loop (fix_pass will handle).

        Args:
            shard: Shard record from DB
            conn: Database connection

        Returns:
            True if split was performed
        """
        shard_id = UUID(shard["shard_id"])
        dims = shard["dims"]
        file_path = self._shard_path(shard_id)

        logger.info(f"Splitting shard {shard_id}")
        self._heartbeat()

        # Always use K-means for splitting - guarantees exactly 2 balanced clusters.
        # USearch's native clustering uses HNSW graph structure which does NOT
        # guarantee min_count/max_count constraints, leading to many small clusters
        # that trigger merge operations and cause infinite split->merge cycles.
        cluster_assignments = self._kmeans_fallback(shard, conn, n_clusters=2)

        if not cluster_assignments:
            logger.warning(f"No embeddings to split for shard {shard_id}")
            return False

        # Group embedding_ids by cluster label
        clusters: dict[int, list[int]] = {}
        for emb_id, label in cluster_assignments.items():
            clusters.setdefault(label, []).append(emb_id)

        # Validate cluster sizes to prevent immediate merge cycles
        min_cluster_size = min(len(ids) for ids in clusters.values()) if clusters else 0
        if min_cluster_size < self.config.merge_threshold:
            logger.warning(
                f"Shard {shard_id} split would create cluster with {min_cluster_size} "
                f"vectors (< merge_threshold {self.config.merge_threshold}), skipping"
            )
            return False

        # Need at least 2 clusters to split
        if len(clusters) < 2:
            logger.debug(f"Shard {shard_id} cannot be split into multiple clusters")
            return False

        # Two-phase split: DuckDB FK checks use committed state, so we must commit
        # the UPDATE before DELETE will succeed. If phase 2 fails, fix_pass cleans up.
        self._heartbeat()

        try:
            table_name = f"embeddings_{dims}"

            # Phase 1 (transactional): Create children + reassign embeddings
            with _transaction(conn):
                # Create child shards (file_path derived at runtime per spec I14)
                child_ids: list[UUID] = []
                for cluster_label in sorted(clusters.keys()):
                    child_id = uuid4()
                    child_ids.append(child_id)

                    conn.execute(
                        """
                        INSERT INTO vector_shards
                            (shard_id, dims, provider, model, quantization)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        [
                            str(child_id),
                            dims,
                            shard["provider"],
                            shard["model"],
                            shard.get("quantization", self.config.default_quantization),
                        ],
                    )

                # Reassign embeddings to child shards
                for cluster_label, emb_ids in clusters.items():
                    child_id = child_ids[cluster_label]
                    # Batch update for efficiency
                    placeholders = ",".join(["?"] * len(emb_ids))
                    conn.execute(
                        f"""
                        UPDATE {table_name}
                        SET shard_id = ?
                        WHERE id IN ({placeholders})
                        """,
                        [str(child_id)] + emb_ids,
                    )
                    self._heartbeat()

            # Phase 2 (post-commit): Delete parent shard now that FK references are gone
            conn.execute(
                "DELETE FROM vector_shards WHERE shard_id = ?",
                [str(shard_id)],
            )

            logger.info(
                f"Split shard {shard_id} into {len(child_ids)} children: "
                f"{[str(cid)[:8] for cid in child_ids]}"
            )

            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Removed parent shard file: {file_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to split shard {shard_id}: {e}")
            return False

    def _merge_shard(self, shard: dict[str, Any], conn: Any) -> bool:
        """Merge a shard when db_count < merge_threshold.

        Uses per-vector routing: each embedding is assigned to its nearest
        target centroid among top-N candidates (NPA: Nearest Partition Assignment).
        Target indexes are rebuilt by convergence loop.

        DuckDB transaction:
        - UPDATE embeddings SET shard_id = target (batched by target)
        - DELETE source from vector_shards

        Args:
            shard: Shard record from DB (source to be merged)
            conn: Database connection

        Returns:
            True if merge was performed
        """
        shard_id = UUID(shard["shard_id"])
        dims = shard["dims"]

        logger.info(f"Merging shard {shard_id}")
        self._heartbeat()

        # Get compatible shards (same dims/provider/model, different shard_id)
        result = conn.execute(
            """
            SELECT shard_id FROM vector_shards
            WHERE dims = ? AND provider = ? AND model = ? AND shard_id != ?
            """,
            [shard["dims"], shard["provider"], shard["model"], shard["shard_id"]],
        ).fetchall()

        if not result:
            logger.debug(f"No compatible shards to merge with for {shard_id}")
            return False

        # Build list of candidate targets with their centroids
        # Compute centroids on-demand if not in cache (during fix_pass)
        candidates: list[tuple[UUID, np.ndarray]] = []
        for row in result:
            target_id = _parse_uuid(row[0])
            target_centroid = self.centroids.get(target_id)

            # If not in cache, try to compute from index file
            if target_centroid is None:
                target_path = self._shard_path(target_id)
                if target_path.exists():
                    try:
                        target_index = usearch_wrapper.open_view(target_path)
                        if len(target_index) > 0:
                            _, centroid = usearch_wrapper.get_medoid(target_index)
                            target_centroid = centroid
                            self.centroids[target_id] = centroid  # Cache
                    except Exception as e:
                        logger.debug(f"Could not get centroid for {target_id}: {e}")

            if target_centroid is not None:
                candidates.append((target_id, target_centroid))

        if not candidates:
            # Fallback: no centroids available, bulk assign to first target
            first_id = result[0][0]
            fallback = _parse_uuid(first_id)
            logger.warning(f"No centroids for merge targets, fallback: {fallback}")
            candidates = [(fallback, np.zeros(dims))]

        # Get source shard centroid for sorting candidates
        source_centroid = self.centroids.get(shard_id)
        if source_centroid is None:
            file_path = self._shard_path(shard_id)
            if file_path.exists():
                try:
                    index = usearch_wrapper.open_view(file_path)
                    if len(index) > 0:
                        _, source_centroid = usearch_wrapper.get_medoid(index)
                except Exception as e:
                    logger.warning(f"Failed to get centroid for merge source: {e}")

        # Sort candidates by distance to source centroid, take top N
        if source_centroid is not None:
            candidates.sort(
                key=lambda t: 1.0 - np.dot(source_centroid, t[1]) / (
                    np.linalg.norm(source_centroid) * np.linalg.norm(t[1]) + 1e-9
                )
            )
        targets = candidates[: self.config.merge_target_count]

        # Get all embeddings from source shard
        embeddings = self._get_shard_embedding_ids(shard_id, dims, conn)
        if not embeddings:
            # Query snapshot shows empty - skip merge, deletion handled by fix_pass
            # based on authoritative db_count state (not query snapshots)
            logger.info(f"Shard {shard_id} has no embeddings in query, skipping merge")
            return False

        # Per-vector routing: assign each embedding to nearest target centroid
        self._heartbeat()
        assignments: dict[UUID, list[int]] = {t[0]: [] for t in targets}
        routed_count = 0
        for emb_id, vector in embeddings.items():
            vec_array = np.array(vector, dtype=np.float32)
            vec_norm = np.linalg.norm(vec_array)

            # Find nearest target by cosine distance
            best_target = targets[0][0]
            best_distance = float("inf")

            for target_id, target_centroid in targets:
                target_norm = np.linalg.norm(target_centroid)
                if vec_norm > 0 and target_norm > 0:
                    distance = 1.0 - np.dot(vec_array, target_centroid) / (
                        vec_norm * target_norm
                    )
                else:
                    distance = 1.0

                if distance < best_distance:
                    best_distance = distance
                    best_target = target_id

            assignments[best_target].append(emb_id)
            routed_count += 1
            # Heartbeat every 10,000 vectors during bulk merge.
            # Merge processes entire shard (typically 10K-100K vectors),
            # so less frequent heartbeats (operation is memory-bound).
            if routed_count % HEARTBEAT_INTERVAL_MERGE_ROUTING == 0:
                self._heartbeat()

        # Two-phase merge: DuckDB FK checks use committed state, so we must commit
        # the UPDATE before DELETE will succeed. If phase 2 fails, fix_pass cleans up.
        self._heartbeat()

        try:
            table_name = f"embeddings_{dims}"

            # Phase 1 (transactional): Reassign embeddings to targets
            with _transaction(conn):
                for target_id, emb_ids in assignments.items():
                    if not emb_ids:
                        continue

                    placeholders = ",".join(["?" for _ in emb_ids])
                    conn.execute(
                        f"""
                        UPDATE {table_name}
                        SET shard_id = ?
                        WHERE id IN ({placeholders})
                        """,
                        [str(target_id), *emb_ids],
                    )
                    self._heartbeat()
                    logger.debug(
                        f"Routed {len(emb_ids)} embeddings to target {target_id}"
                    )

            # Phase 2 (post-commit): Delete source shard now that FK references are gone
            conn.execute(
                "DELETE FROM vector_shards WHERE shard_id = ?",
                [str(shard_id)],
            )

            summary = {str(t)[:8]: len(ids) for t, ids in assignments.items() if ids}
            logger.info(f"Merged shard {shard_id} -> {summary}")

            file_path = self._shard_path(shard_id)
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Removed merged shard file: {file_path}")

            # Remove source from centroids cache
            self.centroids.pop(shard_id, None)

            return True

        except Exception as e:
            logger.error(f"Failed to merge shard {shard_id}: {e}")
            return False
