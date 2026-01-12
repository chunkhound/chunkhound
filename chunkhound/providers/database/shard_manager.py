"""Shard lifecycle manager for ChunkHound.

Coordinates shard operations including search, insert, delete, and maintenance.
Uses derived state architecture - metrics computed from DuckDB and USearch files.
"""

from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans  # type: ignore[import-untyped]

from chunkhound.core.config.sharding_config import ShardingConfig
from chunkhound.providers.database import usearch_wrapper
from chunkhound.providers.database.shard_state import ShardState, get_shard_state
from chunkhound.providers.database.usearch_wrapper import SearchResult
from chunkhound.utils.windows_constants import fsync_path


def _parse_uuid(value: Any) -> UUID:
    """Parse UUID from DuckDB result (handles both UUID objects and strings)."""
    return value if isinstance(value, UUID) else UUID(value)


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
    ) -> None:
        """Initialize shard manager.

        Args:
            db_provider: Database provider for DuckDB operations
            shard_dir: Directory containing .usearch shard files
            config: Sharding configuration with thresholds
        """
        self.db = db_provider
        self.shard_dir = shard_dir
        self.config = config
        self.centroids: dict[UUID, np.ndarray] = {}

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
            SELECT shard_id FROM vector_shards
            WHERE dims = ? AND provider = ? AND model = ?
            """,
            [dims, provider, model],
        ).fetchall()

        if not result:
            return []

        # Filter shards by centroid similarity
        relevant_shards: list[tuple[UUID, Path]] = []
        for row in result:
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

        # Batch search across relevant shards (limit concurrent)
        all_results: list[SearchResult] = []
        max_concurrent = self.config.max_concurrent_shards

        for i in range(0, len(relevant_shards), max_concurrent):
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
        """Insert embeddings into appropriate shards.

        Routes embeddings to shards based on centroid similarity.
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
            # Find or create shard for these embeddings
            shard_result = self._find_or_create_shard(
                dims, provider, model, conn, embeddings
            )
            if shard_result is None:
                return (False, False)

            target_shard_id, was_created = shard_result

            # Insert embeddings into DuckDB with shard_id
            table_name = f"embeddings_{dims}"

            # Batch update shard_id for all embeddings with IDs
            emb_ids = [emb.get("id") for emb in embeddings if emb.get("id") is not None]
            if emb_ids:
                placeholders = ",".join(["?"] * len(emb_ids))
                query = (
                    f"UPDATE {table_name} SET shard_id = ? "
                    f"WHERE id IN ({placeholders})"
                )
                conn.execute(query, [str(target_shard_id)] + emb_ids)

            # Check if split threshold exceeded
            post_count = self._get_shard_count(target_shard_id, dims, conn)

            # Check if shard file exists - triggers fix_pass for file-less shards
            file_path = self._shard_path(target_shard_id)
            file_missing = not file_path.exists()

            exceeds_split = post_count >= self.config.split_threshold
            needs_fix_pass = was_created or file_missing or exceeds_split

            if needs_fix_pass:
                if was_created:
                    logger.debug(
                        f"New shard {target_shard_id} created, needs fix pass"
                    )
                elif file_missing:
                    logger.debug(
                        f"Shard {target_shard_id} missing file, needs fix pass"
                    )
                else:
                    logger.debug(
                        f"Shard {target_shard_id} has {post_count} embeddings, "
                        f"exceeds split threshold {self.config.split_threshold}"
                    )

            return (True, needs_fix_pass)

        except Exception as e:
            logger.error(f"Failed to insert embeddings: {e}")
            return (False, False)

    def _find_or_create_shard(
        self,
        dims: int,
        provider: str,
        model: str,
        conn: Any,
        embeddings: list[dict] | None = None,
    ) -> tuple[UUID, bool] | None:
        """Find existing shard or create new one.

        Uses centroid-based routing to find best shard.
        Creates new shard if none exist for dims/provider/model.

        Args:
            dims: Embedding dimensions
            provider: Embedding provider name
            model: Embedding model name
            conn: Database connection
            embeddings: Optional list of embedding dicts for centroid routing

        Returns:
            Tuple of (shard_id, was_created) or None if creation failed.
            was_created is True if a new shard was created.
        """
        # Find existing shards for this dims/provider/model
        result = conn.execute(
            """
            SELECT shard_id FROM vector_shards
            WHERE dims = ? AND provider = ? AND model = ?
            """,
            [dims, provider, model],
        ).fetchall()

        if result:
            # Single shard or no centroid routing data - return first
            if len(result) == 1 or not embeddings or not self.centroids:
                first_id = _parse_uuid(result[0][0])
                return (first_id, False)

            # Multiple shards: use centroid routing
            # Compute batch centroid (mean of input embeddings)
            vectors = [emb["embedding"] for emb in embeddings if "embedding" in emb]
            if not vectors:
                first_id = _parse_uuid(result[0][0])
                return (first_id, False)

            batch_centroid = np.mean(vectors, axis=0, dtype=np.float32)
            batch_norm = np.linalg.norm(batch_centroid)
            if batch_norm == 0:
                first_id = _parse_uuid(result[0][0])
                return (first_id, False)

            # Find shard with highest centroid similarity
            best_shard_id: UUID | None = None
            best_similarity = -float("inf")

            for row in result:
                shard_id = _parse_uuid(row[0])
                centroid = self.centroids.get(shard_id)

                if centroid is None:
                    continue

                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm == 0:
                    continue

                similarity = np.dot(batch_centroid, centroid) / (
                    batch_norm * centroid_norm
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_shard_id = shard_id

            # Return best match or fallback to first shard
            if best_shard_id is not None:
                return (best_shard_id, False)
            first_id = _parse_uuid(result[0][0])
            return (first_id, False)

        # Create new shard (file_path derived at runtime per spec I14)
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
            return (new_shard_id, True)

        except Exception as e:
            logger.error(f"Failed to create shard: {e}")
            return None

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

        Hard DELETE from DuckDB (source of truth).
        Fix pass will detect index_live != db_count and rebuild.

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

            # Check remaining shard state
            needs_fix_pass = False
            if shard_id is not None:
                db_count = self._get_shard_count(shard_id, dims, conn)
                # Mark needs_fix_pass if below merge threshold or empty
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

            # Get all shards from DB
            shards = self._list_shards(conn)
            if not shards:
                logger.debug("No shards found, skipping fix_pass")
                break

            for shard in shards:
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
                    self._rebuild_index_from_duckdb(shard, conn)
                    changes_made += 1
                    continue
                except (ValueError, RuntimeError) as e:
                    # Corrupted file requires rebuild
                    logger.warning(f"Shard {shard_id} corrupted: {e}, rebuilding")
                    if file_path.exists():
                        file_path.unlink()
                    self._rebuild_index_from_duckdb(shard, conn)
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
                    self._rebuild_index_from_duckdb(shard, conn)
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
                    self._rebuild_index_from_duckdb(shard, conn)
                    changes_made += 1
                    continue

                # Structural triggers per spec: empty first, then split, then merge
                # Check for empty shard: remove record and file
                if state.db_count == 0:
                    logger.info(f"Removing empty shard {shard_id}")
                    conn.execute(
                        "DELETE FROM vector_shards WHERE shard_id = ?",
                        [str(shard_id)],
                    )
                    if file_path.exists():
                        file_path.unlink()
                    self.centroids.pop(shard_id, None)
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

            if changes_made == 0:
                logger.debug(f"Fix_pass converged after {iteration} iteration(s)")
                break

        if iteration >= max_iterations:
            logger.warning(f"Fix_pass reached max iterations ({max_iterations})")

        # Final: populate centroid cache for routing
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

        return True

    def _rebuild_index_from_duckdb(self, shard: dict[str, Any], conn: Any) -> None:
        """Build fresh USearch index from DuckDB embeddings.

        Args:
            shard: Shard record from DB with shard_id, dims, quantization
            conn: Database connection
        """
        shard_id = UUID(shard["shard_id"])
        dims = shard["dims"]
        quantization = shard.get("quantization", self.config.default_quantization)
        file_path = self._shard_path(shard_id)

        # Get all embeddings for this shard from DB
        embeddings = self._get_shard_embedding_ids(shard_id, dims, conn)

        if not embeddings:
            logger.info(f"Shard {shard_id} has no embeddings, removing empty shard")
            # Remove empty shard file if exists
            if file_path.exists():
                file_path.unlink()
            # Delete the shard record from DB
            conn.execute(
                "DELETE FROM vector_shards WHERE shard_id = ?",
                [str(shard_id)],
            )
            self.centroids.pop(shard_id, None)
            return

        logger.info(f"Rebuilding shard {shard_id} with {len(embeddings)} vectors")

        # Create new index
        index = usearch_wrapper.create(
            dims,
            quantization,
            connectivity=self.config.hnsw_connectivity,
            expansion_add=self.config.hnsw_expansion_add,
            expansion_search=self.config.hnsw_expansion_search,
        )

        # Batch add all vectors
        keys = np.array(list(embeddings.keys()), dtype=np.uint64)
        vectors = np.array(list(embeddings.values()), dtype=np.float32)
        index.add(keys, vectors)

        # Ensure shard directory exists
        self.shard_dir.mkdir(parents=True, exist_ok=True)

        # Save atomically via temp file with fsync for mmap visibility
        tmp_path = file_path.with_suffix(".usearch.tmp")
        index.save(str(tmp_path))
        fsync_path(tmp_path)  # Flush data before rename
        tmp_path.replace(file_path)

        logger.debug(f"Shard {shard_id} rebuilt: {len(embeddings)} vectors")

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
        """Fallback clustering using sklearn KMeans when USearch file unavailable.

        Queries embeddings from DuckDB and clusters using KMeans.

        Args:
            shard: Shard record from DB
            conn: Database connection
            n_clusters: Number of clusters to create

        Returns:
            Dict mapping embedding_id -> cluster_label (0 to n_clusters-1)
        """
        shard_id = UUID(shard["shard_id"])
        dims = shard["dims"]

        # Get embeddings from DB
        embeddings = self._get_shard_embedding_ids(shard_id, dims, conn)
        if not embeddings:
            return {}

        # Prepare data for KMeans
        embedding_ids = list(embeddings.keys())
        vectors = np.array(list(embeddings.values()), dtype=np.float32)

        # Run KMeans
        kmeans = KMeans(n_clusters=min(n_clusters, len(embedding_ids)), n_init="auto")
        labels = kmeans.fit_predict(vectors)

        return dict(zip(embedding_ids, labels))

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

        # Transaction: create child shards, reassign embeddings, delete parent
        try:
            table_name = f"embeddings_{dims}"

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

            # Delete parent shard record
            conn.execute(
                "DELETE FROM vector_shards WHERE shard_id = ?",
                [str(shard_id)],
            )

            logger.info(
                f"Split shard {shard_id} into {len(child_ids)} children: "
                f"{[str(cid)[:8] for cid in child_ids]}"
            )

            # Remove parent shard file if exists
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
            # Empty shard - just delete the record
            conn.execute(
                "DELETE FROM vector_shards WHERE shard_id = ?", [str(shard_id)]
            )
            file_path = self._shard_path(shard_id)
            if file_path.exists():
                file_path.unlink()
            self.centroids.pop(shard_id, None)
            logger.info(f"Removed empty shard {shard_id}")
            return True

        # Per-vector routing: assign each embedding to nearest target centroid
        assignments: dict[UUID, list[int]] = {t[0]: [] for t in targets}
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

        # Transaction: batch update by target, delete source shard
        try:
            table_name = f"embeddings_{dims}"

            # Batch update embeddings per target shard
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
                logger.debug(f"Routed {len(emb_ids)} embeddings to target {target_id}")

            # Delete source shard record
            conn.execute(
                "DELETE FROM vector_shards WHERE shard_id = ?",
                [str(shard_id)],
            )

            summary = {str(t)[:8]: len(ids) for t, ids in assignments.items() if ids}
            logger.info(f"Merged shard {shard_id} -> {summary}")

            # Remove source shard file if exists
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
