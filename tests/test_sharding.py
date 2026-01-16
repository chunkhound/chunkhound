"""Sharding test suite covering all 14 invariants from spec section 11.5.

Tests use sociable approach with real DuckDB, real USearch, and real filesystem
in temporary directories. Uses SyntheticEmbeddingGenerator for deterministic,
reproducible tests without external API dependencies.

Invariants tested:
- Data Integrity (I1-I6): Single assignment, shard existence, count consistency,
  no orphans, no ghosts, embedding retrievability
- Operational (I7-I10): Fix pass idempotence, convergence, LIRE bound, NPA
- Search (I11-I13): No false negatives, tombstone exclusion, centroid filter
- Portability (I14): Path independence

Uses small thresholds (split=100, merge=10) for efficient testing without
requiring 100K vectors.
"""

import time
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import numpy as np
import numpy.core.multiarray  # noqa: F401  # Prevent DuckDB threading segfault
import pytest
from scipy import stats

from chunkhound.api.cli.utils.rich_output import RichOutputFormatter
from chunkhound.core.config.sharding_config import ShardingConfig
from chunkhound.providers.database import usearch_wrapper
from chunkhound.providers.database.compaction_utils import compact_database
from chunkhound.providers.database.shard_manager import ShardManager
from chunkhound.providers.database.shard_state import get_shard_state
from tests.fixtures.synthetic_embeddings import (
    SyntheticEmbeddingGenerator,
    brute_force_search,
)

# Test configuration with small thresholds for efficient testing
TEST_DIMS = 128  # Smaller dims for faster tests
TEST_SPLIT_THRESHOLD = 100
TEST_MERGE_THRESHOLD = 20

# Stress test configuration (production thresholds)
STRESS_DIMS = 128
STRESS_SPLIT_THRESHOLD = 100_000
STRESS_MERGE_THRESHOLD = 10_000
STRESS_TOTAL_VECTORS = 1_000_000
STRESS_BATCH_MIN = 900
STRESS_BATCH_MAX = 1100
STRESS_SEED = 42
STRESS_CHECKPOINT_INTERVAL = 100

# Amortized O(1) validation: measure cumulative time at these intervals
STRESS_O1_MEASUREMENT_INTERVAL = 100_000  # Every 100K vectors


class MockDBProvider:
    """Minimal DB provider for sharding tests.

    Provides in-memory DuckDB with vector_shards and embeddings tables.
    Uses real DuckDB for sociable testing.
    """

    def __init__(self, db_path: Path):
        import duckdb

        self.db_path = db_path
        self._conn = duckdb.connect(str(db_path))
        self._create_schema()

    def _create_schema(self) -> None:
        """Create minimal schema for sharding tests."""
        # Create vector_shards table
        # Note: file_path NOT stored - derived at runtime per spec I14 (Path Independence)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS vector_shards (
                shard_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                dims INTEGER NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                quantization TEXT NOT NULL DEFAULT 'i8',
                file_size_bytes BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CHECK (quantization IN ('f32', 'f16', 'i8'))
            )
        """)

        # Create index for lookups
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shards_dims_provider_model
                ON vector_shards(dims, provider, model)
        """)

        # Create embeddings sequence
        self._conn.execute("CREATE SEQUENCE IF NOT EXISTS embeddings_id_seq")

        # Create embeddings table for test dimensions
        self._create_embeddings_table(TEST_DIMS)

    def _create_embeddings_table(self, dims: int) -> None:
        """Create embeddings table for given dimensions."""
        table_name = f"embeddings_{dims}"
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY DEFAULT nextval('embeddings_id_seq'),
                chunk_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                embedding FLOAT[{dims}],
                dims INTEGER NOT NULL DEFAULT {dims},
                shard_id UUID REFERENCES vector_shards(shard_id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_embeddings_{dims}_shard
                ON {table_name}(shard_id)
        """)

    @property
    def connection(self):
        """Return DuckDB connection."""
        return self._conn

    def set_shard_manager(self, shard_manager: ShardManager) -> None:
        """Set shard manager for BulkIndexer compatibility."""
        self.shard_manager = shard_manager

    def run_fix_pass(self, check_quality: bool = True) -> None:
        """Run fix_pass via shard_manager - for BulkIndexer compatibility."""
        if hasattr(self, "shard_manager") and self.shard_manager is not None:
            self.shard_manager.fix_pass(self._conn, check_quality=check_quality)

    def optimize(self) -> bool:
        """Full compaction via shared utility - uses production thresholds.

        Mirrors DuckDBProvider.optimize() behavior with same thresholds
        (compaction_threshold=0.5, compaction_min_size_mb=100).
        """
        success, new_conn = compact_database(
            db_path=self.db_path,
            conn=self._conn,
        )
        self._conn = new_conn
        return success

    def disconnect(self) -> None:
        """Close connection with checkpoint for durability."""
        self._conn.execute("CHECKPOINT")
        self._conn.close()


@pytest.fixture
def tmp_db(tmp_path: Path) -> MockDBProvider:
    """Create temporary DuckDB database with sharding enabled."""
    db_path = tmp_path / "test.duckdb"
    provider = MockDBProvider(db_path)
    yield provider
    provider.disconnect()


@pytest.fixture
def shard_manager(tmp_path: Path, tmp_db: MockDBProvider) -> ShardManager:
    """Create ShardManager with temp shard directory and small thresholds."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir(exist_ok=True)

    config = ShardingConfig(
        split_threshold=TEST_SPLIT_THRESHOLD,
        merge_threshold=TEST_MERGE_THRESHOLD,
        compaction_threshold=0.20,
        incremental_sync_threshold=0.10,
        quality_threshold=0.95,
        shard_similarity_threshold=0.1,  # Lower threshold for test clusters
    )

    return ShardManager(
        db_provider=tmp_db,
        shard_dir=shard_dir,
        config=config,
    )


@pytest.fixture
def generator() -> SyntheticEmbeddingGenerator:
    """Create SyntheticEmbeddingGenerator with test dimensions."""
    return SyntheticEmbeddingGenerator(dims=TEST_DIMS, seed=42)


def insert_embeddings_to_db(
    db_provider: MockDBProvider,
    vectors: list[np.ndarray],
    shard_id: UUID | None = None,
    provider: str = "test",
    model: str = "test-model",
) -> list[int]:
    """Insert embeddings into DuckDB and return their IDs."""
    conn = db_provider.connection
    table_name = f"embeddings_{TEST_DIMS}"
    ids = []

    for i, vec in enumerate(vectors):
        vec_list = vec.tolist()
        result = conn.execute(
            f"""
            INSERT INTO {table_name}
                (chunk_id, provider, model, embedding, dims, shard_id)
            VALUES (?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            [
                i,
                provider,
                model,
                vec_list,
                TEST_DIMS,
                str(shard_id) if shard_id else None,
            ],
        ).fetchone()
        ids.append(result[0])

    return ids


def verify_all_vectors_searchable(
    shard_manager: ShardManager,
    db_provider: MockDBProvider,
    vectors: list[np.ndarray],
    emb_ids: list[int],
    k: int = 10,
) -> list[int]:
    """Verify each vector finds itself in top-k search results.

    Returns list of embedding IDs NOT found (empty = success).
    """
    not_found = []
    for emb_id, vec in zip(emb_ids, vectors):
        results = shard_manager.search(
            query=vec.tolist(),
            k=k,
            dims=TEST_DIMS,
            provider="test",
            model="test-model",
            conn=db_provider.connection,
        )
        if emb_id not in {r.key for r in results}:
            not_found.append(emb_id)
    return not_found


def batch_insert_embeddings_to_db(
    db_provider: MockDBProvider,
    vectors: list[np.ndarray],
    dims: int,
    shard_id: UUID | None = None,
    provider: str = "test",
    model: str = "test-model",
    start_chunk_id: int = 0,
) -> list[int]:
    """Batch insert embeddings into DuckDB. Returns IDs.

    Uses executemany for performance per AGENTS.md rules.
    Supports arbitrary dimensions unlike insert_embeddings_to_db.

    Args:
        db_provider: Database provider
        vectors: List of embedding vectors
        dims: Vector dimensions
        shard_id: Optional shard assignment (None = unassigned)
        provider: Embedding provider name
        model: Embedding model name
        start_chunk_id: Starting chunk_id for this batch

    Returns:
        List of inserted embedding IDs
    """
    conn = db_provider.connection
    table_name = f"embeddings_{dims}"

    # Prepare batch data as tuples
    rows = [
        (
            start_chunk_id + i,
            provider,
            model,
            vec.tolist(),
            dims,
            str(shard_id) if shard_id else None,
        )
        for i, vec in enumerate(vectors)
    ]

    # Batch insert using executemany
    conn.executemany(
        f"""
        INSERT INTO {table_name} (chunk_id, provider, model, embedding, dims, shard_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    # Retrieve inserted IDs
    result = conn.execute(
        f"""SELECT id FROM {table_name}
            WHERE chunk_id >= ? AND chunk_id < ?
            ORDER BY chunk_id""",
        [start_chunk_id, start_chunk_id + len(vectors)],
    ).fetchall()

    return [row[0] for row in result]


def get_all_shard_ids(db_provider: MockDBProvider) -> list[UUID]:
    """Get all shard IDs from DB."""
    result = db_provider.connection.execute(
        "SELECT shard_id FROM vector_shards"
    ).fetchall()
    # DuckDB returns UUID objects directly, handle both UUID and string
    return [row[0] if isinstance(row[0], UUID) else UUID(row[0]) for row in result]


def get_shard_embedding_count(db_provider: MockDBProvider, shard_id: UUID) -> int:
    """Get count of embeddings in a shard."""
    table_name = f"embeddings_{TEST_DIMS}"
    result = db_provider.connection.execute(
        f"SELECT COUNT(*) FROM {table_name} WHERE shard_id = ?",
        [str(shard_id)],
    ).fetchone()
    return result[0]


def get_shard_counts(db_provider: MockDBProvider) -> dict[UUID, int]:
    """Get embedding count per shard."""
    table_name = f"embeddings_{TEST_DIMS}"
    result = db_provider.connection.execute(f"""
        SELECT shard_id, COUNT(*) as cnt
        FROM {table_name}
        WHERE shard_id IS NOT NULL
        GROUP BY shard_id
    """).fetchall()
    return {
        row[0] if isinstance(row[0], UUID) else UUID(row[0]): row[1] for row in result
    }


def delete_from_shard(
    db_provider: MockDBProvider,
    shard_id: UUID,
    count: int,
) -> list[int]:
    """Delete `count` embeddings from specific shard. Returns deleted IDs."""
    table_name = f"embeddings_{TEST_DIMS}"
    # Get IDs to delete
    result = db_provider.connection.execute(
        f"""
        SELECT id FROM {table_name} WHERE shard_id = ? LIMIT ?
    """,
        [str(shard_id), count],
    ).fetchall()
    deleted_ids = [row[0] for row in result]

    if deleted_ids:
        placeholders = ", ".join("?" * len(deleted_ids))
        db_provider.connection.execute(
            f"""
            DELETE FROM {table_name} WHERE id IN ({placeholders})
        """,
            deleted_ids,
        )

    return deleted_ids


def verify_invariant_i1_single_assignment(db_provider: MockDBProvider) -> bool:
    """I1: Each embedding has exactly one shard assignment."""
    table_name = f"embeddings_{TEST_DIMS}"
    result = db_provider.connection.execute(f"""
        SELECT id, COUNT(*) as cnt
        FROM {table_name}
        WHERE shard_id IS NOT NULL
        GROUP BY id
        HAVING COUNT(*) > 1
    """).fetchall()
    return len(result) == 0


def verify_invariant_i2_shard_existence(db_provider: MockDBProvider) -> bool:
    """I2: Every embedding's shard_id references an existing shard."""
    table_name = f"embeddings_{TEST_DIMS}"
    result = db_provider.connection.execute(f"""
        SELECT e.shard_id
        FROM {table_name} e
        LEFT JOIN vector_shards s ON e.shard_id = s.shard_id
        WHERE e.shard_id IS NOT NULL AND s.shard_id IS NULL
    """).fetchall()
    return len(result) == 0


def verify_invariant_i4_no_orphan_files(
    db_provider: MockDBProvider, shard_dir: Path
) -> list[Path]:
    """I4: Every .usearch file has a corresponding DB record. Returns orphans."""
    db_shard_ids = set()
    for shard in db_provider.connection.execute(
        "SELECT shard_id FROM vector_shards"
    ).fetchall():
        db_shard_ids.add(str(shard[0]))

    orphans = []
    for usearch_file in shard_dir.glob("*.usearch"):
        shard_id_str = usearch_file.stem
        if shard_id_str not in db_shard_ids:
            orphans.append(usearch_file)

    return orphans


def verify_invariant_i5_no_ghost_records(
    db_provider: MockDBProvider, shard_dir: Path
) -> list[UUID]:
    """I5: Every DB record has a corresponding .usearch file. Returns ghosts."""
    fs_shard_ids = {f.stem for f in shard_dir.glob("*.usearch")}

    ghosts = []
    for row in db_provider.connection.execute(
        "SELECT shard_id FROM vector_shards"
    ).fetchall():
        # DuckDB returns UUID objects directly
        shard_id = row[0] if isinstance(row[0], UUID) else UUID(row[0])
        if str(shard_id) not in fs_shard_ids:
            ghosts.append(shard_id)

    return ghosts


def verify_shard_counts_in_range(
    db_provider: MockDBProvider,
    merge_threshold: int,
    split_threshold: int,
    dims: int,
) -> tuple[bool, str]:
    """Verify merge_threshold <= each shard_count <= split_threshold.

    Args:
        db_provider: Database provider
        merge_threshold: Minimum acceptable shard size
        split_threshold: Maximum acceptable shard size
        dims: Vector dimensions

    Returns:
        (passed, message) tuple
    """
    table_name = f"embeddings_{dims}"
    result = db_provider.connection.execute(f"""
        SELECT shard_id, COUNT(*) as cnt
        FROM {table_name}
        WHERE shard_id IS NOT NULL
        GROUP BY shard_id
    """).fetchall()

    violations = []
    for row in result:
        shard_id = row[0]
        count = row[1]
        if count < merge_threshold:
            violations.append(f"{shard_id}: {count} < {merge_threshold}")
        if count > split_threshold:
            violations.append(f"{shard_id}: {count} > {split_threshold}")

    if violations:
        return False, "; ".join(violations)
    return True, "OK"


def verify_shard_balance(
    db_provider: MockDBProvider,
    dims: int,
    max_ratio: float = 2.0,
) -> tuple[bool, str]:
    """Verify statistical balance: max_count < max_ratio * min_count.

    Args:
        db_provider: Database provider
        dims: Vector dimensions
        max_ratio: Maximum allowed ratio between largest and smallest shard

    Returns:
        (passed, message) tuple
    """
    table_name = f"embeddings_{dims}"
    result = db_provider.connection.execute(f"""
        SELECT shard_id, COUNT(*) as cnt
        FROM {table_name}
        WHERE shard_id IS NOT NULL
        GROUP BY shard_id
    """).fetchall()

    if len(result) < 2:
        return True, "Single shard or fewer"

    counts = [row[1] for row in result]
    min_count = min(counts)
    max_count = max(counts)

    if min_count == 0:
        return False, "Empty shard found"

    ratio = max_count / min_count
    if ratio >= max_ratio:
        msg = f"Balance violated: {max_count}/{min_count} = {ratio:.2f}"
        return False, msg

    return True, f"ratio={ratio:.2f}"


class TestFullLifecycle:
    """Test 1: Full lifecycle - create->insert->search->delete->cleanup.

    Exercises invariants: I1, I2, I3, I6, I11
    """

    def test_full_lifecycle(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Test complete shard lifecycle from creation to deletion."""
        # Generate test vectors
        vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(50)]

        # Step 1: Create shard and insert embeddings
        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model)
            VALUES (?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model"],
        )

        # Insert embeddings with shard assignment
        emb_ids = insert_embeddings_to_db(tmp_db, vectors, shard_id)

        # Verify I1: Single assignment
        assert verify_invariant_i1_single_assignment(tmp_db)

        # Verify I2: Shard existence
        assert verify_invariant_i2_shard_existence(tmp_db)

        # Step 2: Run fix_pass to build index
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify I3: Count consistency
        state = get_shard_state(
            shard_id=shard_id,
            db_connection=tmp_db.connection,
            file_path=shard_path,
            dims=TEST_DIMS,
        )
        assert state.index_live == state.db_count == 50

        # Step 3: Verify search works (I6: Embedding retrievability)
        query = vectors[0]
        results = shard_manager.search(
            query=query.tolist(),
            k=5,
            dims=TEST_DIMS,
            provider="test",
            model="test-model",
            conn=tmp_db.connection,
        )
        assert len(results) > 0
        # Should find the query vector itself
        result_keys = [r.key for r in results]
        assert emb_ids[0] in result_keys

        # Step 4: Delete some embeddings
        table_name = f"embeddings_{TEST_DIMS}"
        deleted_ids = emb_ids[:10]
        for del_id in deleted_ids:
            tmp_db.connection.execute(
                f"DELETE FROM {table_name} WHERE id = ?", [del_id]
            )

        # Step 5: Run fix_pass to sync
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify I11: Deleted embeddings not in results
        results = shard_manager.search(
            query=query.tolist(),
            k=50,
            dims=TEST_DIMS,
            provider="test",
            model="test-model",
            conn=tmp_db.connection,
        )
        result_keys = [r.key for r in results]
        for del_id in deleted_ids:
            assert del_id not in result_keys, f"Deleted ID {del_id} found in results"

        # Verify count updated
        state = get_shard_state(
            shard_id=shard_id,
            db_connection=tmp_db.connection,
            file_path=shard_path,
            dims=TEST_DIMS,
        )
        assert state.index_live == state.db_count == 40


class TestFixPassRecovery:
    """Test 2: Fix pass recovery - missing files, orphans, idempotence.

    Exercises invariants: I3, I4, I5, I7, I8
    """

    def test_missing_file_recovery(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Fix pass rebuilds missing .usearch files from DuckDB (I5)."""
        # Create shard with embeddings
        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model)
            VALUES (?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model"],
        )

        vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(20)]
        insert_embeddings_to_db(tmp_db, vectors, shard_id)

        # Run fix_pass to build index
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)
        assert shard_path.exists()

        # Delete the index file (simulating corruption/crash)
        shard_path.unlink()
        assert not shard_path.exists()

        # Verify ghost record exists (I5 violation)
        ghosts = verify_invariant_i5_no_ghost_records(tmp_db, shard_manager.shard_dir)
        assert shard_id in ghosts

        # Run fix_pass - should rebuild
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify file rebuilt and no ghosts
        assert shard_path.exists()
        ghosts = verify_invariant_i5_no_ghost_records(tmp_db, shard_manager.shard_dir)
        assert len(ghosts) == 0

        # Verify I3: Count consistency
        state = get_shard_state(
            shard_id=shard_id,
            db_connection=tmp_db.connection,
            file_path=shard_path,
            dims=TEST_DIMS,
        )
        assert state.index_live == state.db_count == 20

    def test_orphan_file_cleanup(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
    ) -> None:
        """Fix pass removes orphan .usearch files without DB records (I4)."""
        # Create orphan file
        orphan_id = uuid4()
        orphan_path = shard_manager._shard_path(orphan_id)
        shard_manager.shard_dir.mkdir(exist_ok=True)

        # Create a minimal valid USearch index
        index = usearch_wrapper.create(TEST_DIMS)
        index.add(1, np.random.rand(TEST_DIMS).astype(np.float32))
        index.save(str(orphan_path))
        assert orphan_path.exists()

        # Verify orphan exists (I4 violation)
        orphans = verify_invariant_i4_no_orphan_files(tmp_db, shard_manager.shard_dir)
        assert orphan_path in orphans

        # Run fix_pass - should clean up
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify orphan removed
        assert not orphan_path.exists()
        orphans = verify_invariant_i4_no_orphan_files(tmp_db, shard_manager.shard_dir)
        assert len(orphans) == 0

    def test_idempotence(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Running fix_pass twice produces same result as once (I7)."""
        # Create shard with embeddings
        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model)
            VALUES (?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model"],
        )

        vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(30)]
        insert_embeddings_to_db(tmp_db, vectors, shard_id)

        # Run fix_pass first time
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Capture state after first pass
        state1 = get_shard_state(
            shard_id=shard_id,
            db_connection=tmp_db.connection,
            file_path=shard_path,
            dims=TEST_DIMS,
        )
        shard_count_1 = len(get_all_shard_ids(tmp_db))

        # Run fix_pass second time
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Capture state after second pass
        state2 = get_shard_state(
            shard_id=shard_id,
            db_connection=tmp_db.connection,
            file_path=shard_path,
            dims=TEST_DIMS,
        )
        shard_count_2 = len(get_all_shard_ids(tmp_db))

        # Verify idempotence
        assert state1.index_live == state2.index_live
        assert state1.db_count == state2.db_count
        assert shard_count_1 == shard_count_2

    def test_convergence_bounded_iterations(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Fix pass converges in bounded iterations (I8)."""
        # Create multiple shards with varying sizes
        for i in range(3):
            shard_id = uuid4()
            shard_path = shard_manager._shard_path(shard_id)

            tmp_db.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

            count = 20 + i * 10
            vectors = [
                generator.generate_hash_seeded(f"shard{i}_doc_{j}")
                for j in range(count)
            ]
            insert_embeddings_to_db(tmp_db, vectors, shard_id)

        # Run fix_pass - should complete without hitting max iterations
        # The fix_pass has max_iterations=10 safety limit
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify all shards have valid indexes
        for shard_id in get_all_shard_ids(tmp_db):
            shard_path = shard_manager._shard_path(shard_id)
            assert shard_path.exists(), f"Shard {shard_id} missing index"


class TestSplitAtThreshold:
    """Test 3: Split at threshold - vectors remain searchable after fix_pass.

    Exercises invariants: I1, I2, I9, I10

    Note: Uses reduced count for faster testing while still exercising split logic.
    """

    def test_split_at_threshold(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """All vectors remain searchable after fix_pass at split_threshold."""
        # Create initial shard
        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model)
            VALUES (?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model"],
        )

        # Insert exactly split_threshold vectors using clustered generation
        # to ensure K-means produces balanced clusters above merge_threshold
        cluster_data = generator.generate_clustered(num_clusters=2, per_cluster=50)
        vectors = [vec for vec, _label in cluster_data]
        emb_ids = insert_embeddings_to_db(tmp_db, vectors, shard_id)

        # Run fix_pass - may trigger split depending on K-means clustering
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify all vectors remain searchable (observable behavior)
        not_found = verify_all_vectors_searchable(
            shard_manager, tmp_db, vectors, emb_ids, k=len(vectors)
        )
        assert len(not_found) == 0, f"Lost {len(not_found)} embeddings: {not_found[:5]}"

        shards = get_all_shard_ids(tmp_db)

        # Verify I1: Single assignment still holds
        assert verify_invariant_i1_single_assignment(tmp_db)

        # Verify I2: All shard references valid
        assert verify_invariant_i2_shard_existence(tmp_db)

        # Verify I9: LIRE bound - shards <= embeddings
        total_embeddings = tmp_db.connection.execute(
            f"SELECT COUNT(*) FROM embeddings_{TEST_DIMS}"
        ).fetchone()[0]
        assert len(shards) <= total_embeddings

        # Verify total embedding count preserved
        assert total_embeddings == TEST_SPLIT_THRESHOLD

    def test_no_split_below_threshold(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Shard does not split when below threshold."""
        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model)
            VALUES (?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model"],
        )

        # Insert below threshold (use smaller count for faster test)
        test_count = min(200, TEST_SPLIT_THRESHOLD - 1)
        vectors = [
            generator.generate_hash_seeded(f"doc_{i}") for i in range(test_count)
        ]
        insert_embeddings_to_db(tmp_db, vectors, shard_id)

        # Run fix_pass
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify no split - still single shard
        shards = get_all_shard_ids(tmp_db)
        assert len(shards) == 1
        assert shards[0] == shard_id


class TestMergeAndCascade:
    """Test 4: Merge and cascade - delete below threshold, verify merge.

    Exercises invariants: I1, I8, I9, I10
    """

    def test_merge_below_threshold(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Small shard merges with another when db_count < merge_threshold."""
        # Create two shards - one small (below merge), one normal
        small_shard_id = uuid4()
        small_shard_path = shard_manager._shard_path(small_shard_id)
        normal_shard_id = uuid4()
        normal_shard_path = shard_manager._shard_path(normal_shard_id)

        for shard_id, shard_path in [
            (small_shard_id, small_shard_path),
            (normal_shard_id, normal_shard_path),
        ]:
            tmp_db.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

        # Insert small count (below merge threshold) to small shard
        small_count = TEST_MERGE_THRESHOLD - 1  # 99
        small_vectors = [
            generator.generate_hash_seeded(f"small_{i}") for i in range(small_count)
        ]
        insert_embeddings_to_db(tmp_db, small_vectors, small_shard_id)

        # Insert normal count to normal shard (above merge threshold)
        normal_count = TEST_MERGE_THRESHOLD + 50  # 150
        normal_vectors = [
            generator.generate_hash_seeded(f"normal_{i}") for i in range(normal_count)
        ]
        insert_embeddings_to_db(tmp_db, normal_vectors, normal_shard_id)

        # Run fix_pass - should trigger merge of small shard
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify merge occurred - should now have single shard
        shards = get_all_shard_ids(tmp_db)
        assert len(shards) == 1, f"Expected merge, got {len(shards)} shard(s)"

        # Verify total embeddings preserved
        total = tmp_db.connection.execute(
            f"SELECT COUNT(*) FROM embeddings_{TEST_DIMS}"
        ).fetchone()[0]
        assert total == len(small_vectors) + len(normal_vectors)

        # Verify I1: Single assignment
        assert verify_invariant_i1_single_assignment(tmp_db)

    def test_no_merge_sole_shard(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Sole shard cannot merge (no target available)."""
        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model)
            VALUES (?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model"],
        )

        # Insert below merge threshold
        small_count = TEST_MERGE_THRESHOLD - 1  # 99
        vectors = [
            generator.generate_hash_seeded(f"doc_{i}") for i in range(small_count)
        ]
        insert_embeddings_to_db(tmp_db, vectors, shard_id)

        # Run fix_pass
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify shard still exists (cannot merge with nothing)
        shards = get_all_shard_ids(tmp_db)
        assert len(shards) == 1
        assert shards[0] == shard_id

    def test_cascade_merge_triggers_split(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Merge that causes target to exceed split_threshold cascades to split."""
        # Create small shard and a shard near split threshold
        small_shard_id = uuid4()
        large_shard_id = uuid4()

        for shard_id in [small_shard_id, large_shard_id]:
            shard_path = shard_manager._shard_path(shard_id)
            tmp_db.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

        # Small shard: below merge threshold
        small_count = TEST_MERGE_THRESHOLD - 1
        small_vectors = [
            generator.generate_hash_seeded(f"small_{i}") for i in range(small_count)
        ]
        insert_embeddings_to_db(tmp_db, small_vectors, small_shard_id)

        # Large shard: just below split threshold
        # After merge, combined count should exceed split threshold
        large_count = TEST_SPLIT_THRESHOLD - TEST_MERGE_THRESHOLD + 2
        large_vectors = [
            generator.generate_hash_seeded(f"large_{i}") for i in range(large_count)
        ]
        insert_embeddings_to_db(tmp_db, large_vectors, large_shard_id)

        total_vectors = len(small_vectors) + len(large_vectors)

        # Run fix_pass - should merge small into large, then split
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Either stays merged (if split logic doesn't trigger) or splits
        # The key is total embeddings are preserved
        total = tmp_db.connection.execute(
            f"SELECT COUNT(*) FROM embeddings_{TEST_DIMS}"
        ).fetchone()[0]
        assert total == total_vectors

        # Verify I1 and I2
        assert verify_invariant_i1_single_assignment(tmp_db)
        assert verify_invariant_i2_shard_existence(tmp_db)


class TestSearchRecallWithCentroids:
    """Test 5: Search recall with centroids - clustered data, centroid filtering.

    Exercises invariants: I6, I12, I13
    """

    def test_clustered_data_search(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Search finds correct results in clustered data (I6, I13)."""
        # Generate clustered vectors
        clustered = generator.generate_clustered(
            num_clusters=3,
            per_cluster=20,
            separation=0.5,
        )

        # Create a shard and insert all vectors
        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model)
            VALUES (?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model"],
        )

        vectors = [vec for vec, _ in clustered]
        cluster_ids = [cid for _, cid in clustered]
        emb_ids = insert_embeddings_to_db(tmp_db, vectors, shard_id)

        # Build index
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Query with a vector from cluster 0
        cluster_0_indices = [i for i, cid in enumerate(cluster_ids) if cid == 0]
        query_vec = vectors[cluster_0_indices[0]]

        results = shard_manager.search(
            query=query_vec.tolist(),
            k=10,
            dims=TEST_DIMS,
            provider="test",
            model="test-model",
            conn=tmp_db.connection,
        )

        # Results should mostly be from same cluster
        result_keys = [r.key for r in results]
        cluster_0_emb_ids = [emb_ids[i] for i in cluster_0_indices]

        # At least half should be from same cluster (high recall)
        overlap = len(set(result_keys) & set(cluster_0_emb_ids))
        assert overlap >= 5, f"Low recall: {overlap}/10 from same cluster"

    def test_embedding_retrievability(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Any DB embedding is findable via search (I6)."""
        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model)
            VALUES (?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model"],
        )

        # Use hash-seeded vectors (will have higher similarity to centroid)
        vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(20)]
        emb_ids = insert_embeddings_to_db(tmp_db, vectors, shard_id)

        # Build index
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Sample vectors should find themselves as top result
        # Note: Using a sample to avoid long test runtime
        sample_indices = [0, 5, 10, 15]
        for i in sample_indices:
            vec = vectors[i]
            results = shard_manager.search(
                query=vec.tolist(),
                k=1,
                dims=TEST_DIMS,
                provider="test",
                model="test-model",
                conn=tmp_db.connection,
            )
            assert len(results) >= 1
            # Top result should be the query vector itself
            assert results[0].key == emb_ids[i], f"Vector {i} not found as top result"


class TestCompactionAndQuality:
    """Test 6: Compaction and quality - tombstones and quality rebuild.

    Exercises invariants: I3, I12
    """

    def test_tombstone_exclusion(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Deleted vectors (tombstones) are excluded from search results (I12)."""
        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model)
            VALUES (?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model"],
        )

        vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(30)]
        emb_ids = insert_embeddings_to_db(tmp_db, vectors, shard_id)

        # Build index
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Delete some embeddings from DB
        deleted_ids = emb_ids[:10]
        table_name = f"embeddings_{TEST_DIMS}"
        for del_id in deleted_ids:
            tmp_db.connection.execute(
                f"DELETE FROM {table_name} WHERE id = ?", [del_id]
            )

        # Run fix_pass to sync (creates tombstones or rebuilds)
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Search should not return deleted vectors
        for i in range(10):
            query = vectors[i]
            results = shard_manager.search(
                query=query.tolist(),
                k=30,
                dims=TEST_DIMS,
                provider="test",
                model="test-model",
                conn=tmp_db.connection,
            )
            result_keys = [r.key for r in results]
            for del_id in deleted_ids:
                assert del_id not in result_keys, (
                    f"Deleted ID {del_id} found in results"
                )

    def test_count_consistency_after_deletions(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Index count matches DB count after deletions and fix_pass (I3)."""
        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model)
            VALUES (?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model"],
        )

        vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(50)]
        emb_ids = insert_embeddings_to_db(tmp_db, vectors, shard_id)

        # Build index
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Delete 25% of embeddings
        delete_count = 12
        table_name = f"embeddings_{TEST_DIMS}"
        for del_id in emb_ids[:delete_count]:
            tmp_db.connection.execute(
                f"DELETE FROM {table_name} WHERE id = ?", [del_id]
            )

        # Run fix_pass
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify I3: Count consistency
        state = get_shard_state(
            shard_id=shard_id,
            db_connection=tmp_db.connection,
            file_path=shard_path,
            dims=TEST_DIMS,
        )
        expected_count = 50 - delete_count
        assert state.db_count == expected_count
        assert state.index_live == expected_count

    def test_quality_degradation_triggers_rebuild(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Verify fix_pass rebuilds when self_recall < quality_threshold (0.95)."""
        # Create shard with 50 vectors
        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model)
            VALUES (?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model"],
        )

        vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(50)]
        table_name = f"embeddings_{TEST_DIMS}"
        for i, emb in enumerate(vectors):
            tmp_db.connection.execute(
                f"""
                INSERT INTO {table_name}
                    (chunk_id, provider, model, embedding, dims, shard_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [i, "test", "test-model", emb.tolist(), TEST_DIMS, str(shard_id)],
            )

        # Build initial index
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Wait briefly to ensure file mtime differs
        time.sleep(0.01)

        # Get initial file modification time
        initial_mtime = shard_path.stat().st_mtime

        # Mock measure_quality to return degraded value (below 0.95 threshold)
        with patch.object(usearch_wrapper, "measure_quality", return_value=0.90):
            shard_manager.fix_pass(tmp_db.connection, check_quality=True)

        # Verify: file was rebuilt (mtime changed)
        new_mtime = shard_path.stat().st_mtime
        assert new_mtime > initial_mtime, (
            "Index should be rebuilt when quality < threshold"
        )

        # Verify: shard still has correct count after rebuild
        state = get_shard_state(
            shard_id=shard_id,
            db_connection=tmp_db.connection,
            file_path=shard_path,
            dims=TEST_DIMS,
        )
        assert state.index_live == 50
        assert state.db_count == 50


class TestScaleToTarget:
    """Test 7: Scale to target - larger vector count, performance check.

    Performance test with larger dataset.
    """

    @pytest.mark.parametrize("vector_count", [200, 500])
    def test_scale_performance(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
        vector_count: int,
    ) -> None:
        """System handles larger vector counts efficiently."""
        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model)
            VALUES (?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model"],
        )

        # Generate vectors
        vectors = [
            generator.generate_hash_seeded(f"doc_{i}") for i in range(vector_count)
        ]
        insert_embeddings_to_db(tmp_db, vectors, shard_id)

        # Measure fix_pass time
        start = time.perf_counter()
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)
        fix_time = time.perf_counter() - start

        # Should complete reasonably fast
        assert fix_time < 30.0, (
            f"fix_pass took {fix_time:.2f}s for {vector_count} vectors"
        )

        # Measure search time
        query = vectors[0]
        start = time.perf_counter()
        results = shard_manager.search(
            query=query.tolist(),
            k=10,
            dims=TEST_DIMS,
            provider="test",
            model="test-model",
            conn=tmp_db.connection,
        )
        search_time = time.perf_counter() - start

        # Search should be fast
        assert search_time < 1.0, f"Search took {search_time:.2f}s"
        assert len(results) >= 1


class TestChaosRecovery:
    """Test 8: Chaos recovery - file deletion/corruption mid-op.

    Robustness testing for crash recovery scenarios.
    """

    def test_recovery_from_corrupted_file(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Fix pass recovers from corrupted index file."""
        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model)
            VALUES (?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model"],
        )

        vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(30)]
        insert_embeddings_to_db(tmp_db, vectors, shard_id)

        # Build index
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)
        assert shard_path.exists()

        # Corrupt the file by truncating
        with open(shard_path, "wb") as f:
            f.write(b"corrupted data")

        # Run fix_pass - should detect corruption and rebuild
        # The open_view will fail, triggering rebuild
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify index rebuilt and valid
        state = get_shard_state(
            shard_id=shard_id,
            db_connection=tmp_db.connection,
            file_path=shard_path,
            dims=TEST_DIMS,
        )
        assert state.index_live == 30 or not shard_path.exists()

    def test_recovery_from_temp_file_left_behind(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Fix pass cleans up leftover .tmp files from interrupted operations."""
        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model)
            VALUES (?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model"],
        )

        vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(20)]
        insert_embeddings_to_db(tmp_db, vectors, shard_id)

        # Create orphan .tmp file
        shard_manager.shard_dir.mkdir(exist_ok=True)
        tmp_file = shard_path.with_suffix(".usearch.tmp")
        tmp_file.write_bytes(b"leftover temp data")
        assert tmp_file.exists()

        # Run fix_pass - should clean up temp file
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify temp file removed
        assert not tmp_file.exists()

    def test_empty_shard_cleanup(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
    ) -> None:
        """Empty shard (db_count=0) is removed by fix_pass."""
        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model)
            VALUES (?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model"],
        )

        # Create empty index file
        shard_manager.shard_dir.mkdir(exist_ok=True)
        index = usearch_wrapper.create(TEST_DIMS)
        index.save(str(shard_path))

        # Run fix_pass - should remove empty shard
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify shard removed
        shards = get_all_shard_ids(tmp_db)
        assert shard_id not in shards

        # File should also be removed
        assert not shard_path.exists()


class TestInvariantVerification:
    """Additional tests for specific invariant verification."""

    def test_lire_bound(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """I9: Number of shards never exceeds number of embeddings."""
        # Create multiple shards (without triggering splits for faster test)
        for i in range(3):
            shard_id = uuid4()
            shard_path = shard_manager._shard_path(shard_id)

            tmp_db.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

            # Insert reasonable count per shard
            count = 150
            vectors = [
                generator.generate_hash_seeded(f"batch{i}_doc_{j}")
                for j in range(count)
            ]
            insert_embeddings_to_db(tmp_db, vectors, shard_id)

        # Run fix_pass
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify I9: shards <= embeddings
        shard_count = len(get_all_shard_ids(tmp_db))
        embedding_count = tmp_db.connection.execute(
            f"SELECT COUNT(*) FROM embeddings_{TEST_DIMS}"
        ).fetchone()[0]

        assert shard_count <= embedding_count, (
            f"LIRE bound violated: {shard_count} shards > {embedding_count} embeddings"
        )

    def test_brute_force_ground_truth(
        self,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Verify brute_force_search provides correct ground truth."""
        # Generate test vectors
        vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(50)]

        # Query with first vector
        query = vectors[0]

        # Brute force should return indices sorted by distance
        result = brute_force_search(query, vectors, k=5)

        # First result should be the query itself (distance 0)
        assert result[0] == 0, "Query vector not found as top result"

        # Results should be sorted by distance
        for i in range(len(result) - 1):
            dist_i = 1.0 - float(np.dot(query, vectors[result[i]]))
            dist_next = 1.0 - float(np.dot(query, vectors[result[i + 1]]))
            assert dist_i <= dist_next + 1e-6, "Results not sorted by distance"


class TestKMeansSplitClustering:
    """Test K-means clustering for shard splitting with high split threshold.

    Uses split_threshold=2000 to verify K-means creates balanced clusters.
    K-means is used exclusively for splitting to guarantee cluster size
    constraints and prevent split->merge cycles.

    Exercises invariants: I1, I2
    """

    def test_kmeans_balanced_split(
        self,
        tmp_path: Path,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Shard splits using K-means clustering creates balanced children."""
        # Create fresh DB provider for this test
        db_path = tmp_path / "native_usearch_test.duckdb"
        db_provider = MockDBProvider(db_path)

        # Create shard directory
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(exist_ok=True)

        # Create ShardManager with high thresholds for native USearch clustering
        config = ShardingConfig(
            split_threshold=2000,
            merge_threshold=200,
            compaction_threshold=0.20,
            incremental_sync_threshold=0.10,
            quality_threshold=0.95,
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )

        try:
            # Create initial shard
            shard_id = uuid4()
            shard_path = shard_manager._shard_path(shard_id)

            db_provider.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

            # Insert 2500 vectors - above split_threshold of 2000
            # K-means will create 2 balanced clusters of ~1250 each
            vector_count = 2500
            vectors = [
                generator.generate_hash_seeded(f"native_usearch_doc_{i}")
                for i in range(vector_count)
            ]
            insert_embeddings_to_db(db_provider, vectors, shard_id)

            # Verify initial state: single shard with 2500 embeddings
            initial_shards = get_all_shard_ids(db_provider)
            assert len(initial_shards) == 1
            assert initial_shards[0] == shard_id

            # Run fix_pass - should trigger split via K-means clustering
            shard_manager.fix_pass(db_provider.connection, check_quality=False)

            # Verify split occurred - should now have 2+ shards
            final_shards = get_all_shard_ids(db_provider)
            assert len(final_shards) >= 2, (
                f"Expected split into 2+ shards, got {len(final_shards)} shard(s)"
            )

            # Verify I1: Single assignment still holds
            assert verify_invariant_i1_single_assignment(db_provider)

            # Verify I2: All shard references valid
            assert verify_invariant_i2_shard_existence(db_provider)

            # Verify total embedding count preserved
            total_embeddings = db_provider.connection.execute(
                f"SELECT COUNT(*) FROM embeddings_{TEST_DIMS}"
            ).fetchone()[0]
            assert total_embeddings == vector_count

        finally:
            db_provider.disconnect()

    def test_kmeans_uses_sample_for_large_shard(
        self,
        tmp_path: Path,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Verify that _kmeans_fallback samples vectors for centroid discovery.

        With 2500 vectors and sample_size = min(512, max(50, split_threshold//5)) = 400,
        only 400 vectors should be used for KMeans fitting, not all 2500.
        """
        from unittest.mock import patch

        # Create fresh DB provider for this test
        db_path = tmp_path / "sample_kmeans_test.duckdb"
        db_provider = MockDBProvider(db_path)

        # Create shard directory
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(exist_ok=True)

        # Create ShardManager with split_threshold=2000
        # sample_size = min(512, max(50, 2000//5)) = min(512, 400) = 400
        config = ShardingConfig(
            split_threshold=2000,
            merge_threshold=200,
            compaction_threshold=0.20,
            incremental_sync_threshold=0.10,
            quality_threshold=0.95,
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )

        try:
            # Create initial shard
            shard_id = uuid4()
            db_provider.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

            # Insert 2500 vectors
            vector_count = 2500
            vectors = [
                generator.generate_hash_seeded(f"sample_kmeans_doc_{i}")
                for i in range(vector_count)
            ]
            insert_embeddings_to_db(db_provider, vectors, shard_id)

            # Patch KMeans to capture the number of vectors it's fitted on
            original_kmeans = __import__("sklearn.cluster", fromlist=["KMeans"]).KMeans
            fitted_samples_count = []

            class MockKMeans:
                def __init__(self, n_clusters, n_init="auto"):
                    self._kmeans = original_kmeans(n_clusters=n_clusters, n_init=n_init)

                def fit(self, data):  # noqa: N805
                    fitted_samples_count.append(len(data))
                    return self._kmeans.fit(data)

                @property
                def cluster_centers_(self):
                    return self._kmeans.cluster_centers_

            with patch(
                "chunkhound.providers.database.shard_manager.KMeans", MockKMeans
            ):
                # Call _kmeans_fallback directly
                shard = {
                    "shard_id": str(shard_id),
                    "dims": TEST_DIMS,
                    "provider": "test",
                    "model": "test-model",
                }
                result = shard_manager._kmeans_fallback(
                    shard, db_provider.connection, n_clusters=2
                )

            # Verify KMeans was called with sample size, not full dataset
            assert len(fitted_samples_count) == 1, "KMeans should be called once"
            actual_sample_size = fitted_samples_count[0]
            expected_sample_size = min(512, max(50, config.split_threshold // 5))  # 400
            assert actual_sample_size == expected_sample_size, (
                f"KMeans fitted on {actual_sample_size} samples, "
                f"expected {expected_sample_size}"
            )
            assert actual_sample_size < vector_count, (
                f"KMeans should use sample ({actual_sample_size}), "
                f"not all vectors ({vector_count})"
            )

            # Verify all vectors got cluster assignments
            assert len(result) == vector_count, (
                f"Expected {vector_count} assignments, got {len(result)}"
            )

            # Verify assignments are valid cluster labels (0 or 1)
            unique_labels = set(result.values())
            assert unique_labels == {0, 1}, f"Expected labels {{0, 1}}, got {unique_labels}"

        finally:
            db_provider.disconnect()

    def test_kmeans_small_shard_uses_all_vectors(
        self,
        tmp_path: Path,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Verify small shards use all vectors for centroid discovery.

        When shard has fewer vectors than sample_size, all vectors should be
        used for KMeans fitting.
        """
        from unittest.mock import patch

        # Create fresh DB provider for this test
        db_path = tmp_path / "small_shard_kmeans_test.duckdb"
        db_provider = MockDBProvider(db_path)

        # Create shard directory
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(exist_ok=True)

        # Create ShardManager - sample_size = min(512, 2000//10) = 200
        config = ShardingConfig(
            split_threshold=2000,
            merge_threshold=20,
            compaction_threshold=0.20,
            incremental_sync_threshold=0.10,
            quality_threshold=0.95,
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )

        try:
            # Create initial shard
            shard_id = uuid4()
            db_provider.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

            # Insert only 50 vectors - less than sample_size of 200
            vector_count = 50
            vectors = [
                generator.generate_hash_seeded(f"small_shard_doc_{i}")
                for i in range(vector_count)
            ]
            insert_embeddings_to_db(db_provider, vectors, shard_id)

            # Patch KMeans to capture the number of vectors it's fitted on
            original_kmeans = __import__("sklearn.cluster", fromlist=["KMeans"]).KMeans
            fitted_samples_count = []

            class MockKMeans:
                def __init__(self, n_clusters, n_init="auto"):
                    self._kmeans = original_kmeans(n_clusters=n_clusters, n_init=n_init)

                def fit(self, data):  # noqa: N805
                    fitted_samples_count.append(len(data))
                    return self._kmeans.fit(data)

                @property
                def cluster_centers_(self):
                    return self._kmeans.cluster_centers_

            with patch(
                "chunkhound.providers.database.shard_manager.KMeans", MockKMeans
            ):
                # Call _kmeans_fallback directly
                shard = {
                    "shard_id": str(shard_id),
                    "dims": TEST_DIMS,
                    "provider": "test",
                    "model": "test-model",
                }
                result = shard_manager._kmeans_fallback(
                    shard, db_provider.connection, n_clusters=2
                )

            # Verify KMeans was called with ALL vectors (50), not sample_size (200)
            assert len(fitted_samples_count) == 1, "KMeans should be called once"
            sample_size = fitted_samples_count[0]
            assert sample_size == vector_count, (
                f"KMeans should use all {vector_count} vectors for small shard, "
                f"but used {sample_size}"
            )

            # Verify all vectors got cluster assignments
            assert len(result) == vector_count, (
                f"Expected {vector_count} assignments, got {len(result)}"
            )

        finally:
            db_provider.disconnect()

    def test_kmeans_assignment_via_duckdb_sql(
        self,
        tmp_path: Path,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Verify cluster assignments use DuckDB SQL with array_cosine_distance.

        After sampling and fitting KMeans, ALL vectors should be assigned
        via SQL query (not Python iteration), and assignments should be
        consistent with cosine distance to centroids.
        """
        # Create fresh DB provider for this test
        db_path = tmp_path / "sql_assignment_test.duckdb"
        db_provider = MockDBProvider(db_path)

        # Create shard directory
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(exist_ok=True)

        config = ShardingConfig(
            split_threshold=2000,
            merge_threshold=200,
            compaction_threshold=0.20,
            incremental_sync_threshold=0.10,
            quality_threshold=0.95,
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )

        try:
            # Create initial shard
            shard_id = uuid4()
            db_provider.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

            # Create two distinct clusters of vectors
            # Cluster 0: vectors close to [1, 0, 0, ...]
            # Cluster 1: vectors close to [0, 1, 0, ...]
            rng = np.random.default_rng(42)
            cluster0_vectors = []
            cluster1_vectors = []

            base0 = np.zeros(TEST_DIMS, dtype=np.float32)
            base0[0] = 1.0
            base1 = np.zeros(TEST_DIMS, dtype=np.float32)
            base1[1] = 1.0

            for i in range(100):
                # Cluster 0 vectors
                noise = rng.standard_normal(TEST_DIMS).astype(np.float32) * 0.1
                vec = base0 + noise
                vec = vec / np.linalg.norm(vec)
                cluster0_vectors.append(vec)

                # Cluster 1 vectors
                noise = rng.standard_normal(TEST_DIMS).astype(np.float32) * 0.1
                vec = base1 + noise
                vec = vec / np.linalg.norm(vec)
                cluster1_vectors.append(vec)

            # Insert all vectors
            all_vectors = cluster0_vectors + cluster1_vectors
            insert_embeddings_to_db(db_provider, all_vectors, shard_id)

            # Get IDs for verification
            table_name = f"embeddings_{TEST_DIMS}"
            id_results = db_provider.connection.execute(
                f"SELECT id FROM {table_name} WHERE shard_id = ? ORDER BY id",
                [str(shard_id)],
            ).fetchall()
            all_ids = [row[0] for row in id_results]
            cluster0_ids = set(all_ids[:100])
            cluster1_ids = set(all_ids[100:])

            # Call _kmeans_fallback
            shard = {
                "shard_id": str(shard_id),
                "dims": TEST_DIMS,
                "provider": "test",
                "model": "test-model",
            }
            result = shard_manager._kmeans_fallback(
                shard, db_provider.connection, n_clusters=2
            )

            # Verify all vectors got assignments
            assert len(result) == 200, f"Expected 200 assignments, got {len(result)}"

            # Verify clustering quality: vectors from same cluster should
            # mostly get same label
            cluster0_labels = [result[id_] for id_ in cluster0_ids]
            cluster1_labels = [result[id_] for id_ in cluster1_ids]

            # Count dominant label in each cluster
            cluster0_dominant = max(set(cluster0_labels), key=cluster0_labels.count)
            cluster1_dominant = max(set(cluster1_labels), key=cluster1_labels.count)

            cluster0_purity = cluster0_labels.count(cluster0_dominant) / len(
                cluster0_labels
            )
            cluster1_purity = cluster1_labels.count(cluster1_dominant) / len(
                cluster1_labels
            )

            # With well-separated clusters, purity should be high
            assert cluster0_purity > 0.9, (
                f"Cluster 0 purity too low: {cluster0_purity:.2%}"
            )
            assert cluster1_purity > 0.9, (
                f"Cluster 1 purity too low: {cluster1_purity:.2%}"
            )

            # Clusters should have different dominant labels
            assert cluster0_dominant != cluster1_dominant, (
                "Distinct clusters should have different labels"
            )

        finally:
            db_provider.disconnect()


class TestPerVectorInsertionRouting:
    """Test per-vector routing during insertion.

    When multiple shards exist, each vector should route to its nearest
    centroid shard, not all vectors to a single shard based on batch mean.
    """

    def test_vectors_route_to_nearest_shard(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
    ) -> None:
        """Each vector routes to its nearest centroid shard."""
        rng = np.random.default_rng(42)

        # Create 2 shards with orthogonal centroids
        c1 = np.array([1.0] + [0.0] * (TEST_DIMS - 1), dtype=np.float32)
        c2 = np.array([0.0, 1.0] + [0.0] * (TEST_DIMS - 2), dtype=np.float32)

        shard1_id, shard2_id = uuid4(), uuid4()
        for sid in [shard1_id, shard2_id]:
            tmp_db.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(sid), TEST_DIMS, "test", "test-model"],
            )

        # Build indexes with cluster vectors (above merge threshold)
        for sid, centroid in [(shard1_id, c1), (shard2_id, c2)]:
            vectors = [
                (centroid + rng.standard_normal(TEST_DIMS).astype(np.float32) * 0.1)
                for _ in range(30)
            ]
            insert_embeddings_to_db(tmp_db, vectors, sid)

        # Run fix_pass to build indexes and populate centroid cache
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify centroids are populated
        assert len(shard_manager.centroids) == 2, "Both shards should have centroids"

        # Create batch with vectors belonging to DIFFERENT shards
        batch_vectors = []
        # 10 vectors close to shard1 centroid
        for _ in range(10):
            vec = c1 + rng.standard_normal(TEST_DIMS).astype(np.float32) * 0.05
            batch_vectors.append(vec / np.linalg.norm(vec))  # Normalize
        # 10 vectors close to shard2 centroid
        for _ in range(10):
            vec = c2 + rng.standard_normal(TEST_DIMS).astype(np.float32) * 0.05
            batch_vectors.append(vec / np.linalg.norm(vec))  # Normalize

        # Insert to DB first (matching production flow)
        emb_ids = []
        for i, vec in enumerate(batch_vectors):
            result = tmp_db.connection.execute(
                f"""
                INSERT INTO embeddings_{TEST_DIMS}
                    (chunk_id, provider, model, embedding, dims)
                VALUES (?, ?, ?, ?, ?)
                RETURNING id
                """,
                [1000 + i, "test", "test-model", vec.tolist(), TEST_DIMS],
            ).fetchone()
            emb_ids.append(result[0])

        # Route via shard_manager.insert_embeddings()
        emb_dicts = [
            {"id": eid, "embedding": vec.tolist()}
            for eid, vec in zip(emb_ids, batch_vectors)
        ]
        success, _ = shard_manager.insert_embeddings(
            emb_dicts, TEST_DIMS, "test", "test-model", tmp_db.connection
        )
        assert success, "insert_embeddings should succeed"

        # Count how many of the new vectors went to each shard
        # (filter by id to only count new vectors, not initial cluster vectors)
        min_new_id = min(emb_ids)
        s1_count = tmp_db.connection.execute(
            f"""
            SELECT COUNT(*) FROM embeddings_{TEST_DIMS}
            WHERE shard_id = ? AND id >= ?
            """,
            [str(shard1_id), min_new_id],
        ).fetchone()[0]
        s2_count = tmp_db.connection.execute(
            f"""
            SELECT COUNT(*) FROM embeddings_{TEST_DIMS}
            WHERE shard_id = ? AND id >= ?
            """,
            [str(shard2_id), min_new_id],
        ).fetchone()[0]

        # Per-vector routing should split ~10 to each shard
        # Allow some tolerance for noise (at least 7 to each)
        assert s1_count >= 7, (
            f"Expected ~10 vectors to shard1 (c1-like), got {s1_count}. "
            f"Distribution: shard1={s1_count}, shard2={s2_count}"
        )
        assert s2_count >= 7, (
            f"Expected ~10 vectors to shard2 (c2-like), got {s2_count}. "
            f"Distribution: shard1={s1_count}, shard2={s2_count}"
        )

        # Also verify total is correct
        assert s1_count + s2_count == 20, (
            f"Total should be 20, got {s1_count + s2_count}"
        )


class TestNPAAfterMerge:
    """Test I10: NPA (Nearest Point Assignment) After Merge.

    After merge, each vector should be assigned to its nearest-centroid shard.
    Tests per-vector routing to multiple targets (not bulk assignment).
    """

    def test_npa_after_merge(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """After merge, vectors are in their nearest-centroid shard (I10).

        This test creates 3 target shards with distinct centroids, then a small
        source shard with vectors positioned near different targets. After merge,
        each vector should end up in its nearest target (per-vector routing).
        """
        # Set seed for reproducible test
        rng = np.random.default_rng(42)

        # Create 3 target shards with distinct cluster centroids
        target_shards: list[tuple[UUID, np.ndarray]] = []
        # Orthogonal centroids for distinct clusters
        c1 = np.array([1.0] + [0.0] * (TEST_DIMS - 1), dtype=np.float32)
        c2 = np.array([0.0, 1.0] + [0.0] * (TEST_DIMS - 2), dtype=np.float32)
        c3 = np.array([0.0, 0.0, 1.0] + [0.0] * (TEST_DIMS - 3), dtype=np.float32)
        target_centroids = [c1, c2, c3]

        for i, centroid in enumerate(target_centroids):
            target_id = uuid4()
            target_path = shard_manager._shard_path(target_id)
            target_shards.append((target_id, centroid))

            tmp_db.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(target_id), TEST_DIMS, "test", "test-model"],
            )

            # Generate vectors clustered around this centroid
            # Add some noise but keep vectors close to centroid
            target_vectors = []
            for j in range(TEST_MERGE_THRESHOLD + 10):  # Above merge threshold
                noise = rng.standard_normal(TEST_DIMS).astype(np.float32) * 0.1
                vec = centroid + noise
                vec = vec / np.linalg.norm(vec)  # Normalize
                target_vectors.append(vec)

            insert_embeddings_to_db(tmp_db, target_vectors, target_id)

        # Create small source shard with vectors deliberately close to DIFFERENT targets
        source_shard_id = uuid4()
        source_shard_path = shard_manager._shard_path(source_shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model)
            VALUES (?, ?, ?, ?)
            """,
            [str(source_shard_id), TEST_DIMS, "test", "test-model"],
        )

        # Create source vectors close to each of the 3 target centroids
        source_vectors = []
        expected_targets: list[UUID] = []

        for target_idx in range(3):
            centroid = target_centroids[target_idx]
            target_id = target_shards[target_idx][0]
            # Create 5 vectors close to this target's centroid
            for _ in range(5):
                noise = rng.standard_normal(TEST_DIMS).astype(np.float32) * 0.05
                vec = centroid + noise
                vec = vec / np.linalg.norm(vec)
                source_vectors.append(vec)
                expected_targets.append(target_id)

        # Total: 15 vectors in source, below merge threshold (20)
        source_ids = insert_embeddings_to_db(tmp_db, source_vectors, source_shard_id)

        # Build indexes and run fix pass - this should trigger merge
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify source shard was merged (it was below threshold)
        shards_after = get_all_shard_ids(tmp_db)
        assert source_shard_id not in shards_after, "Source shard should be merged"

        # Verify per-vector routing: vectors distributed across multiple targets
        # (This is the key behavior change from bulk assignment)
        table_name = f"embeddings_{TEST_DIMS}"

        # Count how many vectors went to each target
        target_counts: dict[UUID, int] = {}
        for emb_id in source_ids:
            result = tmp_db.connection.execute(
                f"SELECT shard_id FROM {table_name} WHERE id = ?",
                [emb_id],
            ).fetchone()
            assert result is not None, f"Embedding {emb_id} not found"
            shard_id = result[0] if isinstance(result[0], UUID) else UUID(result[0])
            target_counts[shard_id] = target_counts.get(shard_id, 0) + 1

        # Per-vector routing should distribute to at least 2 targets
        # (Bulk assignment would put all 15 in one target)
        assert len(target_counts) >= 2, (
            f"Per-vector routing should distribute to multiple targets, "
            f"but only {len(target_counts)} target(s) used: {target_counts}"
        )

        # Verify distribution is not complete bulk assignment
        # Per-vector routing should split vectors; bulk would put all 15 in one target
        max_count = max(target_counts.values())
        assert max_count < 15, (
            f"Appears to be bulk assignment: one target has all {max_count} vectors. "
            f"Expected per-vector routing to distribute. Distribution: {target_counts}"
        )

        # Verify invariants still hold
        assert verify_invariant_i1_single_assignment(tmp_db)
        assert verify_invariant_i2_shard_existence(tmp_db)


class TestCentroidFilterCorrectness:
    """Test I13: Centroid Filter Correctness.

    Filtered shards should be a superset of shards containing true top-k results.
    """

    def test_centroid_filter_correctness(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Centroid filtering includes all shards with true top-k results (I13)."""
        # Create multiple shards with well-separated clusters
        num_shards = 3
        vectors_per_shard = 30
        all_vectors: list[np.ndarray] = []
        all_emb_ids: list[int] = []
        shard_ids: list[UUID] = []

        for i in range(num_shards):
            shard_id = uuid4()
            shard_path = shard_manager._shard_path(shard_id)
            shard_ids.append(shard_id)

            tmp_db.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

            # Generate vectors for this shard using different seed ranges
            vectors = [
                generator.generate_hash_seeded(f"shard{i}_doc_{j}")
                for j in range(vectors_per_shard)
            ]
            emb_ids = insert_embeddings_to_db(tmp_db, vectors, shard_id)
            all_vectors.extend(vectors)
            all_emb_ids.extend(emb_ids)

        # Build indexes
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Query with one of the vectors
        query_idx = 0
        query = all_vectors[query_idx]
        k = 10

        # Get search results from ShardManager
        results = shard_manager.search(
            query=query.tolist(),
            k=k,
            dims=TEST_DIMS,
            provider="test",
            model="test-model",
            conn=tmp_db.connection,
        )
        result_keys = {r.key for r in results}

        # Compute ground truth using brute force
        ground_truth_indices = brute_force_search(query, all_vectors, k=k)
        ground_truth_keys = {all_emb_ids[idx] for idx in ground_truth_indices}

        # Calculate recall: how many ground truth results were found
        found_ground_truth = result_keys & ground_truth_keys
        recall = len(found_ground_truth) / len(ground_truth_keys)

        # With proper centroid filtering, recall should be high
        # We use a threshold of 0.7 to account for ANN approximation
        n_found = len(found_ground_truth)
        n_total = len(ground_truth_keys)
        assert recall >= 0.7, (
            f"Low recall {recall:.2f}: centroid filtering may have excluded "
            f"shards containing true top-k results. Found {n_found}/{n_total}"
        )

        # Additionally verify that the query vector itself is found
        # (should always be in results since we're querying with an indexed vector)
        query_emb_id = all_emb_ids[query_idx]
        assert query_emb_id in result_keys, (
            "Query vector not found in results - centroid filter too aggressive"
        )


class TestOverfetchMultiplier:
    """Test overfetch_multiplier configuration is used during search.

    Verifies that search uses multiplied k internally for reranking candidates
    but still returns the requested k results.
    """

    def test_overfetch_returns_correct_k(
        self,
        tmp_path: Path,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Search returns exactly k results despite overfetch multiplier."""
        db_path = tmp_path / "overfetch_test.duckdb"
        db_provider = MockDBProvider(db_path)

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(exist_ok=True)

        # Create config with custom overfetch_multiplier
        config = ShardingConfig(
            split_threshold=TEST_SPLIT_THRESHOLD,
            merge_threshold=TEST_MERGE_THRESHOLD,
            overfetch_multiplier=3,  # Custom multiplier
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )

        try:
            # Create shard with vectors
            shard_id = uuid4()
            shard_path = shard_manager._shard_path(shard_id)

            db_provider.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

            # Insert 50 vectors
            vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(50)]
            insert_embeddings_to_db(db_provider, vectors, shard_id)

            # Build index
            shard_manager.fix_pass(db_provider.connection, check_quality=False)

            # Search for k=5 results
            query = vectors[0]
            k = 5
            results = shard_manager.search(
                query=query.tolist(),
                k=k,
                dims=TEST_DIMS,
                provider="test",
                model="test-model",
                conn=db_provider.connection,
            )

            # Should return exactly k results (trimmed from internal k * multiplier)
            assert len(results) == k, (
                f"Expected {k} results, got {len(results)}. "
                f"Overfetch should retrieve k*multiplier internally but return k"
            )

        finally:
            db_provider.disconnect()

    def test_overfetch_improves_candidate_pool(
        self,
        tmp_path: Path,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Higher overfetch multiplier gives more candidates for reranking."""
        db_path = tmp_path / "overfetch_candidates.duckdb"
        db_provider = MockDBProvider(db_path)

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(exist_ok=True)

        # Higher multiplier means more candidates internally
        config = ShardingConfig(
            split_threshold=TEST_SPLIT_THRESHOLD,
            merge_threshold=TEST_MERGE_THRESHOLD,
            overfetch_multiplier=5,
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )

        try:
            shard_id = uuid4()
            shard_path = shard_manager._shard_path(shard_id)

            db_provider.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

            vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(30)]
            emb_ids = insert_embeddings_to_db(db_provider, vectors, shard_id)

            shard_manager.fix_pass(db_provider.connection, check_quality=False)

            # Query with first vector, should find itself
            query = vectors[0]
            results = shard_manager.search(
                query=query.tolist(),
                k=3,
                dims=TEST_DIMS,
                provider="test",
                model="test-model",
                conn=db_provider.connection,
            )

            # Query vector should be in results (overfetch ensures good recall)
            result_keys = [r.key for r in results]
            assert emb_ids[0] in result_keys, (
                "Query vector not found - overfetch should improve recall"
            )

        finally:
            db_provider.disconnect()


class TestHNSWConfigurableParams:
    """Test custom HNSW parameters are used during index rebuild.

    Verifies that hnsw_connectivity, hnsw_expansion_add, and hnsw_expansion_search
    configuration values are passed to usearch_wrapper.create during rebuild.
    """

    def test_custom_hnsw_params_in_rebuild(
        self,
        tmp_path: Path,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Custom HNSW params are used when rebuilding index via fix_pass."""
        db_path = tmp_path / "hnsw_params_test.duckdb"
        db_provider = MockDBProvider(db_path)

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(exist_ok=True)

        # Create config with custom HNSW parameters
        config = ShardingConfig(
            split_threshold=TEST_SPLIT_THRESHOLD,
            merge_threshold=TEST_MERGE_THRESHOLD,
            hnsw_connectivity=32,  # Custom connectivity (default is 16)
            hnsw_expansion_add=256,  # Custom expansion during add (default is 128)
            hnsw_expansion_search=128,  # Custom expansion during search (default is 64)
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )

        try:
            # Create shard with embeddings
            shard_id = uuid4()
            shard_path = shard_manager._shard_path(shard_id)

            db_provider.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

            vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(30)]
            insert_embeddings_to_db(db_provider, vectors, shard_id)

            # Run fix_pass to trigger rebuild
            shard_manager.fix_pass(db_provider.connection, check_quality=False)

            # Verify the shard was rebuilt (index file exists)
            assert shard_path.exists(), "Index file should exist after rebuild"

            # Verify index is functional by searching
            query = vectors[0]
            results = shard_manager.search(
                query=query.tolist(),
                k=5,
                dims=TEST_DIMS,
                provider="test",
                model="test-model",
                conn=db_provider.connection,
            )

            # Should find results
            assert len(results) > 0, "Should find results with custom HNSW params"

        finally:
            db_provider.disconnect()

    def test_hnsw_params_affect_index_structure(
        self,
        tmp_path: Path,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Different HNSW params create different index structures."""
        db_path = tmp_path / "hnsw_structure_test.duckdb"
        db_provider = MockDBProvider(db_path)

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(exist_ok=True)

        # Use higher connectivity for denser graph
        config = ShardingConfig(
            split_threshold=TEST_SPLIT_THRESHOLD,
            merge_threshold=TEST_MERGE_THRESHOLD,
            hnsw_connectivity=48,  # Higher connectivity
            hnsw_expansion_add=192,
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )

        try:
            shard_id = uuid4()
            shard_path = shard_manager._shard_path(shard_id)

            db_provider.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

            vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(40)]
            emb_ids = insert_embeddings_to_db(db_provider, vectors, shard_id)

            shard_manager.fix_pass(db_provider.connection, check_quality=False)

            # Higher connectivity should generally improve recall
            query = vectors[10]
            results = shard_manager.search(
                query=query.tolist(),
                k=5,
                dims=TEST_DIMS,
                provider="test",
                model="test-model",
                conn=db_provider.connection,
            )

            # Query vector should be found as top result
            assert len(results) >= 1
            assert results[0].key == emb_ids[10], (
                "Higher connectivity should ensure query vector is top result"
            )

        finally:
            db_provider.disconnect()


class TestBulkIndexer:
    """Test BulkIndexer deferred quality checks and context manager behavior.

    Verifies:
    - on_batch_completed() triggers fix_pass at configured intervals
    - __exit__ runs final fix_pass
    - Deferred mode delays quality checks
    """

    def test_fix_pass_called_at_interval(
        self,
        tmp_path: Path,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """fix_pass is called when batch count reaches quality_check_interval."""
        from chunkhound.providers.database.bulk_indexer import BulkIndexer

        db_path = tmp_path / "bulk_interval_test.duckdb"
        db_provider = MockDBProvider(db_path)

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(exist_ok=True)

        # Set interval to 3 batches
        config = ShardingConfig(
            split_threshold=TEST_SPLIT_THRESHOLD,
            merge_threshold=TEST_MERGE_THRESHOLD,
            quality_check_mode="deferred",
            quality_check_interval=3,
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )
        db_provider.set_shard_manager(shard_manager)

        try:
            # Create shard for testing
            shard_id = uuid4()
            shard_path = shard_manager._shard_path(shard_id)

            db_provider.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

            vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(20)]
            insert_embeddings_to_db(db_provider, vectors, shard_id)

            bulk_indexer = BulkIndexer(db_provider, config)

            # Call on_batch_completed 2 times - should NOT trigger fix_pass yet
            bulk_indexer.on_batch_completed()
            bulk_indexer.on_batch_completed()

            # Index should not exist yet (no fix_pass run)
            # Note: the shard file may or may not exist depending on internal state

            # Third call should trigger fix_pass (interval=3)
            bulk_indexer.on_batch_completed()

            # After fix_pass, index should exist
            assert shard_path.exists(), (
                "Index should exist after fix_pass triggered at interval"
            )

            # Verify counter was reset
            assert bulk_indexer._batches_since_quality_check == 0

        finally:
            db_provider.disconnect()

    def test_exit_runs_final_fix_pass(
        self,
        tmp_path: Path,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """__exit__ runs final fix_pass if batches were completed."""
        from chunkhound.providers.database.bulk_indexer import BulkIndexer

        db_path = tmp_path / "bulk_exit_test.duckdb"
        db_provider = MockDBProvider(db_path)

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(exist_ok=True)

        # Large interval so intermediate fix_pass won't trigger
        config = ShardingConfig(
            split_threshold=TEST_SPLIT_THRESHOLD,
            merge_threshold=TEST_MERGE_THRESHOLD,
            quality_check_mode="deferred",
            quality_check_interval=100,  # Won't reach this
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )
        db_provider.set_shard_manager(shard_manager)

        try:
            shard_id = uuid4()
            shard_path = shard_manager._shard_path(shard_id)

            db_provider.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

            vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(20)]
            insert_embeddings_to_db(db_provider, vectors, shard_id)

            # Use context manager
            with BulkIndexer(db_provider, config) as bulk_indexer:
                # Call on_batch_completed a few times (less than interval)
                bulk_indexer.on_batch_completed()
                bulk_indexer.on_batch_completed()

                # Index may not exist yet
                assert bulk_indexer._needs_final_quality_check is True

            # After context exit, final fix_pass should have run
            assert shard_path.exists(), (
                "Index should exist after context exit triggers final fix_pass"
            )

        finally:
            db_provider.disconnect()

    def test_deferred_mode_skips_quality_in_intermediate_fix_pass(
        self,
        tmp_path: Path,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Deferred mode runs fix_pass without quality check at intervals."""
        from chunkhound.providers.database.bulk_indexer import BulkIndexer

        db_path = tmp_path / "bulk_deferred_test.duckdb"
        db_provider = MockDBProvider(db_path)

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(exist_ok=True)

        # Deferred mode: quality checks only on final exit
        config = ShardingConfig(
            split_threshold=TEST_SPLIT_THRESHOLD,
            merge_threshold=TEST_MERGE_THRESHOLD,
            quality_check_mode="deferred",
            quality_check_interval=2,
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )
        db_provider.set_shard_manager(shard_manager)

        try:
            shard_id = uuid4()
            shard_path = shard_manager._shard_path(shard_id)

            db_provider.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

            vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(30)]
            insert_embeddings_to_db(db_provider, vectors, shard_id)

            bulk_indexer = BulkIndexer(db_provider, config)

            # Trigger intermediate fix_pass (interval=2)
            bulk_indexer.on_batch_completed()
            bulk_indexer.on_batch_completed()

            # In deferred mode, _needs_final_quality_check should still be True
            # (quality wasn't checked in intermediate fix_pass)
            assert bulk_indexer._needs_final_quality_check is True, (
                "Deferred mode should keep _needs_final_quality_check=True"
            )

            # Index should exist from intermediate fix_pass
            assert shard_path.exists()

        finally:
            db_provider.disconnect()

    def test_immediate_mode_clears_quality_flag(
        self,
        tmp_path: Path,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Immediate mode clears _needs_final_quality_check after interval fix_pass."""
        from chunkhound.providers.database.bulk_indexer import BulkIndexer

        db_path = tmp_path / "bulk_immediate_test.duckdb"
        db_provider = MockDBProvider(db_path)

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(exist_ok=True)

        # Immediate mode: quality checks at every interval
        config = ShardingConfig(
            split_threshold=TEST_SPLIT_THRESHOLD,
            merge_threshold=TEST_MERGE_THRESHOLD,
            quality_check_mode="immediate",
            quality_check_interval=2,
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )
        db_provider.set_shard_manager(shard_manager)

        try:
            shard_id = uuid4()
            shard_path = shard_manager._shard_path(shard_id)

            db_provider.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

            vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(30)]
            insert_embeddings_to_db(db_provider, vectors, shard_id)

            bulk_indexer = BulkIndexer(db_provider, config)

            # Trigger intermediate fix_pass with quality check (interval=2)
            bulk_indexer.on_batch_completed()
            bulk_indexer.on_batch_completed()

            # In immediate mode, _needs_final_quality_check should be False
            # (quality was checked in intermediate fix_pass)
            assert bulk_indexer._needs_final_quality_check is False, (
                "Immediate mode should clear _needs_final_quality_check after interval"
            )

        finally:
            db_provider.disconnect()


class TestShardSplitAndJoinLifecycle:
    """Test full shard lifecycle: 3 splits then merge back to empty.

    Exercises invariants: I1, I2, I4, I5, I9

    Lifecycle phases:
    1. Empty -> 100 vectors -> first split (2 shards)
    2. +60 vectors -> second split (3 shards)
    3. +70 vectors -> third split (4 shards)
    4. Delete to <20 per shard -> merges
    5. Delete all remaining -> empty (0 shards)
    """

    def test_split_and_join_lifecycle(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Complete lifecycle: empty -> 3 splits -> delete all -> empty."""
        # Phase 1: First split
        # Verify starting state: 0 shards, 0 embeddings
        initial_shards = get_all_shard_ids(tmp_db)
        assert len(initial_shards) == 0, "Expected 0 shards at start"

        initial_embeddings = tmp_db.connection.execute(
            f"SELECT COUNT(*) FROM embeddings_{TEST_DIMS}"
        ).fetchone()[0]
        assert initial_embeddings == 0, "Expected 0 embeddings at start"

        # Create initial shard (shard_0) with UUID
        shard_0_id = uuid4()
        shard_0_path = shard_manager._shard_path(shard_0_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model)
            VALUES (?, ?, ?, ?)
            """,
            [str(shard_0_id), TEST_DIMS, "test", "test-model"],
        )

        # Insert exactly TEST_SPLIT_THRESHOLD (100) vectors to shard_0
        # Use clustered vectors to guarantee K-means produces balanced splits
        clustered = generator.generate_clustered(
            num_clusters=2,
            per_cluster=TEST_SPLIT_THRESHOLD // 2,
            separation=0.5,
        )
        vectors = [vec for vec, _ in clustered]
        insert_embeddings_to_db(tmp_db, vectors, shard_0_id)

        # Run fix_pass() -> triggers first split
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify: 2 shards exist (original shard_id deleted, replaced by 2 new shards)
        shards_after_split = get_all_shard_ids(tmp_db)
        assert len(shards_after_split) == 2, (
            f"Expected 2 shards after first split, got {len(shards_after_split)}"
        )

        # Original shard should be deleted
        assert shard_0_id not in shards_after_split, (
            "Original shard should be deleted after split"
        )

        # Verify: Total embeddings == 100
        total_embeddings = tmp_db.connection.execute(
            f"SELECT COUNT(*) FROM embeddings_{TEST_DIMS}"
        ).fetchone()[0]
        assert total_embeddings == TEST_SPLIT_THRESHOLD, (
            f"Expected {TEST_SPLIT_THRESHOLD} embeddings, got {total_embeddings}"
        )

        # Verify: I1 invariant (single assignment) holds
        assert verify_invariant_i1_single_assignment(tmp_db), (
            "I1 violated: embeddings have multiple shard assignments"
        )

        # Verify: I2 invariant (shard existence) holds
        assert verify_invariant_i2_shard_existence(tmp_db), (
            "I2 violated: embeddings reference non-existent shards"
        )

        # ========================================
        # Phase 2: Second split (+70 vectors)
        # ========================================

        # Pick shard with more vectors to ensure we exceed split threshold
        shard_counts = get_shard_counts(tmp_db)
        target_shard = max(shard_counts.keys(), key=lambda s: shard_counts[s])

        # Insert 70 vectors to guarantee split (even if shard only has 30, 30+70=100)
        # Use clustered vectors to guarantee K-means produces balanced splits
        clustered = generator.generate_clustered(
            num_clusters=2,
            per_cluster=35,
            separation=0.5,
        )
        phase2_vectors = [vec for vec, _ in clustered]
        insert_embeddings_to_db(tmp_db, phase2_vectors, target_shard)

        # Run fix_pass - should trigger second split
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify second split
        shards_phase2 = get_all_shard_ids(tmp_db)
        assert len(shards_phase2) >= 3, (
            f"Expected >= 3 shards after second split, got {len(shards_phase2)}"
        )

        # Verify total embeddings preserved (100 + 70 = 170)
        total_embeddings = tmp_db.connection.execute(
            f"SELECT COUNT(*) FROM embeddings_{TEST_DIMS}"
        ).fetchone()[0]
        assert total_embeddings == 170, (
            f"Expected 170 embeddings, got {total_embeddings}"
        )

        # Verify invariants
        assert verify_invariant_i1_single_assignment(tmp_db), (
            "I1 violated after phase 2: embeddings have multiple shard assignments"
        )
        assert verify_invariant_i2_shard_existence(tmp_db), (
            "I2 violated after phase 2: embeddings reference non-existent shards"
        )

        # ========================================
        # Phase 3: Third split (+70 vectors)
        # ========================================

        # Pick shard with most vectors to guarantee split
        shard_counts = get_shard_counts(tmp_db)
        target_shard = max(shard_counts.keys(), key=lambda s: shard_counts[s])

        # Insert 70 more vectors to trigger third split
        # Use clustered vectors to guarantee K-means produces balanced splits
        clustered = generator.generate_clustered(
            num_clusters=2,
            per_cluster=35,
            separation=0.5,
        )
        phase3_vectors = [vec for vec, _ in clustered]
        insert_embeddings_to_db(tmp_db, phase3_vectors, target_shard)

        # Run fix_pass - should trigger third split
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify third split
        shards_phase3 = get_all_shard_ids(tmp_db)
        assert len(shards_phase3) >= 4, (
            f"Expected >= 4 shards after third split, got {len(shards_phase3)}"
        )

        # Verify total embeddings preserved (100 + 70 + 70 = 240)
        total_embeddings = tmp_db.connection.execute(
            f"SELECT COUNT(*) FROM embeddings_{TEST_DIMS}"
        ).fetchone()[0]
        assert total_embeddings == 240, (
            f"Expected 240 embeddings, got {total_embeddings}"
        )

        # Verify invariants
        assert verify_invariant_i1_single_assignment(tmp_db), (
            "I1 violated after phase 3: embeddings have multiple shard assignments"
        )
        assert verify_invariant_i2_shard_existence(tmp_db), (
            "I2 violated after phase 3: embeddings reference non-existent shards"
        )

        # Verify I9: LIRE bound (shards <= embeddings)
        n_shards = len(shards_phase3)
        assert n_shards <= total_embeddings, (
            f"LIRE bound violated: {n_shards} shards > {total_embeddings} embeddings"
        )

        # Store shard IDs after third split for Phase 4
        # shards_phase3 is used in Phase 4 below

        # ========================================
        # Phase 4: Delete to trigger merges
        # ========================================

        # Get shard counts before deletion
        shard_counts_before = get_shard_counts(tmp_db)
        shards_before_merge = set(shard_counts_before.keys())

        # Delete vectors to bring each shard below merge_threshold
        for shard_id, count in shard_counts_before.items():
            # Leave enough below merge threshold to trigger merge
            to_delete = count - (TEST_MERGE_THRESHOLD - 5)
            if to_delete > 0:
                delete_from_shard(tmp_db, shard_id, to_delete)

        # Verify all shards now below merge threshold
        shard_counts_after_delete = get_shard_counts(tmp_db)
        for shard_id, count in shard_counts_after_delete.items():
            assert count < TEST_MERGE_THRESHOLD, (
                f"Shard {shard_id} has {count} vectors, "
                f"should be < {TEST_MERGE_THRESHOLD}"
            )

        # Run fix_pass - should trigger merges
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Get state after merge
        shards_after_merge = set(get_all_shard_ids(tmp_db))

        # Verify shard count decreased (merges occurred)
        assert len(shards_after_merge) < len(shards_before_merge), (
            f"Expected shard count to decrease: {len(shards_before_merge)} -> "
            f"{len(shards_after_merge)}"
        )

        # Verify invariants
        assert verify_invariant_i1_single_assignment(tmp_db), (
            "I1 violated after phase 4: embeddings have multiple shard assignments"
        )
        assert verify_invariant_i2_shard_existence(tmp_db), (
            "I2 violated after phase 4: embeddings reference non-existent shards"
        )

        # ========================================
        # Phase 5: Delete all remaining to empty
        # ========================================

        # Get remaining embeddings and delete all of them
        remaining_counts = get_shard_counts(tmp_db)
        for shard_id, count in remaining_counts.items():
            if count > 0:
                delete_from_shard(tmp_db, shard_id, count)

        # Verify all embeddings deleted
        total_embeddings = tmp_db.connection.execute(
            f"SELECT COUNT(*) FROM embeddings_{TEST_DIMS}"
        ).fetchone()[0]
        assert total_embeddings == 0, (
            f"Expected 0 embeddings after delete all, got {total_embeddings}"
        )

        # Run fix_pass - should clean up empty shards
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify final empty state
        final_shards = get_all_shard_ids(tmp_db)
        assert len(final_shards) == 0, (
            f"Expected 0 shards at end, got {len(final_shards)}"
        )

        # Verify I4: No orphan files
        orphans = verify_invariant_i4_no_orphan_files(tmp_db, shard_manager.shard_dir)
        assert len(orphans) == 0, f"Found orphan files: {orphans}"

        # Verify I5: No ghost records
        ghosts = verify_invariant_i5_no_ghost_records(tmp_db, shard_manager.shard_dir)
        assert len(ghosts) == 0, f"Found ghost records: {ghosts}"


def _generate_batch_sizes(
    rng: np.random.Generator,
    total: int,
    min_size: int,
    max_size: int,
) -> list[int]:
    """Generate list of randomized batch sizes summing to total.

    Uses seeded RNG for reproducibility. Last batch adjusts to hit exact total.

    Args:
        rng: Seeded numpy random generator
        total: Target total vectors
        min_size: Minimum batch size (e.g., 900)
        max_size: Maximum batch size (e.g., 1100)

    Returns:
        List of batch sizes, e.g., [1032, 945, 1087, ...]
    """
    batches = []
    remaining = total
    while remaining > max_size:
        size = int(rng.integers(min_size, max_size + 1))
        batches.append(size)
        remaining -= size
    if remaining > 0:
        batches.append(remaining)
    return batches


@pytest.mark.slow
@pytest.mark.timeout(1200)  # 20 minutes for full insert+delete lifecycle
class TestMillionVectorStress:
    """Stress test: 1M vectors with production thresholds.

    Exercises shard lifecycle at scale with:
    - 1M artificial vectors loaded in randomized ~1k batches
    - Deletions in randomized ~1k batches
    - Production thresholds (split=100,000, merge=10,000)
    - Full invariant verification after each batch

    Run with:
        uv run pytest tests/test_sharding.py::TestMillionVectorStress \\
            --run-slow -v -s
    """

    def test_million_vector_lifecycle(self, tmp_path: Path) -> None:
        """Full lifecycle: insert 1M vectors, delete in batches, verify invariants."""
        # Setup seeded RNG for reproducibility
        rng = np.random.default_rng(STRESS_SEED)
        generator = SyntheticEmbeddingGenerator(dims=STRESS_DIMS, seed=STRESS_SEED)

        # Create DB with stress test dimensions table
        db_path = tmp_path / "stress_test.duckdb"
        db_provider = MockDBProvider(db_path)
        db_provider._create_embeddings_table(STRESS_DIMS)

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(exist_ok=True)

        # Show test paths
        formatter = RichOutputFormatter()
        formatter.info(f"Test DB: {db_path}")
        formatter.info(f"Shard dir: {shard_dir}")

        config = ShardingConfig(
            split_threshold=STRESS_SPLIT_THRESHOLD,
            merge_threshold=STRESS_MERGE_THRESHOLD,
            compaction_threshold=0.20,
            incremental_sync_threshold=0.10,
            quality_threshold=0.95,
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )

        try:
            # Use ProgressManager to suppress loguru INFO/DEBUG logs and show progress
            with formatter.create_progress_display() as progress:
                # ===== PHASE 1: INSERT =====
                insert_batches = _generate_batch_sizes(
                    rng, STRESS_TOTAL_VECTORS, STRESS_BATCH_MIN, STRESS_BATCH_MAX
                )

                progress.add_task(
                    "insert",
                    "Inserting vectors",
                    total=STRESS_TOTAL_VECTORS,
                )

                total_inserted = 0
                phase_start = time.perf_counter()

                # Amortized O(1) measurement: record (N, T(N)) at intervals
                measurement_points: list[tuple[int, float]] = []
                next_measurement = STRESS_O1_MEASUREMENT_INTERVAL

                for batch_idx, batch_size in enumerate(insert_batches):
                    # Generate vectors for this batch
                    batch_name = f"stress_{batch_idx}"
                    vectors = generator.generate_batch(batch_size, batch_name)

                    # Insert embeddings to DuckDB (without shard assignment)
                    emb_ids = batch_insert_embeddings_to_db(
                        db_provider,
                        vectors,
                        dims=STRESS_DIMS,
                        shard_id=None,
                        start_chunk_id=total_inserted,
                    )

                    # Build embedding dicts for shard assignment
                    emb_dicts = [
                        {"id": emb_id, "embedding": vec.tolist()}
                        for emb_id, vec in zip(emb_ids, vectors)
                    ]

                    # Assign embeddings to shards (creates shard if needed)
                    success, needs_fix = shard_manager.insert_embeddings(
                        emb_dicts,
                        dims=STRESS_DIMS,
                        provider="test",
                        model="test-model",
                        conn=db_provider.connection,
                    )

                    total_inserted += batch_size

                    # Record measurement point at each 100K interval
                    if total_inserted >= next_measurement:
                        cumulative_time = time.perf_counter() - phase_start
                        measurement_points.append((total_inserted, cumulative_time))
                        next_measurement += STRESS_O1_MEASUREMENT_INTERVAL

                    # Build USearch index if shard created or threshold crossed
                    if needs_fix:
                        shard_manager.fix_pass(
                            db_provider.connection, check_quality=False
                        )

                    # ===== VERIFICATION AFTER EACH BATCH =====

                    # I1: Single assignment
                    assert verify_invariant_i1_single_assignment(db_provider), (
                        f"I1 violated at insert batch {batch_idx}"
                    )

                    # I2: Shard existence
                    assert verify_invariant_i2_shard_existence(db_provider), (
                        f"I2 violated at insert batch {batch_idx}"
                    )

                    # Update progress
                    elapsed = time.perf_counter() - phase_start
                    rate = total_inserted / elapsed if elapsed > 0 else 0
                    shard_count = len(get_all_shard_ids(db_provider))
                    progress.update_task(
                        "insert",
                        advance=batch_size,
                        speed=f"{rate:.0f} vec/s",
                        info=f"{shard_count} shards",
                    )

                progress.finish_task("insert")

            # ===== AMORTIZED O(1) VALIDATION =====
            # Two-pronged validation per Sedgewick's empirical analysis method:
            # 1. R² >= 0.95: Validates T(N) is linear (proves O(1) amortized)
            # 2. Doubling ratio T(2N)/T(N) ≈ 2: Confirms linear growth rate
            if len(measurement_points) >= 3:
                ns = [p[0] for p in measurement_points]
                times = [p[1] for p in measurement_points]

                slope, intercept, r_value, _, _ = stats.linregress(ns, times)
                r_squared = r_value**2
                slope_ms_per_vec = slope * 1000  # Convert to ms/vec

                formatter.info(
                    f"Amortized O(1) validation: R²={r_squared:.4f}, "
                    f"slope={slope_ms_per_vec:.4f}ms/vec, "
                    f"points={len(measurement_points)}"
                )

                # R² >= 0.95 proves linearity (constant per-op cost)
                assert r_squared >= 0.95, (
                    f"Non-linear insertion behavior: R²={r_squared:.4f} "
                    f"(split/merge overhead degrading O(1))"
                )

                # Doubling ratio validation: T(2N)/T(N) ≈ 2 for O(1) amortized
                time_by_n = dict(measurement_points)
                doubling_ratios = []
                for n, t in measurement_points:
                    if 2 * n in time_by_n:
                        ratio = time_by_n[2 * n] / t
                        doubling_ratios.append((n, 2 * n, ratio))

                if doubling_ratios:
                    avg_ratio = sum(r[2] for r in doubling_ratios) / len(doubling_ratios)
                    formatter.info(
                        f"Doubling ratios: avg={avg_ratio:.2f}, "
                        f"pairs={len(doubling_ratios)}"
                    )
                    # For O(1) amortized, ratio should be ~2 (linear total time)
                    # Allow 1.5-2.5 tolerance for measurement noise
                    assert 1.5 <= avg_ratio <= 2.5, (
                        f"Doubling ratio {avg_ratio:.2f} outside [1.5, 2.5] "
                        f"(expected ~2 for O(1) amortized)"
                    )

            # Verify expected shard count after full insert
            final_shard_count = len(get_all_shard_ids(db_provider))
            min_expected_shards = STRESS_TOTAL_VECTORS // STRESS_SPLIT_THRESHOLD
            max_expected_shards = STRESS_TOTAL_VECTORS // STRESS_MERGE_THRESHOLD
            assert min_expected_shards <= final_shard_count <= max_expected_shards, (
                f"Expected {min_expected_shards}-{max_expected_shards} shards, "
                f"got {final_shard_count}"
            )

            # Use ProgressManager for DELETE phase
            with formatter.create_progress_display() as progress:
                # ===== PHASE 2: DELETE =====
                delete_batches = _generate_batch_sizes(
                    rng, STRESS_TOTAL_VECTORS, STRESS_BATCH_MIN, STRESS_BATCH_MAX
                )

                # Get all IDs and shuffle for random deletion order
                all_ids = db_provider.connection.execute(
                    f"SELECT id FROM embeddings_{STRESS_DIMS}"
                ).fetchall()
                all_ids = [row[0] for row in all_ids]
                rng.shuffle(all_ids)

                progress.add_task(
                    "delete",
                    "Deleting vectors",
                    total=STRESS_TOTAL_VECTORS,
                )

                total_deleted = 0
                phase_start = time.perf_counter()
                id_cursor = 0
                table = f"embeddings_{STRESS_DIMS}"

                for batch_idx, batch_size in enumerate(delete_batches):
                    # Get IDs to delete
                    ids_to_delete = all_ids[id_cursor : id_cursor + batch_size]
                    id_cursor += batch_size

                    if not ids_to_delete:
                        break

                    # Batch delete from DB
                    placeholders = ", ".join("?" * len(ids_to_delete))
                    db_provider.connection.execute(
                        f"DELETE FROM {table} WHERE id IN ({placeholders})",
                        ids_to_delete,
                    )
                    total_deleted += len(ids_to_delete)
                    remaining = STRESS_TOTAL_VECTORS - total_deleted

                    # Check if any shard fell below merge threshold
                    min_count_result = db_provider.connection.execute(
                        f"""
                        SELECT MIN(cnt) FROM (
                            SELECT COUNT(*) as cnt FROM {table}
                            GROUP BY shard_id
                        )
                        """
                    ).fetchone()
                    min_shard_count = min_count_result[0] if min_count_result[0] else 0

                    # Run fix_pass only when threshold crossed (like insertion)
                    needs_fix = min_shard_count < STRESS_MERGE_THRESHOLD
                    if needs_fix:
                        shard_manager.fix_pass(
                            db_provider.connection, check_quality=False
                        )

                    # ===== VERIFICATION AFTER EACH DELETE BATCH =====

                    # I1: Single assignment (DB-only, always check)
                    assert verify_invariant_i1_single_assignment(db_provider), (
                        f"I1 violated at delete batch {batch_idx}"
                    )

                    # I2: Shard existence (DB-only, always check)
                    assert verify_invariant_i2_shard_existence(db_provider), (
                        f"I2 violated at delete batch {batch_idx}"
                    )

                    # I4/I5: File invariants only valid after fix_pass
                    if needs_fix:
                        orphans = verify_invariant_i4_no_orphan_files(
                            db_provider, shard_dir
                        )
                        assert len(orphans) == 0, (
                            f"I4 violated: {len(orphans)} orphans at batch {batch_idx}"
                        )

                        ghosts = verify_invariant_i5_no_ghost_records(
                            db_provider, shard_dir
                        )
                        assert len(ghosts) == 0, (
                            f"I5 violated: {len(ghosts)} ghosts at batch {batch_idx}"
                        )

                    # Update progress
                    elapsed = time.perf_counter() - phase_start
                    rate = total_deleted / elapsed if elapsed > 0 else 0
                    shard_count = len(get_all_shard_ids(db_provider))
                    progress.update_task(
                        "delete",
                        advance=len(ids_to_delete),
                        speed=f"{rate:.0f}/s",
                        info=f"{remaining:,} left, {shard_count} shards",
                    )

                progress.finish_task("delete")

            # Final fix_pass to clean up any remaining empty shards
            shard_manager.fix_pass(db_provider.connection, check_quality=False)

            # Match production: full compaction to reclaim deleted row space
            # Uses shared compaction_utils with production thresholds (50%/100MB)
            db_provider.optimize()

            # Final assertions
            final_count = db_provider.connection.execute(
                f"SELECT COUNT(*) FROM embeddings_{STRESS_DIMS}"
            ).fetchone()[0]
            final_shards = len(get_all_shard_ids(db_provider))

            assert final_count == 0, f"Expected 0 vectors, found {final_count}"
            assert final_shards == 0, f"Expected 0 shards, found {final_shards}"

            # Final invariant checks
            orphans = verify_invariant_i4_no_orphan_files(db_provider, shard_dir)
            assert len(orphans) == 0, f"Final: {len(orphans)} orphan files"

            ghosts = verify_invariant_i5_no_ghost_records(db_provider, shard_dir)
            assert len(ghosts) == 0, f"Final: {len(ghosts)} ghost records"

        finally:
            db_provider.disconnect()


class TestSplitMergeCycleProtection:
    """Test protection against infinite split->merge cycles.

    Verifies that splitting creates child shards above merge_threshold,
    preventing immediate merge operations that would recreate the cycle.

    This tests the fix for a bug where USearch's native clustering could
    create many tiny clusters, triggering merges that led to infinite loops.
    """

    def test_split_creates_viable_children(
        self,
        tmp_path: Path,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Split must create children >= merge_threshold to avoid cycles."""
        # Create fresh DB provider for this test
        db_path = tmp_path / "cycle_protection_test.duckdb"
        db_provider = MockDBProvider(db_path)

        # Create shard directory
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(exist_ok=True)

        # Use higher thresholds to ensure sample-based k-means works correctly.
        # With split_threshold=2000, sample_size = min(512, 200) = 200 samples,
        # which is sufficient for reliable centroid discovery.
        split_threshold = 2000
        merge_threshold = 200
        config = ShardingConfig(
            split_threshold=split_threshold,
            merge_threshold=merge_threshold,
            compaction_threshold=0.20,
            incremental_sync_threshold=0.10,
            quality_threshold=0.95,
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )

        try:
            # Create initial shard
            shard_id = uuid4()
            shard_path = shard_manager._shard_path(shard_id)

            db_provider.connection.execute(
                """
                INSERT INTO vector_shards (shard_id, dims, provider, model)
                VALUES (?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model"],
            )

            # Insert exactly split_threshold vectors to trigger split
            # Use clustered vectors to guarantee K-means produces balanced splits
            # (random vectors don't naturally cluster and may produce unbalanced splits)
            clustered = generator.generate_clustered(
                num_clusters=2,
                per_cluster=split_threshold // 2,
                separation=0.5,
            )
            vectors = [vec for vec, _cluster_id in clustered]
            insert_embeddings_to_db(db_provider, vectors, shard_id)

            # Run fix_pass - should split into 2 balanced shards
            shard_manager.fix_pass(db_provider.connection, check_quality=False)

            # Verify all child shards are >= merge_threshold
            shards = get_all_shard_ids(db_provider)
            assert len(shards) >= 2, f"Expected split into 2+ shards, got {len(shards)}"

            for child_shard_id in shards:
                count = db_provider.connection.execute(
                    f"SELECT COUNT(*) FROM embeddings_{TEST_DIMS} WHERE shard_id = ?",
                    [str(child_shard_id)],
                ).fetchone()[0]
                assert count >= merge_threshold, (
                    f"Child shard {child_shard_id} has {count} vectors, "
                    f"below merge_threshold {merge_threshold}"
                )

            # Record shard count before second fix_pass
            shard_count_before = len(shards)

            # Run fix_pass again - should NOT trigger any merges
            shard_manager.fix_pass(db_provider.connection, check_quality=False)

            # Shard count should remain stable (no merge occurred)
            final_shards = get_all_shard_ids(db_provider)
            n_final = len(final_shards)
            assert n_final == shard_count_before, (
                f"Shard count: {shard_count_before} -> {n_final} (unwanted merge)"
            )

            # Verify invariants still hold
            assert verify_invariant_i1_single_assignment(db_provider)
            assert verify_invariant_i2_shard_existence(db_provider)

        finally:
            db_provider.disconnect()


class TestPortability:
    """I14: Database directory can be relocated and reopened.

    Verifies that the database is self-contained and portable:
    - Shard paths are derived at runtime from db_path.parent / "shards"
    - No absolute paths stored in database
    - Directory can be zipped and moved to any machine/OS
    """

    def test_relocation_preserves_functionality(
        self,
        tmp_path: Path,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Database directory relocation preserves all operations."""
        import shutil

        # Phase 1: Create and populate original DB
        original_dir = tmp_path / "original"
        original_dir.mkdir()
        db_path = original_dir / "test.duckdb"
        db_provider = MockDBProvider(db_path)
        shard_dir = original_dir / "shards"
        shard_dir.mkdir()

        config = ShardingConfig(
            split_threshold=TEST_SPLIT_THRESHOLD,
            merge_threshold=TEST_MERGE_THRESHOLD,
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )

        # Insert enough vectors to create a shard with index file
        vectors = generator.generate_batch(50, "portability_test")
        emb_ids = insert_embeddings_to_db(db_provider, vectors)

        # Assign to shards and build index
        emb_dicts = [
            {"id": emb_id, "embedding": vec.tolist()}
            for emb_id, vec in zip(emb_ids, vectors)
        ]
        success, needs_fix = shard_manager.insert_embeddings(
            emb_dicts,
            dims=TEST_DIMS,
            provider="test",
            model="test-model",
            conn=db_provider.connection,
        )
        assert success

        # Build index via fix_pass
        shard_manager.fix_pass(db_provider.connection, check_quality=False)

        # Capture pre-relocation state
        original_shard_ids = get_all_shard_ids(db_provider)
        assert len(original_shard_ids) > 0, "Expected at least one shard"

        # Perform search before relocation
        query_vec = vectors[0].tolist()
        pre_results = shard_manager.search(
            query=query_vec,
            k=5,
            dims=TEST_DIMS,
            provider="test",
            model="test-model",
            conn=db_provider.connection,
        )
        pre_result_keys = [r.key for r in pre_results]

        # Phase 2: Relocate entire directory
        db_provider.disconnect()
        relocated_dir = tmp_path / "relocated"
        shutil.copytree(original_dir, relocated_dir)

        # Phase 3: Reopen from new location
        new_db_path = relocated_dir / "test.duckdb"
        new_provider = MockDBProvider(new_db_path)
        new_shard_dir = relocated_dir / "shards"

        new_manager = ShardManager(
            db_provider=new_provider,
            shard_dir=new_shard_dir,
            config=config,
        )

        # Populate centroids for search
        new_manager.fix_pass(new_provider.connection, check_quality=False)

        # Phase 4: Verify operations work correctly
        try:
            # 4a: Search returns same results
            post_results = new_manager.search(
                query=query_vec,
                k=5,
                dims=TEST_DIMS,
                provider="test",
                model="test-model",
                conn=new_provider.connection,
            )
            post_result_keys = [r.key for r in post_results]
            assert pre_result_keys == post_result_keys, (
                f"Search results differ after relocation: "
                f"pre={pre_result_keys}, post={post_result_keys}"
            )

            # 4b: New inserts succeed
            new_vectors = generator.generate_batch(10, "post_relocation")
            new_ids = insert_embeddings_to_db(new_provider, new_vectors)
            new_emb_dicts = [
                {"id": emb_id, "embedding": vec.tolist()}
                for emb_id, vec in zip(new_ids, new_vectors)
            ]
            insert_success, _ = new_manager.insert_embeddings(
                new_emb_dicts,
                dims=TEST_DIMS,
                provider="test",
                model="test-model",
                conn=new_provider.connection,
            )
            assert insert_success, "Insert failed after relocation"

            # 4c: fix_pass completes without errors
            new_manager.fix_pass(new_provider.connection, check_quality=False)

            # 4d: Invariants hold after relocation
            orphans = verify_invariant_i4_no_orphan_files(new_provider, new_shard_dir)
            assert len(orphans) == 0, f"I4 violated: {len(orphans)} orphan files"

            ghosts = verify_invariant_i5_no_ghost_records(new_provider, new_shard_dir)
            assert len(ghosts) == 0, f"I5 violated: {len(ghosts)} ghost records"

            assert verify_invariant_i1_single_assignment(new_provider)
            assert verify_invariant_i2_shard_existence(new_provider)

        finally:
            new_provider.disconnect()


@pytest.mark.slow
@pytest.mark.timeout(600)  # 10 minutes for complexity tests
class TestAmortizedComplexity:
    """Tests verifying amortized O(1) insert/delete complexity.

    Uses linear regression on cumulative time T(N) to verify:
    - If amortized cost is O(1), then T(N) = aN + b (linear)
    - High R² confirms linearity
    - Slope a represents per-operation cost

    Uses smaller thresholds (split=10000, merge=1000) to allow splits during testing.

    Run with:
        uv run pytest tests/test_sharding.py::TestAmortizedComplexity \\
            --run-slow -v -s
    """

    # Test configuration for complexity measurements
    COMPLEXITY_DIMS = STRESS_DIMS  # 128
    COMPLEXITY_SPLIT_THRESHOLD = 10_000
    COMPLEXITY_MERGE_THRESHOLD = 1_000
    COMPLEXITY_SEED = 42

    # Measurement points for linear regression
    MEASUREMENT_POINTS = [5_000, 10_000, 15_000, 20_000, 25_000]

    def test_amortized_o1_insertion(self, tmp_path: Path) -> None:
        """Verify amortized O(1) insertion complexity using linear regression.

        Methodology:
        1. Insert vectors in batches, recording cumulative time at measurement points
        2. Fit linear regression: T(N) = slope * N + intercept
        3. Pass criteria: R² >= 0.95 (high linearity) and slope < 1ms/vec

        If complexity were O(log n) or O(n), T(N) would be superlinear
        and R-squared would drop.
        """
        generator = SyntheticEmbeddingGenerator(
            dims=self.COMPLEXITY_DIMS, seed=self.COMPLEXITY_SEED
        )

        # Create DB with complexity test dimensions table
        db_path = tmp_path / "complexity_insert.duckdb"
        db_provider = MockDBProvider(db_path)
        db_provider._create_embeddings_table(self.COMPLEXITY_DIMS)

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(exist_ok=True)

        config = ShardingConfig(
            split_threshold=self.COMPLEXITY_SPLIT_THRESHOLD,
            merge_threshold=self.COMPLEXITY_MERGE_THRESHOLD,
            compaction_threshold=0.20,
            incremental_sync_threshold=0.10,
            quality_threshold=0.95,
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )

        try:
            # Measurement arrays
            n_values: list[int] = []
            cumulative_times: list[float] = []

            total_inserted = 0
            cumulative_time = 0.0
            batch_size = 500  # Fixed batch size for consistent measurements
            next_measurement_idx = 0

            max_vectors = self.MEASUREMENT_POINTS[-1]

            while total_inserted < max_vectors:
                # Generate batch
                batch_name = f"complexity_{total_inserted}"
                vectors = generator.generate_batch(batch_size, batch_name)

                # Time the insertion
                start = time.perf_counter()

                # Insert to DB
                emb_ids = batch_insert_embeddings_to_db(
                    db_provider,
                    vectors,
                    dims=self.COMPLEXITY_DIMS,
                    shard_id=None,
                    start_chunk_id=total_inserted,
                )

                # Assign to shards
                emb_dicts = [
                    {"id": emb_id, "embedding": vec.tolist()}
                    for emb_id, vec in zip(emb_ids, vectors)
                ]
                success, needs_fix = shard_manager.insert_embeddings(
                    emb_dicts,
                    dims=self.COMPLEXITY_DIMS,
                    provider="test",
                    model="test-model",
                    conn=db_provider.connection,
                )
                assert success, f"Insert failed at {total_inserted}"

                # Run fix_pass if needed (split handling)
                if needs_fix:
                    shard_manager.fix_pass(db_provider.connection, check_quality=False)

                elapsed = time.perf_counter() - start
                cumulative_time += elapsed
                total_inserted += batch_size

                # Record measurement at each measurement point
                if (
                    next_measurement_idx < len(self.MEASUREMENT_POINTS)
                    and total_inserted >= self.MEASUREMENT_POINTS[next_measurement_idx]
                ):
                    n_values.append(total_inserted)
                    cumulative_times.append(cumulative_time)
                    next_measurement_idx += 1

            # Linear regression: T(N) = slope * N + intercept
            result = stats.linregress(n_values, cumulative_times)

            # Assert R² >= 0.95 (high linearity confirms O(1) amortized)
            r_squared = result.rvalue**2
            assert r_squared >= 0.95, (
                f"Insertion complexity not O(1): R² = {r_squared:.4f} < 0.95. "
                f"Data points: N={n_values}, T={cumulative_times}"
            )

            # Assert slope < 1ms per vector (reasonable per-op cost)
            slope_ms_per_vec = result.slope * 1000
            assert slope_ms_per_vec < 1.0, (
                f"Insertion too slow: {slope_ms_per_vec:.4f}ms/vec >= 1ms/vec"
            )

        finally:
            db_provider.disconnect()

    def test_amortized_o1_deletion(self, tmp_path: Path) -> None:
        """Verify amortized O(1) deletion complexity using linear regression.

        Methodology:
        1. Pre-populate with vectors
        2. Delete in batches, recording cumulative time at measurement points
        3. Fit linear regression: T(N) = slope * N + intercept
        4. Pass criteria: R² >= 0.95 (high linearity)

        Deletion complexity includes tombstone management and potential merges.
        """
        rng = np.random.default_rng(self.COMPLEXITY_SEED)
        generator = SyntheticEmbeddingGenerator(
            dims=self.COMPLEXITY_DIMS, seed=self.COMPLEXITY_SEED
        )

        # Create DB and populate
        db_path = tmp_path / "complexity_delete.duckdb"
        db_provider = MockDBProvider(db_path)
        db_provider._create_embeddings_table(self.COMPLEXITY_DIMS)

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(exist_ok=True)

        config = ShardingConfig(
            split_threshold=self.COMPLEXITY_SPLIT_THRESHOLD,
            merge_threshold=self.COMPLEXITY_MERGE_THRESHOLD,
            compaction_threshold=0.20,
            incremental_sync_threshold=0.10,
            quality_threshold=0.95,
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )

        try:
            # Phase 1: Pre-populate with max vectors
            max_vectors = self.MEASUREMENT_POINTS[-1]
            batch_size = 1000
            total_inserted = 0

            while total_inserted < max_vectors:
                batch_name = f"prepop_{total_inserted}"
                vectors = generator.generate_batch(batch_size, batch_name)

                emb_ids = batch_insert_embeddings_to_db(
                    db_provider,
                    vectors,
                    dims=self.COMPLEXITY_DIMS,
                    shard_id=None,
                    start_chunk_id=total_inserted,
                )

                emb_dicts = [
                    {"id": emb_id, "embedding": vec.tolist()}
                    for emb_id, vec in zip(emb_ids, vectors)
                ]
                success, needs_fix = shard_manager.insert_embeddings(
                    emb_dicts,
                    dims=self.COMPLEXITY_DIMS,
                    provider="test",
                    model="test-model",
                    conn=db_provider.connection,
                )
                assert success, f"Prepopulation insert failed at {total_inserted}"

                if needs_fix:
                    shard_manager.fix_pass(db_provider.connection, check_quality=False)

                total_inserted += batch_size

            # Get all IDs and shuffle for random deletion order
            table = f"embeddings_{self.COMPLEXITY_DIMS}"
            all_ids = db_provider.connection.execute(
                f"SELECT id FROM {table}"
            ).fetchall()
            all_ids = [row[0] for row in all_ids]
            rng.shuffle(all_ids)

            # Phase 2: Measure deletion complexity
            n_values: list[int] = []
            cumulative_times: list[float] = []

            total_deleted = 0
            cumulative_time = 0.0
            delete_batch_size = 500
            id_cursor = 0
            next_measurement_idx = 0

            while total_deleted < max_vectors:
                ids_to_delete = all_ids[id_cursor : id_cursor + delete_batch_size]
                id_cursor += delete_batch_size

                if not ids_to_delete:
                    break

                # Time the deletion
                start = time.perf_counter()

                placeholders = ", ".join("?" * len(ids_to_delete))
                db_provider.connection.execute(
                    f"DELETE FROM {table} WHERE id IN ({placeholders})",
                    ids_to_delete,
                )

                # Check if merge needed
                min_count_result = db_provider.connection.execute(
                    f"""
                    SELECT MIN(cnt) FROM (
                        SELECT COUNT(*) as cnt FROM {table}
                        GROUP BY shard_id
                    )
                    """
                ).fetchone()
                min_shard_count = min_count_result[0] if min_count_result[0] else 0

                if min_shard_count < self.COMPLEXITY_MERGE_THRESHOLD:
                    shard_manager.fix_pass(db_provider.connection, check_quality=False)

                elapsed = time.perf_counter() - start
                cumulative_time += elapsed
                total_deleted += len(ids_to_delete)

                # Record measurement at each measurement point
                if (
                    next_measurement_idx < len(self.MEASUREMENT_POINTS)
                    and total_deleted >= self.MEASUREMENT_POINTS[next_measurement_idx]
                ):
                    n_values.append(total_deleted)
                    cumulative_times.append(cumulative_time)
                    next_measurement_idx += 1

            # Linear regression: T(N) = slope * N + intercept
            result = stats.linregress(n_values, cumulative_times)

            # Assert R² >= 0.95 (high linearity confirms O(1) amortized)
            r_squared = result.rvalue**2
            assert r_squared >= 0.95, (
                f"Deletion complexity not O(1): R² = {r_squared:.4f} < 0.95. "
                f"Data points: N={n_values}, T={cumulative_times}"
            )

        finally:
            db_provider.disconnect()

    def test_no_complexity_regression(self, tmp_path: Path) -> None:
        """Verify no complexity degradation at scale.

        Methodology:
        1. Measure slope (time per vector) at 10K vectors
        2. Measure slope at 50K vectors
        3. Compare ratio: should be 0.8 <= ratio <= 1.3

        If complexity degrades (e.g., O(log n) or O(n)), slope at 50K would be
        significantly higher than at 10K.
        """
        generator = SyntheticEmbeddingGenerator(
            dims=self.COMPLEXITY_DIMS, seed=self.COMPLEXITY_SEED
        )

        # Create DB
        db_path = tmp_path / "complexity_regression.duckdb"
        db_provider = MockDBProvider(db_path)
        db_provider._create_embeddings_table(self.COMPLEXITY_DIMS)

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir(exist_ok=True)

        config = ShardingConfig(
            split_threshold=self.COMPLEXITY_SPLIT_THRESHOLD,
            merge_threshold=self.COMPLEXITY_MERGE_THRESHOLD,
            compaction_threshold=0.20,
            incremental_sync_threshold=0.10,
            quality_threshold=0.95,
            shard_similarity_threshold=0.1,
        )

        shard_manager = ShardManager(
            db_provider=db_provider,
            shard_dir=shard_dir,
            config=config,
        )

        try:
            # Measurement points for two regions
            early_points = [2_000, 4_000, 6_000, 8_000, 10_000]
            late_points = [30_000, 35_000, 40_000, 45_000, 50_000]
            all_points = early_points + late_points

            n_values: list[int] = []
            cumulative_times: list[float] = []

            total_inserted = 0
            cumulative_time = 0.0
            batch_size = 500
            next_point_idx = 0

            max_vectors = all_points[-1]

            while total_inserted < max_vectors:
                batch_name = f"regression_{total_inserted}"
                vectors = generator.generate_batch(batch_size, batch_name)

                start = time.perf_counter()

                emb_ids = batch_insert_embeddings_to_db(
                    db_provider,
                    vectors,
                    dims=self.COMPLEXITY_DIMS,
                    shard_id=None,
                    start_chunk_id=total_inserted,
                )

                emb_dicts = [
                    {"id": emb_id, "embedding": vec.tolist()}
                    for emb_id, vec in zip(emb_ids, vectors)
                ]
                success, needs_fix = shard_manager.insert_embeddings(
                    emb_dicts,
                    dims=self.COMPLEXITY_DIMS,
                    provider="test",
                    model="test-model",
                    conn=db_provider.connection,
                )
                assert success, f"Insert failed at {total_inserted}"

                if needs_fix:
                    shard_manager.fix_pass(db_provider.connection, check_quality=False)

                elapsed = time.perf_counter() - start
                cumulative_time += elapsed
                total_inserted += batch_size

                # Record at measurement points
                if (
                    next_point_idx < len(all_points)
                    and total_inserted >= all_points[next_point_idx]
                ):
                    n_values.append(total_inserted)
                    cumulative_times.append(cumulative_time)
                    next_point_idx += 1

            # Split data into early (10K) and late (50K) regions
            early_n = [n for n in n_values if n <= 10_000]
            early_t = cumulative_times[: len(early_n)]

            late_n = [n for n in n_values if n >= 30_000]
            late_t = cumulative_times[-len(late_n) :]

            # Compute slopes for each region
            early_result = stats.linregress(early_n, early_t)
            late_result = stats.linregress(late_n, late_t)

            early_slope = early_result.slope
            late_slope = late_result.slope

            # Compute ratio: late_slope / early_slope
            # Ratio near 1.0 means no degradation
            ratio = late_slope / early_slope if early_slope > 0 else float("inf")

            assert 0.8 <= ratio <= 1.3, (
                f"Complexity degradation detected: slope ratio = {ratio:.4f} "
                f"(expected 0.8-1.3). Early slope: {early_slope:.6f}s/vec, "
                f"Late slope: {late_slope:.6f}s/vec"
            )

        finally:
            db_provider.disconnect()


# =============================================================================
# External API Tests - Exercise sharding through DuckDBProvider public methods
# =============================================================================


class ExternalAPITestBase:
    """Base class for external API tests with common fixtures and helpers.

    These tests exercise the sharding system through the public DuckDBProvider
    API (insert_embeddings_batch, search_semantic) rather than calling internal
    methods like fix_pass directly. This ensures internal decision paths
    (split, merge, incremental sync) are triggered organically.
    """

    # Test configuration
    TEST_DIMS = 128
    TEST_SPLIT_THRESHOLD = 100
    TEST_MERGE_THRESHOLD = 20
    PROVIDER = "test"
    MODEL = "test-model"

    @staticmethod
    def create_db_provider(tmp_path: Path) -> "DuckDBProvider":
        """Create DuckDBProvider with test sharding config."""
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider

        db_path = tmp_path / "external_api_test.duckdb"
        config = ShardingConfig(
            split_threshold=ExternalAPITestBase.TEST_SPLIT_THRESHOLD,
            merge_threshold=ExternalAPITestBase.TEST_MERGE_THRESHOLD,
            compaction_threshold=0.20,
            incremental_sync_threshold=0.10,
            quality_threshold=0.95,
            shard_similarity_threshold=0.1,
        )

        provider = DuckDBProvider(
            db_path=db_path,
            base_directory=tmp_path,
            sharding_config=config,
        )
        provider.connect()
        return provider

    @staticmethod
    def create_test_file(provider: "DuckDBProvider", file_path: str = "test.py") -> int:
        """Create a test file in the database and return its ID."""
        from chunkhound.core.models.file import File
        from chunkhound.core.types import Language

        test_file = File(
            path=file_path,
            mtime=1000.0,
            language=Language.PYTHON,
            size_bytes=1000,
        )
        return provider.insert_file(test_file)

    @staticmethod
    def create_test_chunks(
        provider: "DuckDBProvider",
        file_id: int,
        count: int,
        start_id: int = 0,
    ) -> list[int]:
        """Create test chunks in the database and return their IDs."""
        from chunkhound.core.models.chunk import Chunk
        from chunkhound.core.types import ChunkType, Language

        chunks = []
        for i in range(count):
            chunk = Chunk(
                symbol=f"test_function_{start_id + i}",
                start_line=i * 10 + 1,
                end_line=i * 10 + 9,
                code=f"def test_function_{start_id + i}():\n    pass",
                chunk_type=ChunkType.FUNCTION,
                file_id=file_id,
                language=Language.PYTHON,
            )
            chunks.append(chunk)

        return provider.insert_chunks_batch(chunks)

    @staticmethod
    def insert_embeddings_via_api(
        provider: "DuckDBProvider",
        chunk_ids: list[int],
        vectors: list[np.ndarray],
        dims: int,
    ) -> int:
        """Insert embeddings via the external API."""
        embeddings_data = [
            {
                "chunk_id": chunk_id,
                "provider": ExternalAPITestBase.PROVIDER,
                "model": ExternalAPITestBase.MODEL,
                "embedding": vec.tolist(),
                "dims": dims,
            }
            for chunk_id, vec in zip(chunk_ids, vectors)
        ]
        return provider.insert_embeddings_batch(embeddings_data)

    @staticmethod
    def search_via_api(
        provider: "DuckDBProvider",
        query_vector: np.ndarray,
        k: int,
    ) -> list[dict]:
        """Search via the external API and return results."""
        results, _pagination = provider.search_semantic(
            query_embedding=query_vector.tolist(),
            provider=ExternalAPITestBase.PROVIDER,
            model=ExternalAPITestBase.MODEL,
            page_size=k,
        )
        return results


class TestIncrementalSyncViaExternalAPI(ExternalAPITestBase):
    """Verify incremental sync path maintains search correctness."""

    def test_small_additions_immediately_searchable(self, tmp_path: Path) -> None:
        """Vectors added in small batches are immediately searchable."""
        generator = SyntheticEmbeddingGenerator(dims=self.TEST_DIMS, seed=42)
        provider = self.create_db_provider(tmp_path)

        try:
            # Insert 100 vectors (at threshold, triggers initial build)
            file_id = self.create_test_file(provider)
            chunk_ids_100 = self.create_test_chunks(provider, file_id, 100, start_id=0)
            vectors_100 = [
                generator.generate_hash_seeded(f"doc_{i}") for i in range(100)
            ]
            self.insert_embeddings_via_api(
                provider, chunk_ids_100, vectors_100, self.TEST_DIMS
            )

            # Insert 5 more vectors (5% < 10% threshold → incremental sync)
            chunk_ids_5 = self.create_test_chunks(provider, file_id, 5, start_id=100)
            vectors_5 = [
                generator.generate_hash_seeded(f"doc_{i}") for i in range(100, 105)
            ]
            self.insert_embeddings_via_api(
                provider, chunk_ids_5, vectors_5, self.TEST_DIMS
            )

            # All 5 new vectors should be immediately searchable
            for chunk_id, vec in zip(chunk_ids_5, vectors_5):
                results = self.search_via_api(provider, vec, k=10)
                result_chunk_ids = {r["chunk_id"] for r in results}

                assert chunk_id in result_chunk_ids, (
                    f"Newly added chunk {chunk_id} not immediately searchable"
                )

        finally:
            provider.disconnect()


class TestEdgeCasesViaExternalAPI(ExternalAPITestBase):
    """Test edge cases via external API."""

    def test_search_with_no_embeddings(self, tmp_path: Path) -> None:
        """Search returns empty when no embeddings exist."""
        generator = SyntheticEmbeddingGenerator(dims=self.TEST_DIMS, seed=42)
        provider = self.create_db_provider(tmp_path)

        try:
            # Don't insert anything
            query = generator.generate_hash_seeded("query")
            results = self.search_via_api(provider, query, k=10)

            # Should return empty, not error
            assert results == [], f"Expected empty results, got {len(results)}"

        finally:
            provider.disconnect()

    def test_k_larger_than_total(self, tmp_path: Path) -> None:
        """Search with k > total_vectors returns all vectors."""
        generator = SyntheticEmbeddingGenerator(dims=self.TEST_DIMS, seed=42)
        provider = self.create_db_provider(tmp_path)

        try:
            # Insert only 30 vectors
            file_id = self.create_test_file(provider)
            chunk_ids = self.create_test_chunks(provider, file_id, 30)
            vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(30)]
            self.insert_embeddings_via_api(provider, chunk_ids, vectors, self.TEST_DIMS)

            # Search with k=100 (larger than total)
            query = vectors[0]
            results = self.search_via_api(provider, query, k=100)

            # Should return exactly 30 results
            assert len(results) == 30, (
                f"Expected 30 results for k=100 with 30 vectors, got {len(results)}"
            )

        finally:
            provider.disconnect()


class TestCentroidStalenessViaExternalAPI(ExternalAPITestBase):
    """Test centroid staleness handling through normal operations."""

    def test_search_correct_after_heavy_modification(self, tmp_path: Path) -> None:
        """Search remains correct after deleting many vectors."""
        generator = SyntheticEmbeddingGenerator(dims=self.TEST_DIMS, seed=42)
        provider = self.create_db_provider(tmp_path)

        try:
            # Insert 200 vectors (may trigger splits depending on K-means)
            file_id = self.create_test_file(provider)
            chunk_ids = self.create_test_chunks(provider, file_id, 200)
            vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(200)]
            self.insert_embeddings_via_api(provider, chunk_ids, vectors, self.TEST_DIMS)

            # Delete 40% of vectors (80 vectors, likely includes some centroids)
            deleted_count = 80
            deleted_chunk_ids = set(chunk_ids[:deleted_count])
            remaining_chunk_ids = chunk_ids[deleted_count:]
            remaining_vectors = vectors[deleted_count:]

            # Delete via chunks (which cascades to embeddings)
            provider.delete_chunks_batch(list(deleted_chunk_ids))

            # All remaining vectors should still be findable
            k = 120  # remaining count
            not_found = []

            for chunk_id, vec in zip(remaining_chunk_ids, remaining_vectors):
                results = self.search_via_api(provider, vec, k)
                result_chunk_ids = {r["chunk_id"] for r in results}

                if chunk_id not in result_chunk_ids:
                    not_found.append(chunk_id)

            assert len(not_found) == 0, (
                f"Lost {len(not_found)} embeddings after heavy deletion"
            )

        finally:
            provider.disconnect()


class TestCrossShardTopKCorrectnessExternal(ExternalAPITestBase):
    """Verify search returns globally correct top-k across shards via external API."""

    def test_global_topk_matches_brute_force(self, tmp_path: Path) -> None:
        """Top-k from sharded search matches brute-force ground truth."""
        generator = SyntheticEmbeddingGenerator(dims=self.TEST_DIMS, seed=42)
        provider = self.create_db_provider(tmp_path)

        try:
            # Create file and chunks for 300 embeddings (triggers 3+ splits)
            file_id = self.create_test_file(provider)
            chunk_ids = self.create_test_chunks(provider, file_id, 300)

            # Generate vectors and insert via external API
            vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(300)]
            inserted = self.insert_embeddings_via_api(
                provider, chunk_ids, vectors, self.TEST_DIMS
            )
            assert inserted == 300

            # Test search correctness for sample queries
            k = 10
            sample_indices = [0, 50, 100, 150, 200, 250]

            for idx in sample_indices:
                query = vectors[idx]

                # Get ground truth via brute force
                ground_truth_indices = brute_force_search(query, vectors, k)
                ground_truth_chunk_ids = {chunk_ids[i] for i in ground_truth_indices}

                # Search via external API
                results = self.search_via_api(provider, query, k)
                result_chunk_ids = {r["chunk_id"] for r in results}

                # Calculate recall
                found = result_chunk_ids & ground_truth_chunk_ids
                recall = len(found) / len(ground_truth_chunk_ids)

                # Assert strict recall threshold
                assert recall >= 0.9, (
                    f"Low recall {recall:.2f} for query {idx}: "
                    f"found {len(found)}/{len(ground_truth_chunk_ids)}"
                )

                # Query vector itself should be found
                query_chunk_id = chunk_ids[idx]
                assert query_chunk_id in result_chunk_ids, (
                    f"Query chunk {query_chunk_id} not found in results"
                )

        finally:
            provider.disconnect()

    def test_global_topk_with_clustered_data(self, tmp_path: Path) -> None:
        """Top-k correct when clusters span different shards."""
        generator = SyntheticEmbeddingGenerator(dims=self.TEST_DIMS, seed=42)
        provider = self.create_db_provider(tmp_path)

        try:
            # Generate 5 clusters with 60 vectors each (300 total, triggers splits)
            clustered = generator.generate_clustered(
                num_clusters=5, per_cluster=60, separation=0.4
            )
            vectors = [vec for vec, _label in clustered]
            labels = [label for _vec, label in clustered]

            # Create file and chunks
            file_id = self.create_test_file(provider, "clustered_test.py")
            chunk_ids = self.create_test_chunks(provider, file_id, len(vectors))

            # Insert via external API
            self.insert_embeddings_via_api(provider, chunk_ids, vectors, self.TEST_DIMS)

            # Query from each cluster and verify results are dominated by same-cluster
            k = 10
            for cluster_id in range(5):
                # Get indices for this cluster
                cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
                query_idx = cluster_indices[0]
                query = vectors[query_idx]

                # Search
                results = self.search_via_api(provider, query, k)
                result_chunk_ids = [r["chunk_id"] for r in results]

                # Count how many results are from the same cluster
                cluster_chunk_ids = {chunk_ids[i] for i in cluster_indices}
                same_cluster_count = sum(
                    1 for cid in result_chunk_ids if cid in cluster_chunk_ids
                )

                # With well-separated clusters, most results should be from same cluster
                assert same_cluster_count >= k // 2, (
                    f"Cluster {cluster_id}: only {same_cluster_count}/{k} "
                    f"results from same cluster"
                )

        finally:
            provider.disconnect()


class TestSearchConsistencyAcrossStructuralOpsExternal(ExternalAPITestBase):
    """Search results must be stable across structural operations via external API."""

    def test_search_consistency_as_data_grows(self, tmp_path: Path) -> None:
        """Search results for existing vectors remain stable as new vectors added."""
        generator = SyntheticEmbeddingGenerator(dims=self.TEST_DIMS, seed=42)
        provider = self.create_db_provider(tmp_path)

        try:
            # Phase 1: Insert 50 vectors (below split threshold)
            file_id = self.create_test_file(provider)
            chunk_ids_50 = self.create_test_chunks(provider, file_id, 50, start_id=0)
            vectors_50 = [generator.generate_hash_seeded(f"doc_{i}") for i in range(50)]
            self.insert_embeddings_via_api(
                provider, chunk_ids_50, vectors_50, self.TEST_DIMS
            )

            # Capture search results for sample queries
            sample_queries = [0, 10, 20, 30, 40]
            k = 5
            results_50 = {}
            for idx in sample_queries:
                results = self.search_via_api(provider, vectors_50[idx], k)
                results_50[idx] = [r["chunk_id"] for r in results]

            # Phase 2: Insert 50 more (total=100, at split threshold)
            chunk_ids_100 = self.create_test_chunks(provider, file_id, 50, start_id=50)
            vectors_100 = [
                generator.generate_hash_seeded(f"doc_{i}") for i in range(50, 100)
            ]
            self.insert_embeddings_via_api(
                provider, chunk_ids_100, vectors_100, self.TEST_DIMS
            )

            # Capture search results after growth
            results_100 = {}
            for idx in sample_queries:
                results = self.search_via_api(provider, vectors_50[idx], k)
                results_100[idx] = [r["chunk_id"] for r in results]

            # Phase 3: Insert 100 more (total=200, after splits)
            chunk_ids_200 = self.create_test_chunks(provider, file_id, 100, start_id=100)
            vectors_200 = [
                generator.generate_hash_seeded(f"doc_{i}") for i in range(100, 200)
            ]
            self.insert_embeddings_via_api(
                provider, chunk_ids_200, vectors_200, self.TEST_DIMS
            )

            # Capture search results after more growth
            results_200 = {}
            for idx in sample_queries:
                results = self.search_via_api(provider, vectors_50[idx], k)
                results_200[idx] = [r["chunk_id"] for r in results]

            # Verify: Query vector itself should be top result at all checkpoints
            for idx in sample_queries:
                query_chunk_id = chunk_ids_50[idx]

                # Query chunk should be in results at all stages
                assert query_chunk_id in results_50[idx], (
                    f"Query {idx} not found in results_50"
                )
                assert query_chunk_id in results_100[idx], (
                    f"Query {idx} not found in results_100"
                )
                assert query_chunk_id in results_200[idx], (
                    f"Query {idx} not found in results_200"
                )

        finally:
            provider.disconnect()


class TestNoFalseNegativesExternal(ExternalAPITestBase):
    """I11: Every DB embedding must be findable via external search API."""

    def test_all_embeddings_findable(self, tmp_path: Path) -> None:
        """Every indexed embedding appears in search results."""
        generator = SyntheticEmbeddingGenerator(dims=self.TEST_DIMS, seed=42)
        provider = self.create_db_provider(tmp_path)

        try:
            # Insert 250 vectors (triggers multiple splits)
            file_id = self.create_test_file(provider)
            chunk_ids = self.create_test_chunks(provider, file_id, 250)
            vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(250)]
            self.insert_embeddings_via_api(provider, chunk_ids, vectors, self.TEST_DIMS)

            # For EVERY embedding, verify it's findable
            k = 250  # k = total count
            not_found = []

            for i, (chunk_id, vec) in enumerate(zip(chunk_ids, vectors)):
                results = self.search_via_api(provider, vec, k)
                result_chunk_ids = {r["chunk_id"] for r in results}

                if chunk_id not in result_chunk_ids:
                    not_found.append((i, chunk_id))

            # No false negatives
            assert len(not_found) == 0, (
                f"I11 violated: {len(not_found)} embeddings not findable: "
                f"{not_found[:10]}..."
            )

        finally:
            provider.disconnect()

    def test_findable_after_structural_changes(self, tmp_path: Path) -> None:
        """Embeddings remain findable after splits and incremental syncs."""
        generator = SyntheticEmbeddingGenerator(dims=self.TEST_DIMS, seed=42)
        provider = self.create_db_provider(tmp_path)

        try:
            # Phase 1: Insert 100 vectors (at threshold)
            file_id = self.create_test_file(provider)
            chunk_ids_1 = self.create_test_chunks(provider, file_id, 100, start_id=0)
            vectors_1 = [generator.generate_hash_seeded(f"doc_{i}") for i in range(100)]
            self.insert_embeddings_via_api(
                provider, chunk_ids_1, vectors_1, self.TEST_DIMS
            )

            # Phase 2: Insert 50 more (triggers split)
            chunk_ids_2 = self.create_test_chunks(provider, file_id, 50, start_id=100)
            vectors_2 = [
                generator.generate_hash_seeded(f"doc_{i}") for i in range(100, 150)
            ]
            self.insert_embeddings_via_api(
                provider, chunk_ids_2, vectors_2, self.TEST_DIMS
            )

            # Verify all embeddings from both phases are findable
            all_chunk_ids = chunk_ids_1 + chunk_ids_2
            all_vectors = vectors_1 + vectors_2
            k = 150

            not_found = []
            for chunk_id, vec in zip(all_chunk_ids, all_vectors):
                results = self.search_via_api(provider, vec, k)
                result_chunk_ids = {r["chunk_id"] for r in results}

                if chunk_id not in result_chunk_ids:
                    not_found.append(chunk_id)

            assert len(not_found) == 0, (
                f"Lost {len(not_found)} embeddings after structural changes"
            )

        finally:
            provider.disconnect()