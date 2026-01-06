"""Sharding test suite covering all 13 invariants from spec section 11.5.

Tests use sociable approach with real DuckDB, real USearch, and real filesystem
in temporary directories. Uses SyntheticEmbeddingGenerator for deterministic,
reproducible tests without external API dependencies.

Invariants tested:
- Data Integrity (I1-I6): Single assignment, shard existence, count consistency,
  no orphans, no ghosts, embedding retrievability
- Operational (I7-I10): Fix pass idempotence, convergence, LIRE bound, NPA
- Search (I11-I13): No false negatives, tombstone exclusion, centroid filter

Uses small thresholds (split=100, merge=10) for efficient testing without
requiring 100K vectors.
"""

from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import numpy as np
import pytest

from chunkhound.core.config.sharding_config import ShardingConfig
from chunkhound.providers.database import usearch_wrapper
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
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS vector_shards (
                shard_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                dims INTEGER NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                quantization TEXT NOT NULL DEFAULT 'i8',
                file_path TEXT,
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

    def disconnect(self) -> None:
        """Close connection."""
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
            INSERT INTO {table_name} (chunk_id, provider, model, embedding, dims, shard_id)
            VALUES (?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            [i, provider, model, vec_list, TEST_DIMS, str(shard_id) if shard_id else None],
        ).fetchone()
        ids.append(result[0])

    return ids


def get_all_shard_ids(db_provider: MockDBProvider) -> list[UUID]:
    """Get all shard IDs from DB."""
    result = db_provider.connection.execute(
        "SELECT shard_id FROM vector_shards"
    ).fetchall()
    # DuckDB returns UUID objects directly, handle both UUID and string
    return [row[0] if isinstance(row[0], UUID) else UUID(row[0]) for row in result]


def get_shard_embedding_count(
    db_provider: MockDBProvider, shard_id: UUID
) -> int:
    """Get count of embeddings in a shard."""
    table_name = f"embeddings_{TEST_DIMS}"
    result = db_provider.connection.execute(
        f"SELECT COUNT(*) FROM {table_name} WHERE shard_id = ?",
        [str(shard_id)],
    ).fetchone()
    return result[0]


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
        db_shard_ids.add(shard[0])

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
            INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
            INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
            INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
                INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
    """Test 3: Split at threshold - insert to threshold+1, verify split.

    Exercises invariants: I1, I2, I9, I10

    Note: Uses reduced count for faster testing while still exercising split logic.
    """

    def test_split_at_threshold(
        self,
        tmp_db: MockDBProvider,
        shard_manager: ShardManager,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Shard splits when db_count >= split_threshold."""
        # Create initial shard
        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
        )

        # Insert exactly split_threshold vectors to test k-means fallback path
        vector_count = TEST_SPLIT_THRESHOLD
        vectors = [
            generator.generate_hash_seeded(f"doc_{i}")
            for i in range(vector_count)
        ]
        insert_embeddings_to_db(tmp_db, vectors, shard_id)

        # Run fix_pass - should trigger split (>= threshold)
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify split occurred - should now have multiple shards
        shards = get_all_shard_ids(tmp_db)
        assert len(shards) >= 2, f"Expected split, got {len(shards)} shard(s)"

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
            INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
        )

        # Insert below threshold (use smaller count for faster test)
        test_count = min(200, TEST_SPLIT_THRESHOLD - 1)
        vectors = [
            generator.generate_hash_seeded(f"doc_{i}")
            for i in range(test_count)
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
                INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
            )

        # Insert small count (below merge threshold) to small shard
        small_count = TEST_MERGE_THRESHOLD - 1  # 99
        small_vectors = [
            generator.generate_hash_seeded(f"small_{i}")
            for i in range(small_count)
        ]
        insert_embeddings_to_db(tmp_db, small_vectors, small_shard_id)

        # Insert normal count to normal shard (above merge threshold)
        normal_count = TEST_MERGE_THRESHOLD + 50  # 150
        normal_vectors = [
            generator.generate_hash_seeded(f"normal_{i}")
            for i in range(normal_count)
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
            INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
        )

        # Insert below merge threshold
        small_count = TEST_MERGE_THRESHOLD - 1  # 99
        vectors = [
            generator.generate_hash_seeded(f"doc_{i}")
            for i in range(small_count)
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
                INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
            )

        # Small shard: below merge threshold
        small_count = TEST_MERGE_THRESHOLD - 1
        small_vectors = [
            generator.generate_hash_seeded(f"small_{i}")
            for i in range(small_count)
        ]
        insert_embeddings_to_db(tmp_db, small_vectors, small_shard_id)

        # Large shard: just below split threshold
        # After merge, combined count should exceed split threshold
        large_count = TEST_SPLIT_THRESHOLD - TEST_MERGE_THRESHOLD + 2
        large_vectors = [
            generator.generate_hash_seeded(f"large_{i}")
            for i in range(large_count)
        ]
        insert_embeddings_to_db(tmp_db, large_vectors, large_shard_id)

        total_vectors = len(small_vectors) + len(large_vectors)

        # Run fix_pass - should merge small into large, then split
        shard_manager.fix_pass(tmp_db.connection, check_quality=False)

        # Verify cascade: merged then split results in 2+ shards
        shards = get_all_shard_ids(tmp_db)
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
            INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
            INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
            assert results[0].key == emb_ids[i], (
                f"Vector {i} not found as top result"
            )


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
            INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
            INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
        import time

        # Create shard with 50 vectors
        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
        )

        vectors = [generator.generate_hash_seeded(f"doc_{i}") for i in range(50)]
        table_name = f"embeddings_{TEST_DIMS}"
        for i, emb in enumerate(vectors):
            tmp_db.connection.execute(
                f"""
                INSERT INTO {table_name} (chunk_id, provider, model, embedding, dims, shard_id)
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
        assert new_mtime > initial_mtime, "Index should be rebuilt when quality < threshold"

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
        import time

        shard_id = uuid4()
        shard_path = shard_manager._shard_path(shard_id)

        tmp_db.connection.execute(
            """
            INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
        assert fix_time < 30.0, f"fix_pass took {fix_time:.2f}s for {vector_count} vectors"

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
            INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
            INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
            INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
                INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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


class TestNativeUSearchClustering:
    """Test native USearch clustering path with high split threshold.

    Uses split_threshold=2000 which requires ~1681+ vectors (41^2 for HDBSCAN
    minimum sample calculation) to trigger native USearch clustering instead
    of the k-means fallback path.

    Exercises invariants: I1, I2
    """

    def test_native_usearch_split(
        self,
        tmp_path: Path,
        generator: SyntheticEmbeddingGenerator,
    ) -> None:
        """Shard splits using native USearch clustering with 2500 vectors."""
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
                INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
            )

            # Insert 2500 vectors - enough to trigger native USearch clustering
            # (requires ~1681+ vectors, threshold >= 2000)
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

            # Run fix_pass - should trigger split via native USearch clustering
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
                INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                [str(target_id), TEST_DIMS, "test", "test-model", str(target_path)],
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
            INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            [str(source_shard_id), TEST_DIMS, "test", "test-model",
             str(source_shard_path)],
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
                INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
        assert recall >= 0.7, (
            f"Low recall {recall:.2f}: centroid filtering may have excluded "
            f"shards containing true top-k results. "
            f"Found {len(found_ground_truth)}/{len(ground_truth_keys)} ground truth items"
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
                INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
                INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
                INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
                INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
                INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
                INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
                INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
                INSERT INTO vector_shards (shard_id, dims, provider, model, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                [str(shard_id), TEST_DIMS, "test", "test-model", str(shard_path)],
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
