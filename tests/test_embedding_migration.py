"""Contract tests for the client-side truncation migration path.

Verifies the invariants a user would notice if broken:
1. Full-length vectors are read, truncated, and written without API calls.
2. Only the target-dimension table is checked when looking for existing embeddings.
3. The source table is dropped when requested.
"""

import math

import pytest

from chunkhound.core.models.embedding import Embedding
from chunkhound.providers.database.duckdb_provider import DuckDBProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vector(dims: int, seed: int = 1) -> list[float]:
    """Return a deterministic unit-norm vector of the given length."""
    raw = [float((i + seed) % 17 + 1) for i in range(dims)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _l2_norm(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


def _make_db(tmp_path) -> DuckDBProvider:
    """Create a fully-initialized in-memory DuckDBProvider."""
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()
    return db


def _insert_embedding(db: DuckDBProvider, chunk_id: int, dims: int,
                       provider: str = "openai", model: str = "text-embedding-3-small") -> None:
    """Insert a single embedding record via the high-level provider API."""
    db.insert_embedding(
        Embedding(
            chunk_id=chunk_id,
            provider=provider,
            model=model,
            dims=dims,
            vector=_unit_vector(dims, seed=chunk_id),
        )
    )


# ---------------------------------------------------------------------------
# Contract 1 & 2 & variant: get_full_embeddings_for_migration
# ---------------------------------------------------------------------------

class TestGetFullEmbeddingsForMigration:

    def test_returns_empty_when_no_larger_table_exists(self, tmp_path):
        """No source table larger than target_dims → ([], None)."""
        db = _make_db(tmp_path)
        records, source_dims = db.get_full_embeddings_for_migration(
            [1, 2, 3], "openai", "text-embedding-3-small", target_dims=256
        )
        assert records == []
        assert source_dims is None

    def test_reads_vectors_from_source_table(self, tmp_path):
        """When embeddings_1536 exists, reads matching chunk_ids and returns full vector."""
        db = _make_db(tmp_path)
        _insert_embedding(db, chunk_id=42, dims=1536)

        records, source_dims = db.get_full_embeddings_for_migration(
            [42], "openai", "text-embedding-3-small", target_dims=256
        )

        assert source_dims == 1536
        assert len(records) == 1
        assert records[0]["chunk_id"] == 42
        assert len(records[0]["embedding"]) == 1536

    def test_ignores_chunk_ids_not_in_request(self, tmp_path):
        """Returns only the chunk_ids that are in the requested list."""
        db = _make_db(tmp_path)
        _insert_embedding(db, chunk_id=42, dims=1536)
        _insert_embedding(db, chunk_id=99, dims=1536)

        records, source_dims = db.get_full_embeddings_for_migration(
            [42], "openai", "text-embedding-3-small", target_dims=256
        )

        assert source_dims == 1536
        assert len(records) == 1
        assert records[0]["chunk_id"] == 42

    def test_returns_source_dims_with_empty_chunk_ids_for_discovery(self, tmp_path):
        """Empty chunk_ids returns ([], source_dims) for table discovery without reading rows."""
        db = _make_db(tmp_path)
        _insert_embedding(db, chunk_id=42, dims=1536)

        records, source_dims = db.get_full_embeddings_for_migration(
            [], "openai", "text-embedding-3-small", target_dims=256
        )

        assert records == []
        assert source_dims == 1536


# ---------------------------------------------------------------------------
# Contract 3 & 4: drop_embedding_table
# ---------------------------------------------------------------------------

class TestDropEmbeddingTable:

    def test_drops_existing_table(self, tmp_path):
        """Table is gone after drop — subsequent queries find no rows."""
        db = _make_db(tmp_path)
        _insert_embedding(db, chunk_id=1, dims=1536)

        # Verify the row is present before drop
        records_before, _ = db.get_full_embeddings_for_migration(
            [1], "openai", "text-embedding-3-small", target_dims=256
        )
        assert len(records_before) == 1

        db.drop_embedding_table(1536)

        # After drop, no source table exists → empty result
        records_after, source_dims_after = db.get_full_embeddings_for_migration(
            [1], "openai", "text-embedding-3-small", target_dims=256
        )
        assert records_after == []
        assert source_dims_after is None

    def test_drop_nonexistent_table_is_noop(self, tmp_path):
        """No error when dropping a table that doesn't exist."""
        db = _make_db(tmp_path)
        # Should not raise — idempotent
        db.drop_embedding_table(9999)


# ---------------------------------------------------------------------------
# Contract 5: apply_client_side_truncation
# ---------------------------------------------------------------------------

class TestTruncationCorrectness:

    def test_truncation_produces_unit_norm_vector(self):
        """Truncated vector is L2-normalized to unit length."""
        from chunkhound.providers.embeddings.shared_utils import apply_client_side_truncation

        full = [float(i) for i in range(1, 1537)]
        result = apply_client_side_truncation([full], 256)
        truncated = result[0]

        assert len(truncated) == 256
        norm = math.sqrt(sum(x * x for x in truncated))
        assert abs(norm - 1.0) < 1e-5

    def test_truncation_uses_first_n_dims(self):
        """Truncated vector uses the first N components of the full vector, then normalizes."""
        from chunkhound.providers.embeddings.shared_utils import apply_client_side_truncation

        full = [float(i) for i in range(100)]
        result = apply_client_side_truncation([full], 10)
        truncated = result[0]

        expected = _l2_norm([float(i) for i in range(10)])
        assert len(truncated) == 10
        for a, b in zip(truncated, expected):
            assert abs(a - b) < 1e-5


# ---------------------------------------------------------------------------
# Contract 6: get_existing_embeddings_in_table
# ---------------------------------------------------------------------------

class TestGetExistingEmbeddingsInTable:

    def test_returns_empty_when_table_does_not_exist(self, tmp_path):
        """When the target-dims table doesn't exist at all, returns empty set."""
        db = _make_db(tmp_path)
        result = db.get_existing_embeddings_in_table(
            [1, 2, 3], "openai", "text-embedding-3-small", dims=256
        )
        assert result == set()

    def test_returns_only_chunks_in_specific_table(self, tmp_path):
        """Chunk in embeddings_1536 is NOT returned when checking embeddings_256."""
        db = _make_db(tmp_path)
        _insert_embedding(db, chunk_id=42, dims=1536)

        # Check the 256-dim table — chunk 42 is only in the 1536 table
        result = db.get_existing_embeddings_in_table(
            [42], "openai", "text-embedding-3-small", dims=256
        )
        assert result == set()

    def test_returns_chunks_when_table_matches(self, tmp_path):
        """Chunk in embeddings_256 IS returned when checking embeddings_256."""
        db = _make_db(tmp_path)
        _insert_embedding(db, chunk_id=42, dims=256)

        result = db.get_existing_embeddings_in_table(
            [42], "openai", "text-embedding-3-small", dims=256
        )
        assert result == {42}

    def test_filters_to_requested_chunk_ids(self, tmp_path):
        """Only chunk_ids in the request are returned, even if the table has more."""
        db = _make_db(tmp_path)
        _insert_embedding(db, chunk_id=10, dims=256)
        _insert_embedding(db, chunk_id=20, dims=256)

        result = db.get_existing_embeddings_in_table(
            [10], "openai", "text-embedding-3-small", dims=256
        )
        assert result == {10}


# ---------------------------------------------------------------------------
# Contract 7: EmbeddingService._migrate_from_full_embeddings (integration)
# ---------------------------------------------------------------------------

class TestMigrateFromFullEmbeddingsMethod:
    """Contract tests for EmbeddingService._migrate_from_full_embeddings."""

    def _make_provider_mock(self, name="openai", model="text-embedding-3-small"):
        """Create a minimal mock embedding provider."""
        from unittest.mock import MagicMock
        provider = MagicMock()
        provider.name = name
        provider.model = model
        return provider

    @pytest.mark.asyncio
    async def test_migrates_full_vectors_to_truncated_table(self, tmp_path):
        """Full vectors in source table are truncated and written to target table."""
        from chunkhound.services.embedding_service import EmbeddingService

        db = _make_db(tmp_path)
        provider = self._make_provider_mock()

        # Insert a 1536-dim embedding into the source table using the existing helper
        _insert_embedding(db, chunk_id=1, dims=1536)

        # Build a minimal EmbeddingService without full __init__
        service = EmbeddingService.__new__(EmbeddingService)
        service._db = db
        service._embedding_provider = provider
        service._db_batch_size = 100

        # Run migration from 1536-dim source to 256-dim target
        migrated = await service._migrate_from_full_embeddings([1], target_dims=256)

        # chunk 1 must be reported as migrated
        assert 1 in migrated

        # The truncated embedding must now exist in embeddings_256
        existing_256 = db.get_existing_embeddings_in_table(
            [1], "openai", "text-embedding-3-small", dims=256
        )
        assert existing_256 == {1}

        # The original full-size embedding must still exist in embeddings_1536
        records, source_dims = db.get_full_embeddings_for_migration(
            [1], "openai", "text-embedding-3-small", target_dims=256
        )
        assert source_dims == 1536
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_set_when_no_source_data(self, tmp_path):
        """Returns empty set when no full-size embeddings exist to migrate."""
        from chunkhound.services.embedding_service import EmbeddingService

        db = _make_db(tmp_path)
        provider = self._make_provider_mock()

        service = EmbeddingService.__new__(EmbeddingService)
        service._db = db
        service._embedding_provider = provider
        service._db_batch_size = 100

        migrated = await service._migrate_from_full_embeddings([1, 2, 3], target_dims=256)
        assert migrated == set()
