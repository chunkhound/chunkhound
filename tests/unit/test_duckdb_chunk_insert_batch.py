"""Contract: insert_chunks_batch() respects _CHUNK_INSERT_BATCH_SIZE cap."""
from unittest.mock import MagicMock


class TestChunkInsertBatchSizeCap:
    """Contract: _CHUNK_INSERT_BATCH_SIZE is a positive int; large batches are split."""

    def test_batch_size_constant_exists_and_is_positive(self):
        from chunkhound.providers.database import duckdb_provider

        assert hasattr(duckdb_provider, "_CHUNK_INSERT_BATCH_SIZE")
        assert isinstance(duckdb_provider._CHUNK_INSERT_BATCH_SIZE, int)
        assert duckdb_provider._CHUNK_INSERT_BATCH_SIZE > 0

    def test_oversized_batch_split_returns_all_ids(self):
        """When len(chunks) > _CHUNK_INSERT_BATCH_SIZE, all IDs must be returned."""
        from chunkhound.providers.database import duckdb_provider
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider

        cap = duckdb_provider._CHUNK_INSERT_BATCH_SIZE
        n_chunks = cap + 1  # one more than the cap

        chunks = [MagicMock() for _ in range(n_chunks)]

        call_sizes = []

        def fake_execute(method_name, sub_chunks):
            call_sizes.append(len(sub_chunks))
            return list(range(len(sub_chunks)))  # fake IDs

        provider = MagicMock(spec=DuckDBProvider)
        provider._execute_in_db_thread_sync = fake_execute

        ids = DuckDBProvider.insert_chunks_batch(provider, chunks)

        assert len(ids) == n_chunks, f"Expected {n_chunks} IDs, got {len(ids)}"
        assert len(call_sizes) >= 2, "Expected sub-batching but got single call"
        assert max(call_sizes) <= cap, f"Sub-batch exceeded cap: {max(call_sizes)}"

    def test_exact_cap_size_is_single_call(self):
        """Exactly _CHUNK_INSERT_BATCH_SIZE chunks must use a single executor call."""
        from chunkhound.providers.database import duckdb_provider
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider

        cap = duckdb_provider._CHUNK_INSERT_BATCH_SIZE
        chunks = [MagicMock() for _ in range(cap)]

        call_sizes = []

        def fake_execute(method_name, sub_chunks):
            call_sizes.append(len(sub_chunks))
            return list(range(len(sub_chunks)))

        provider = MagicMock(spec=DuckDBProvider)
        provider._execute_in_db_thread_sync = fake_execute

        ids = DuckDBProvider.insert_chunks_batch(provider, chunks)

        assert len(ids) == cap
        assert len(call_sizes) == 1, "Exactly-cap batch should not be split"

    def test_empty_list_returns_empty(self):
        """Empty input must return empty list without calling executor."""
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider

        provider = MagicMock(spec=DuckDBProvider)
        ids = DuckDBProvider.insert_chunks_batch(provider, [])
        assert ids == []
        provider._execute_in_db_thread_sync.assert_not_called()

    def test_async_oversized_batch_split_returns_all_ids(self):
        """insert_chunks_batch_async must also split oversized batches."""
        import asyncio
        from chunkhound.providers.database import duckdb_provider
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider

        cap = duckdb_provider._CHUNK_INSERT_BATCH_SIZE
        n_chunks = cap + 1

        chunks = [MagicMock() for _ in range(n_chunks)]

        call_sizes = []

        async def fake_execute_async(method_name, sub_chunks):
            call_sizes.append(len(sub_chunks))
            return list(range(len(sub_chunks)))

        provider = MagicMock(spec=DuckDBProvider)
        provider._execute_in_db_thread = fake_execute_async

        ids = asyncio.run(DuckDBProvider.insert_chunks_batch_async(provider, chunks))

        assert len(ids) == n_chunks, f"Expected {n_chunks} IDs, got {len(ids)}"
        assert len(call_sizes) >= 2, "Expected sub-batching but got single call"
        assert max(call_sizes) <= cap, f"Sub-batch exceeded cap: {max(call_sizes)}"
