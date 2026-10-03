"""Contract: generate_missing_embeddings uses targeted SQL, not full-table scans.

Issue #411 — on a 36K-chunk database, generate_missing_embeddings() called
get_all_chunks_with_metadata() up to three times (loading every chunk's full
source text and json.loads per row), blocking the asyncio event loop for
minutes and starving all concurrent MCP requests.

The contract: given a DuckDB with N chunks where some already have embeddings,
generate_missing_embeddings must:
  1. Embed exactly the chunks that lack embeddings (correctness).
  2. Never call get_all_chunks_with_metadata (performance regression guard).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from tests.contracts.mock_embed import MockEmbeddingProvider

_TOTAL_CHUNKS = 20
_PRE_EMBEDDED = 12


def _seed_db(provider, create_embedding_table: bool = True):
    """Insert files, chunks, and partial embeddings into a fresh DuckDB."""
    provider.connection.execute(
        "INSERT INTO files (id, path, name, content_hash) "
        "VALUES (1, 'src/app.py', 'app.py', 'aaa')"
    )

    for cid in range(1, _TOTAL_CHUNKS + 1):
        provider.connection.execute(
            "INSERT INTO chunks "
            "(id, file_id, code, symbol, chunk_type, start_line, end_line) "
            "VALUES (?, 1, ?, ?, 'function', ?, ?)",
            [cid, f"def func_{cid}(): pass", f"func_{cid}", cid, cid],
        )

    if not create_embedding_table:
        return

    dims = MockEmbeddingProvider.dims
    provider._ensure_embedding_table_exists(dims)

    for cid in range(1, _PRE_EMBEDDED + 1):
        vec = [float(cid) / 100.0] * dims
        provider.connection.execute(
            f"INSERT INTO embeddings_{dims} "
            "(chunk_id, provider, model, embedding, dims) "
            "VALUES (?, ?, ?, ?, ?)",
            [cid, MockEmbeddingProvider.name, MockEmbeddingProvider.model, vec, dims],
        )


class TestMissingEmbeddingsTargetedQuery:
    """generate_missing_embeddings must find gaps via SQL, not full scans."""

    @pytest.mark.asyncio
    async def test_fresh_database_without_embedding_table_uses_targeted_query(
        self, tmp_path: Path
    ):
        """A new DuckDB still uses ID-only SQL when no embedding table exists."""
        pytest.importorskip("duckdb")
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider
        from chunkhound.services.embedding_service import EmbeddingService

        provider = DuckDBProvider(
            db_path=tmp_path / "test.duckdb", base_directory=tmp_path
        )
        provider.connect()

        try:
            _seed_db(provider, create_embedding_table=False)
            service = EmbeddingService(
                database_provider=provider,
                embedding_provider=MockEmbeddingProvider(),
            )

            with patch.object(
                provider,
                "get_all_chunks_with_metadata",
                side_effect=AssertionError(
                    "fresh DuckDB must not load full chunk records"
                ),
            ):
                result = await service.generate_missing_embeddings()

            assert result["status"] == "success", f"unexpected status: {result}"
            assert result["generated"] == _TOTAL_CHUNKS
        finally:
            provider.disconnect(skip_checkpoint=True)

    def test_duckdb_query_errors_are_not_retried_as_full_scans(self, tmp_path: Path):
        """A DuckDB error must surface rather than silently switching to a full scan."""
        pytest.importorskip("duckdb")
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider
        from chunkhound.services.embedding_service import EmbeddingService

        provider = DuckDBProvider(
            db_path=tmp_path / "test.duckdb", base_directory=tmp_path
        )
        provider.connect()
        service = EmbeddingService(
            database_provider=provider,
            embedding_provider=MockEmbeddingProvider(),
        )

        try:
            with (
                patch.object(
                    service,
                    "_get_chunks_by_ids_sql",
                    side_effect=RuntimeError("DuckDB read failed"),
                ),
                patch.object(service, "_get_chunks_by_ids_fallback") as fallback,
            ):
                with pytest.raises(RuntimeError, match="DuckDB read failed"):
                    service._get_chunks_by_ids([1])
                fallback.assert_not_called()
        finally:
            provider.disconnect(skip_checkpoint=True)

    @pytest.mark.asyncio
    async def test_embeds_only_missing_chunks_without_full_scan(self, tmp_path: Path):
        pytest.importorskip("duckdb")
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider
        from chunkhound.services.embedding_service import EmbeddingService

        db_path = tmp_path / "test.duckdb"
        provider = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        provider.connect()

        try:
            _seed_db(provider)

            mock_embed = MockEmbeddingProvider()
            service = EmbeddingService(
                database_provider=provider,
                embedding_provider=mock_embed,
            )

            with patch.object(
                provider,
                "get_all_chunks_with_metadata",
                side_effect=AssertionError(
                    "get_all_chunks_with_metadata must not be called — "
                    "use targeted SQL instead"
                ),
            ):
                result = await service.generate_missing_embeddings()

            assert result["status"] == "success", f"unexpected status: {result}"

            expected_missing = _TOTAL_CHUNKS - _PRE_EMBEDDED
            assert result["generated"] == expected_missing, (
                f"expected {expected_missing} embeddings, got {result['generated']}"
            )

            dims = mock_embed.dims
            rows = provider.execute_query(
                f"SELECT chunk_id FROM embeddings_{dims} "
                f"WHERE provider = ? AND model = ? ORDER BY chunk_id",
                [mock_embed.name, mock_embed.model],
            )
            embedded_ids = {row["chunk_id"] for row in rows}

            assert embedded_ids == set(range(1, _TOTAL_CHUNKS + 1)), (
                f"all {_TOTAL_CHUNKS} chunks should have embeddings; "
                f"got {sorted(embedded_ids)}"
            )

        finally:
            provider.disconnect(skip_checkpoint=True)

    @pytest.mark.asyncio
    async def test_returns_complete_when_all_embedded(self, tmp_path: Path):
        """No missing chunks → status 'complete', zero generated, no full scan."""
        pytest.importorskip("duckdb")
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider
        from chunkhound.services.embedding_service import EmbeddingService

        db_path = tmp_path / "test.duckdb"
        provider = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        provider.connect()

        try:
            _seed_db(provider)

            mock_embed = MockEmbeddingProvider()
            dims = mock_embed.dims

            for cid in range(_PRE_EMBEDDED + 1, _TOTAL_CHUNKS + 1):
                vec = [float(cid) / 100.0] * dims
                provider.connection.execute(
                    f"INSERT INTO embeddings_{dims} "
                    "(chunk_id, provider, model, embedding, dims) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [cid, mock_embed.name, mock_embed.model, vec, dims],
                )

            service = EmbeddingService(
                database_provider=provider,
                embedding_provider=mock_embed,
            )

            with patch.object(
                provider,
                "get_all_chunks_with_metadata",
                side_effect=AssertionError(
                    "get_all_chunks_with_metadata must not be called"
                ),
            ):
                result = await service.generate_missing_embeddings()

            assert result["status"] == "complete"
            assert result["generated"] == 0

        finally:
            provider.disconnect(skip_checkpoint=True)

    @pytest.mark.parametrize("exclude_pattern", ["vendor/*", "vendor/[l]ib.py"])
    @pytest.mark.asyncio
    async def test_exclude_patterns_filter_via_sql(
        self, tmp_path: Path, exclude_pattern: str
    ):
        """Glob filtering preserves fnmatch semantics without loading chunk contents."""
        pytest.importorskip("duckdb")
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider
        from chunkhound.services.embedding_service import EmbeddingService

        db_path = tmp_path / "test.duckdb"
        provider = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        provider.connect()

        try:
            provider.connection.execute(
                "INSERT INTO files (id, path, name, content_hash) VALUES "
                "(1, 'src/app.py', 'app.py', 'aaa'), "
                "(2, 'vendor/lib.py', 'lib.py', 'bbb')"
            )
            for cid, fid in [(1, 1), (2, 1), (3, 2), (4, 2)]:
                provider.connection.execute(
                    "INSERT INTO chunks "
                    "(id, file_id, code, symbol, chunk_type, "
                    "start_line, end_line) "
                    "VALUES (?, ?, ?, ?, 'function', 1, 1)",
                    [cid, fid, f"def f{cid}(): pass", f"f{cid}"],
                )

            mock_embed = MockEmbeddingProvider()
            provider._ensure_embedding_table_exists(mock_embed.dims)
            service = EmbeddingService(
                database_provider=provider,
                embedding_provider=mock_embed,
            )

            with patch.object(
                provider,
                "get_all_chunks_with_metadata",
                side_effect=AssertionError(
                    "get_all_chunks_with_metadata must not be called"
                ),
            ):
                result = await service.generate_missing_embeddings(
                    exclude_patterns=[exclude_pattern]
                )

            assert result["status"] == "success", f"unexpected result: {result}"
            assert result["generated"] == 2, (
                f"only src/app.py chunks should be embedded; got {result['generated']}"
            )

            dims = mock_embed.dims
            rows = provider.execute_query(
                f"SELECT chunk_id FROM embeddings_{dims} ORDER BY chunk_id",
                [],
            )
            embedded_ids = {row["chunk_id"] for row in rows}
            assert embedded_ids == {1, 2}, (
                "only chunk IDs 1,2 (src/app.py) should be "
                f"embedded; got {sorted(embedded_ids)}"
            )

        finally:
            provider.disconnect(skip_checkpoint=True)
