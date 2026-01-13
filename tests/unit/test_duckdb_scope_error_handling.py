"""DuckDB scope helpers should fail gracefully on executor errors."""

from chunkhound.providers.database.duckdb_provider import DuckDBProvider


class _ExplodingConn:
    def execute(self, *args, **kwargs):  # noqa: ANN001 - test stub
        raise RuntimeError("boom")


def test_duckdb_scope_stats_handles_errors() -> None:
    provider = DuckDBProvider.__new__(DuckDBProvider)
    total_files, total_chunks = provider._executor_get_scope_stats(
        _ExplodingConn(), {}, "scope/"
    )
    assert total_files == 0
    assert total_chunks == 0


def test_duckdb_scope_paths_handles_errors() -> None:
    provider = DuckDBProvider.__new__(DuckDBProvider)
    paths = provider._executor_get_scope_file_paths(_ExplodingConn(), {}, "scope/")
    assert paths == []
