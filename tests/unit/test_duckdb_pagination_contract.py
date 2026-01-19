from __future__ import annotations

from chunkhound.providers.database.duckdb_provider import DuckDBProvider


class _UnusedConn:
    pass


def test_duckdb_semantic_early_return_includes_next_offset() -> None:
    provider = DuckDBProvider.__new__(DuckDBProvider)

    # Avoid relying on instance initialization for this early-return path.
    provider._validate_and_normalize_path_filter = (  # type: ignore[method-assign]
        lambda self, path: path
    )
    provider._executor_table_exists = (  # type: ignore[method-assign]
        lambda self, conn, state, table_name: False
    )

    results, pagination = provider._executor_search_semantic(  # type: ignore[arg-type]
        _UnusedConn(),
        {},
        [0.1],
        "provider",
        "model",
        page_size=10,
        offset=0,
        threshold=None,
        path_filter=None,
    )

    assert results == []
    assert pagination["has_more"] is False
    assert pagination["next_offset"] is None
