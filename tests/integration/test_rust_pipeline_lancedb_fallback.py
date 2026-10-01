"""Regression test: CHUNKHOUND_USE_RUST=1 must not break non-DuckDB providers.

`chunkhound_native.IndexingPipeline` only implements a DuckDB backend
(`DuckDbHnswBackend`). Before this fix, `IndexingCoordinator.process_directory()`
detected `CHUNKHOUND_USE_RUST=1` purely from the env var and gated cleanup /
change-detection / provider-disconnect behavior on that alone — with no check
that the configured database provider was actually DuckDB. Pointed at a
LanceDB-backed project, this disconnected the live LanceDB connection and then
handed the Rust pipeline a `db_path` it can't use (see
`chunkhound/utils/rust_pipeline_flag.py`'s module docstring, which
documented this as a known gap).

The fix checks the provider's `supports_rust_pipeline` capability up front —
before any Rust-specific behavior — and falls back to the Python path for any
provider that doesn't declare it (LanceDB doesn't; only `DuckDBProvider`
does). This test proves indexing still succeeds, through the DB the
provider actually is, when `CHUNKHOUND_USE_RUST=1` is set against LanceDB.
"""

import asyncio

import pytest


@pytest.fixture
def coordinator(lancedb_provider, tmp_path):
    from chunkhound.core.types.common import Language
    from chunkhound.parsers.parser_factory import create_parser_for_language
    from chunkhound.services.indexing_coordinator import IndexingCoordinator

    parser = create_parser_for_language(Language.PYTHON)
    return IndexingCoordinator(
        lancedb_provider, tmp_path, None, {Language.PYTHON: parser}
    )


def test_rust_flag_falls_back_to_python_for_lancedb(coordinator, tmp_path, monkeypatch):
    """CHUNKHOUND_USE_RUST=1 against a LanceDB project must fall back to the
    Python path — not disconnect the provider and hand it to the DuckDB-only
    Rust pipeline.
    """
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "1")

    (tmp_path / "main.py").write_text("def hello():\n    return 1\n")

    result = asyncio.run(
        coordinator.process_directory(tmp_path, patterns=["**/*.py"])
    )

    assert result["status"] == "success"
    assert result["files_processed"] >= 1
    assert result["total_chunks"] >= 1

    # The provider must still be alive and queryable — a Rust-path attempt
    # would have disconnected it before failing to write through Rust.
    assert coordinator._db.is_connected
    rows = coordinator._db.execute_query("SELECT path FROM files")
    assert rows, "Expected the indexed file to be recorded in LanceDB"
