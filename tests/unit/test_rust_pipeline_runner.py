"""Public Rust runner contracts using the supported provider and actual native work.

Reconnect follows safe native/cache cleanup; the guard clears only after trusted
reconnect succeeds. In-flight refusal, runtime worker drainage, and unavailable
state on reconnect/client failure live in tests/hub/test_worker_ownership.py.
"""

from types import SimpleNamespace

import duckdb
import pytest

import chunkhound_native as native
from chunkhound.core.exceptions import DatabaseError, RustPipelineError
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.rust_pipeline_runner import run_rust_indexing_phase


@pytest.fixture
def indexing_run(tmp_path):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    try:
        provider.connect()
        source = tmp_path / "example.py"
        source.write_text("def example():\n    return 42\n")
        yield provider, dict(
            db=provider,
            config=None,
            embedding_provider=None,
            progress=None,
            files_to_process=[(source, None)],
            directory=tmp_path,
            force_reindex=True,
            do_cleanup=True,
            diff_task=None,
            parse_task=None,
        )
    finally:
        provider.close()
        native.acquire_install_lease(str(path), "duckdb").close()


@pytest.mark.asyncio
@pytest.mark.parametrize("initially_released", [False, True])
async def test_successful_native_run_reopens_same_provider(
    indexing_run, initially_released
):
    provider, kwargs = indexing_run
    if initially_released:
        # Release is recoverable; final close permanently shuts the executor.
        provider.release_for_rust_pipeline()
        assert not provider.is_connected
    result = await run_rust_indexing_phase(**kwargs)
    assert result.errors == []
    assert result.total_files == provider.get_stats()["files"] == 1
    assert result.total_chunks == provider.get_stats()["chunks"] > 0
    assert provider.execute_query("SELECT path FROM files") == [{"path": "example.py"}]
    assert provider.is_connected and not provider.is_rust_pipeline_in_progress()
    assert provider.execute_query("SELECT 42 AS value") == [{"value": 42}]


@pytest.mark.asyncio
async def test_failed_release_never_starts_native_and_recovers_same_provider(
    indexing_run, monkeypatch
):
    provider, kwargs = indexing_run
    manager = provider.connection
    real_close = duckdb.DuckDBPyConnection.close
    observed_resources = []
    before = provider.get_stats()

    def fail_manager_once(connection):
        if connection is manager and not observed_resources:
            # Observe the admitted owner at the real external close boundary.
            # Its final native_started state proves no native admission occurred.
            observed_resources.append(provider._rust_pipeline_cleanup)
            raise OSError("external manager close failed")
        return real_close(connection)

    monkeypatch.setattr(duckdb.DuckDBPyConnection, "close", fail_manager_once)
    with pytest.raises(DatabaseError, match="external manager close failed") as caught:
        await run_rust_indexing_phase(**kwargs)
    assert caught.value.operation == "release_for_rust_pipeline"
    assert len(observed_resources) == 1
    assert not observed_resources[0].native_started
    assert provider.get_stats() == before
    assert provider.is_connected and not provider.is_rust_pipeline_in_progress()
    assert provider.execute_query("SELECT 42 AS value") == [{"value": 42}]


@pytest.mark.asyncio
async def test_native_startup_validation_error_preserved_after_safe_reconnect(
    indexing_run,
):
    provider, kwargs = indexing_run
    # A negative unsigned field reaches real IndexingPipeline construction.
    # This is startup validation, NOT an already-running native worker failure.
    kwargs["config"] = SimpleNamespace(indexing=SimpleNamespace(db_batch_size=-1))
    before = provider.get_stats()
    with pytest.raises(RustPipelineError) as caught:
        await run_rust_indexing_phase(**kwargs)
    assert isinstance(caught.value.__cause__, OverflowError)
    assert "negative" in str(caught.value.__cause__)
    assert provider.get_stats() == before
    assert provider.is_connected and not provider.is_rust_pipeline_in_progress()
    assert provider.execute_query("SELECT 42 AS value") == [{"value": 42}]
