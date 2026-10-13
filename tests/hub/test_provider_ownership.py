"""Real provider handles exclude installation until deterministic final close."""

from pathlib import Path

import pytest

import chunkhound_native as native
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.database import Database
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.providers.database.lancedb_provider import LanceDBProvider
from tests.hub.test_index_format import run_python, snapshot


@pytest.mark.parametrize(
    "backend,lazy,read_only",
    [
        ("duckdb", False, False),
        ("duckdb", True, False),
        ("duckdb", False, True),
        ("lancedb", False, False),
        ("lancedb", True, False),
    ],
)
def test_live_provider_excludes_install_until_final_close(
    tmp_path, backend, lazy, read_only
):
    cls = DuckDBProvider if backend == "duckdb" else LanceDBProvider
    path = tmp_path / ("chunks.db" if backend == "duckdb" else "lancedb.lancedb")
    seed = cls(path, base_directory=tmp_path)
    seed.connect()
    seed.close()
    provider = cls(
        path, base_directory=tmp_path, config=DatabaseConfig(read_only=read_only)
    )
    try:
        if lazy:
            provider.get_stats()
        else:
            provider.connect()
        run_python(
            """
            import sys
            import chunkhound_native as n
            try:
                n.acquire_install_lease(sys.argv[1], sys.argv[2])
            except n.DatabaseBusyError:
                pass
            else:
                raise AssertionError('live provider was not coordinated')
        """,
            path,
            backend,
        )
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), backend)
    finally:
        provider.close()
        provider.close()
    lease = native.acquire_install_lease(str(path), backend)
    lease.close()


def test_compatible_providers_preserve_each_others_transactions(tmp_path):
    path = tmp_path / "chunks.db"
    first = DuckDBProvider(path, base_directory=tmp_path)
    second = DuckDBProvider(path, base_directory=tmp_path)
    first.connect()
    try:
        first.execute_query("CREATE TABLE retained(value INTEGER)")
        first.execute_query("BEGIN")
        first.execute_query("INSERT INTO retained VALUES (42)")
        second.connect()
        assert second.execute_query("SELECT * FROM retained") == []
        assert first.execute_query("SELECT * FROM retained") == [{"value": 42}]
        first.execute_query("COMMIT")
        assert second.execute_query("SELECT * FROM retained") == [{"value": 42}]
        first.close()
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
        second.connect()
        assert second.get_stats()["files"] == 0
    finally:
        second.close()
        first.close()


def test_high_level_close_drains_executor_when_manager_is_absent(tmp_path):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    provider._connection_manager.disconnect(skip_checkpoint=True)
    assert not provider.is_connected
    database = Database(
        path,
        provider=provider,
        indexing_coordinator=object(),
        search_service=object(),
        embedding_service=object(),
    )
    database.close()
    database.close()
    lease = native.acquire_install_lease(str(path), "duckdb")
    lease.close()


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_interruption_refusal_never_runs_recovery_or_claims_root(tmp_path, backend):
    path = tmp_path / ("chunks.db" if backend == "duckdb" else "lancedb.lancedb")
    marker = Path(str(path) + ".chunkhound-install.json")
    marker.write_bytes(b"{")
    backup = Path(str(path) + ".compact_backup")
    backup.write_bytes(b"old files must not be recovered")
    before = snapshot(tmp_path)
    cls = DuckDBProvider if backend == "duckdb" else LanceDBProvider
    provider = cls(path, base_directory=tmp_path)
    try:
        with pytest.raises(native.DatabaseInterruptedError):
            provider.connect()
    finally:
        provider.close()  # Must not lazy-open while cleaning a refused open.
    assert snapshot(tmp_path) == before
    assert not path.exists()
    assert not Path(str(path) + ".root.json").exists()
