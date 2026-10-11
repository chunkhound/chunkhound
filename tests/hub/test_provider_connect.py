"""Public connect compatibility with real handles, including partial release."""

import duckdb
import pytest

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.index_format import IndexFormatError
from chunkhound.database_factory import create_database_with_dependencies
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from tests.hub.test_index_format import snapshot


@pytest.mark.parametrize("read_only", [False, True])
def test_healthy_provider_connect_is_idempotent(tmp_path, read_only):
    path = tmp_path / "chunks.db"
    writer = DuckDBProvider(path, base_directory=tmp_path)
    writer.connect()
    writer.disconnect()
    provider = DuckDBProvider(
        path, base_directory=tmp_path, config=DatabaseConfig(read_only=read_only)
    )
    provider.connect()
    try:
        before = snapshot(tmp_path)
        provider.connect()
        provider.connect()
        assert provider.get_stats()["files"] == 0
        assert snapshot(tmp_path) == before
    finally:
        provider.disconnect()


def test_public_factory_database_connect_is_idempotent(tmp_path):
    database = create_database_with_dependencies(
        tmp_path / "chunks.db",
        {"database": {"provider": "duckdb"}, "target_dir": str(tmp_path)},
    )
    try:
        assert database.is_connected
        database.connect()
        database.connect()
        assert database.get_stats()["files"] == 0
    finally:
        database.close()


@pytest.mark.parametrize(
    "remaining", ["manager", "executor", "neither", "closed-manager"]
)
@pytest.mark.parametrize("candidate_version", [None, 2])
def test_partial_release_reconnect_preflights_before_recovery(
    tmp_path, remaining, candidate_version
):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    try:
        # Model real half-releases, not fake flags or fake SQL. Both halves are
        # independent existing handles, and manager.is_connected cannot tell
        # whether the executor's connection is still present or usable.
        if remaining != "executor":
            provider._execute_in_db_thread_sync("disconnect", False)
        if remaining in {"executor", "neither"}:
            provider._connection_manager.disconnect(skip_checkpoint=True)
        elif remaining == "closed-manager":
            provider.connection.close()  # stale non-None manager reference
        if candidate_version is not None:
            with duckdb.connect(str(path) + ".old") as conn:
                conn.execute(
                    'CREATE TABLE "SCHEMA_VERSION" AS SELECT ? AS version',
                    [candidate_version],
                )
            (tmp_path / "chunks.db.swap_intent").write_text("phase2")
            before = snapshot(tmp_path)
            with pytest.raises(IndexFormatError, match="Unsupported index format"):
                provider.connect()
            assert snapshot(tmp_path) == before
        else:
            provider.connect()
            assert provider.connection.execute("SELECT 1").fetchone() == (1,)
            assert provider.get_stats()["files"] == 0
            provider.connect()
    finally:
        # Close without accidentally opening a released executor on refusal.
        provider._executor.shutdown()
        provider._connection_manager.disconnect(skip_checkpoint=True)


def test_lazy_read_after_partial_release_does_not_skip_reconnect_preflight(tmp_path):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    try:
        provider._execute_in_db_thread_sync("disconnect", False)
        assert provider.get_stats()["files"] == 0  # recreates only the executor handle
        with duckdb.connect(str(path) + ".old") as conn:
            conn.execute("CREATE TABLE schema_version AS SELECT 2 AS version")
        before = snapshot(tmp_path)
        with pytest.raises(IndexFormatError, match="Unsupported index format"):
            provider.connect()
        assert snapshot(tmp_path) == before
    finally:
        provider._executor.shutdown()
        provider._connection_manager.disconnect(skip_checkpoint=True)


def test_failed_initialization_with_two_live_handles_does_not_skip_preflight(tmp_path):
    path = tmp_path / "chunks.db"
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE files(id INTEGER)")
    provider = DuckDBProvider(path, base_directory=tmp_path)
    try:
        with pytest.raises(duckdb.Error):
            provider.connect()
        assert provider.is_connected
        provider.connection.execute("UPDATE schema_version SET version=2")
        before = snapshot(tmp_path)
        with pytest.raises(IndexFormatError, match="Unsupported index format"):
            provider.connect()
        assert snapshot(tmp_path) == before
    finally:
        provider._executor.shutdown()
        provider._connection_manager.disconnect(skip_checkpoint=True)


@pytest.mark.parametrize("remaining", ["manager", "executor"])
@pytest.mark.parametrize("metadata", ["2", "'1'", "missing-column", "view"])
def test_partial_reconnect_checks_live_metadata_before_closing_handles(
    tmp_path, remaining, metadata
):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    provider.connection.execute("DROP TABLE schema_version")
    if metadata == "view":
        ddl = 'CREATE VIEW "ScHeMa_VeRsIoN" AS SELECT 1 AS version'
    elif metadata == "missing-column":
        ddl = 'CREATE TABLE "ScHeMa_VeRsIoN" AS SELECT 1 AS wrong_column'
    else:
        ddl = f'CREATE TABLE "ScHeMa_VeRsIoN" AS SELECT {metadata} AS version'
    provider.connection.execute(ddl)
    if remaining == "manager":
        provider._execute_in_db_thread_sync("disconnect", True)
    else:
        provider._connection_manager.disconnect(skip_checkpoint=True)
    before = snapshot(tmp_path)
    try:
        with pytest.raises(IndexFormatError):
            provider.connect()
        assert snapshot(tmp_path) == before, "refusal checkpointed a live handle"
    finally:
        provider._executor.shutdown()
        provider._connection_manager.disconnect(skip_checkpoint=True)


def test_reconnect_after_release_checks_live_format(tmp_path):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    provider.release_for_rust_pipeline()
    with duckdb.connect(str(path)) as conn:
        conn.execute('ALTER TABLE schema_version RENAME TO "SCHEMA_VERSION"')
        conn.execute("UPDATE schema_version SET version=2")
    before = snapshot(tmp_path)
    try:
        with pytest.raises(IndexFormatError, match="Unsupported index format"):
            provider.connect()
        assert snapshot(tmp_path) == before
    finally:
        provider._executor.shutdown()
