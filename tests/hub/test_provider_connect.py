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


@pytest.mark.parametrize("remaining", ["manager", "executor"])
@pytest.mark.parametrize("mask", ["temporary", "alternate"])
@pytest.mark.parametrize("version", [1, 2])
def test_partial_reconnect_checks_owned_catalog_despite_session_namespace(
    tmp_path, remaining, mask, version
):
    # Quotes in the catalog name must not become SQL syntax.
    path = tmp_path / 'owned"quoted.db'
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    query = (
        provider.connection.execute
        if remaining == "manager"
        else provider.execute_query
    )
    try:
        query(f"UPDATE schema_version SET version={version}")
        if mask == "temporary":
            query("CREATE TEMP TABLE schema_version AS SELECT 1 AS version")
        else:
            query("ATTACH ':memory:' AS alternate")
            query("USE alternate")
        if remaining == "manager":
            provider._execute_in_db_thread_sync("disconnect", True)
        else:
            provider._connection_manager.disconnect(skip_checkpoint=True)
        before = snapshot(tmp_path)
        assert str(path) + ".wal" in before
        if version == 2:
            with pytest.raises(IndexFormatError, match="Unsupported index format"):
                provider.connect()
            assert snapshot(tmp_path) == before
            # Refusal keeps the live handle and its namespace intact.
            sql = (
                "SELECT version AS value FROM schema_version"
                if mask == "temporary"
                else "SELECT current_database() AS value"
            )
            rows = query(sql)
            value = rows.fetchone()[0] if remaining == "manager" else rows[0]["value"]
            assert value == (1 if mask == "temporary" else "alternate")
        else:
            provider.connect()
            assert provider.get_stats()["files"] == 0
    finally:
        provider._executor.shutdown()
        provider._connection_manager.disconnect(skip_checkpoint=True)


@pytest.mark.parametrize("remaining", ["manager", "executor"])
@pytest.mark.parametrize("scope", ["TEMP", ""])
@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("helper", ["translate", "duckdb_tables"])
def test_partial_reconnect_ignores_catalog_helper_macros(
    tmp_path, remaining, scope, version, helper
):
    path = tmp_path / "owned.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    query = (
        provider.connection.execute
        if remaining == "manager"
        else provider.execute_query
    )
    try:
        query(f"UPDATE schema_version SET version={version}")
        if helper == "translate":
            query(f"CREATE {scope} MACRO translate(a,b,c) AS 'hidden'")
            masked_sql = "translate('A','A','a')"
            masked_value = "hidden"
        else:
            query(
                f"CREATE {scope} MACRO duckdb_tables() AS TABLE "
                "SELECT * FROM system.main.duckdb_tables() "
                "WHERE table_name <> 'schema_version'"
            )
            masked_sql = (
                "(SELECT count(*) FROM duckdb_tables() "
                "WHERE table_name='schema_version' AND database_name='owned')"
            )
            masked_value = 0
        # A TEMP sentinel proves that refusal retains the original session,
        # including in the persistent-macro case where the macro survives reopen.
        query("CREATE TEMP TABLE retained_session AS SELECT 42 AS value")
        if remaining == "manager":
            provider._execute_in_db_thread_sync("disconnect", True)
        else:
            provider._connection_manager.disconnect(skip_checkpoint=True)
        backup = str(path) + ".compact_backup"
        with duckdb.connect(backup) as conn:
            conn.execute("CREATE TABLE schema_version AS SELECT 1 AS version")
        before = snapshot(tmp_path)
        assert str(path) + ".wal" in before
        assert backup in before
        if version == 2:
            with pytest.raises(IndexFormatError, match="Unsupported index format"):
                provider.connect()
            assert snapshot(tmp_path) == before, "refusal changed DB/WAL/backup"
            rows = query(
                f"SELECT value, {masked_sql} AS masked, "
                "(SELECT version FROM owned.main.schema_version) AS version "
                "FROM retained_session"
            )
            retained = (
                rows.fetchone() if remaining == "manager"
                else tuple(rows[0][key] for key in ("value", "masked", "version"))
            )
            assert retained == (42, masked_value, 2)
            assert snapshot(tmp_path) == before
        else:
            provider.connect()
            assert provider.get_stats()["files"] == 0
            assert provider.execute_query(
                "SELECT version FROM owned.main.schema_version"
            ) == [{"version": 1}]
    finally:
        provider._executor.shutdown()
        provider._connection_manager.disconnect(skip_checkpoint=True)


@pytest.mark.parametrize("remaining", ["manager", "executor"])
@pytest.mark.parametrize("scope", ["TEMP", ""])
@pytest.mark.parametrize("mask", ["function", "relation"])
def test_partial_reconnect_refuses_metadata_view_despite_catalog_mask(
    tmp_path, remaining, scope, mask
):
    path = tmp_path / "owned.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    query = (
        provider.connection.execute
        if remaining == "manager"
        else provider.execute_query
    )
    try:
        query("DROP TABLE schema_version")
        query("CREATE VIEW schema_version AS SELECT 1 AS version")
        definition = (
            "MACRO duckdb_views() AS TABLE" if mask == "function"
            else "VIEW duckdb_views AS"
        )
        query(
            f"CREATE {scope} {definition} "
            "SELECT * FROM system.main.duckdb_views() "
            "WHERE view_name <> 'schema_version'"
        )
        query("CREATE TEMP TABLE retained_session AS SELECT 42 AS value")
        if remaining == "manager":
            provider._execute_in_db_thread_sync("disconnect", True)
        else:
            provider._connection_manager.disconnect(skip_checkpoint=True)
        backup = str(path) + ".compact_backup"
        with duckdb.connect(backup) as conn:
            conn.execute("CREATE TABLE schema_version AS SELECT 1 AS version")
        before = snapshot(tmp_path)
        assert str(path) + ".wal" in before
        assert backup in before
        with pytest.raises(IndexFormatError, match="physical table"):
            provider.connect()
        assert snapshot(tmp_path) == before, "refusal changed DB/WAL/backup"
        rows = query("SELECT value FROM retained_session")
        value = rows.fetchone()[0] if remaining == "manager" else rows[0]["value"]
        assert value == 42
        assert snapshot(tmp_path) == before
    finally:
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
@pytest.mark.parametrize(
    "metadata", ["2", "'1'", "missing-column", "view", "generated"]
)
def test_partial_reconnect_checks_live_metadata_before_closing_handles(
    tmp_path, remaining, metadata
):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    provider.connection.execute("DROP TABLE schema_version")
    if metadata == "generated":
        ddl = (
            'CREATE TABLE "ScHeMa_VeRsIoN"(seed INTEGER, version INTEGER '
            "GENERATED ALWAYS AS (1), description VARCHAR)"
        )
    elif metadata == "view":
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


@pytest.mark.parametrize("owner", ["executor-api", "executor-sql", "manager"])
@pytest.mark.parametrize("metadata", ["stale-snapshot", "uncommitted-mask"])
def test_partial_reconnect_refuses_transaction_local_format_without_mutation(
    tmp_path, owner, metadata
):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    manager = provider.connection
    query = manager.execute if owner == "manager" else provider.execute_query
    other_query = provider.execute_query if owner == "manager" else manager.execute
    try:
        if metadata == "uncommitted-mask":
            other_query("UPDATE schema_version SET version=2")
        if owner == "executor-api":
            provider.begin_transaction()
        else:
            query("BEGIN TRANSACTION")
        if metadata == "stale-snapshot":
            query("SELECT version FROM schema_version")
            other_query("UPDATE schema_version SET version=2")
        else:
            query("UPDATE schema_version SET version=1")
        if owner == "manager":
            provider._execute_in_db_thread_sync("disconnect", True)
        else:
            provider._connection_manager.disconnect(skip_checkpoint=True)

        before = snapshot(tmp_path)
        with pytest.raises(IndexFormatError, match="transaction"):
            provider.connect()
        assert snapshot(tmp_path) == before, "refusal changed DB/WAL hashes or mtimes"
        # Refusal must leave the original transaction, including its snapshot,
        # usable until the caller explicitly ends it.
        rows = query("SELECT version FROM schema_version")
        assert (rows.fetchone()[0] if owner == "manager" else rows[0]["version"]) == 1
        if owner == "executor-api":
            provider.rollback_transaction()
        else:
            query("ROLLBACK")
        before = snapshot(tmp_path)
        with pytest.raises(IndexFormatError, match="Unsupported index format"):
            provider.connect()
        assert snapshot(tmp_path) == before
    finally:
        provider._executor.shutdown()
        provider._connection_manager.disconnect(skip_checkpoint=True)


@pytest.mark.parametrize("owner", ["executor-api", "executor-sql", "manager"])
@pytest.mark.parametrize("finish", ["commit", "rollback"])
def test_partial_reconnect_preserves_pending_work_for_explicit_completion(
    tmp_path, owner, finish
):
    provider = DuckDBProvider(tmp_path / "chunks.db", base_directory=tmp_path)
    provider.connect()
    manager = provider.connection
    query = manager.execute if owner == "manager" else provider.execute_query
    try:
        if owner == "executor-api":
            provider.begin_transaction()
        else:
            query("BEGIN")
        query("CREATE TABLE pending_work AS SELECT 42 AS value")
        if owner == "manager":
            provider._execute_in_db_thread_sync("disconnect", True)
        else:
            provider._connection_manager.disconnect(skip_checkpoint=True)
        before = snapshot(tmp_path)
        with pytest.raises(IndexFormatError, match="transaction"):
            provider.connect()
        assert snapshot(tmp_path) == before
        rows = query("SELECT value FROM pending_work")
        assert (rows.fetchone()[0] if owner == "manager" else rows[0]["value"]) == 42
        if owner == "executor-api":
            getattr(provider, f"{finish}_transaction")()
        else:
            query(finish.upper())
        provider.connect()
        if finish == "commit":
            assert provider.execute_query("SELECT value FROM pending_work") == [
                {"value": 42}
            ]
        else:
            assert provider.execute_query(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name='pending_work'"
            ) == []
        provider.begin_transaction()
        provider.commit_transaction()
    finally:
        provider.disconnect()


@pytest.mark.parametrize("owner", ["executor-api", "executor-sql", "manager"])
def test_partial_reconnect_keeps_aborted_transaction_for_explicit_rollback(
    tmp_path, owner
):
    provider = DuckDBProvider(tmp_path / "chunks.db", base_directory=tmp_path)
    provider.connect()
    query = (
        provider.connection.execute if owner == "manager" else provider.execute_query
    )
    try:
        if owner == "executor-api":
            provider.begin_transaction()
        else:
            query("BEGIN")
        query("CREATE TABLE pending_work AS SELECT 42 AS value")
        with pytest.raises(duckdb.ConversionException):
            query("SELECT CAST('bad' AS INTEGER)")
        if owner == "manager":
            provider._execute_in_db_thread_sync("disconnect", True)
        else:
            provider._connection_manager.disconnect(skip_checkpoint=True)
        before = snapshot(tmp_path)
        with pytest.raises(IndexFormatError, match="transaction"):
            provider.connect()
        assert snapshot(tmp_path) == before
        query("ROLLBACK")
        provider.connect()
        assert provider.execute_query(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name='pending_work'"
        ) == []
    finally:
        provider._executor.shutdown()
        provider._connection_manager.disconnect(skip_checkpoint=True)


@pytest.mark.parametrize("owner", ["executor", "manager"])
def test_lazy_partial_reconnect_checks_transactions_on_both_live_handles(
    tmp_path, owner
):
    provider = DuckDBProvider(tmp_path / "chunks.db", base_directory=tmp_path)
    provider.connect()
    try:
        provider._execute_in_db_thread_sync("disconnect", True)
        assert provider.get_stats()["files"] == 0  # lazy recreation is not full init
        query = (
            provider.connection.execute
            if owner == "manager"
            else provider.execute_query
        )
        query("BEGIN")
        query("CREATE TABLE pending_work AS SELECT 42 AS value")
        before = snapshot(tmp_path)
        with pytest.raises(IndexFormatError, match="transaction"):
            provider.connect()
        assert snapshot(tmp_path) == before
        query("COMMIT")
        provider.connect()
        assert provider.execute_query("SELECT value FROM pending_work") == [
            {"value": 42}
        ]
    finally:
        provider._executor.shutdown()
        provider._connection_manager.disconnect(skip_checkpoint=True)


def test_partial_reconnect_resets_tracking_after_explicit_sql_commit(tmp_path):
    provider = DuckDBProvider(tmp_path / "chunks.db", base_directory=tmp_path)
    provider.connect()
    try:
        provider.begin_transaction()
        provider.execute_query("COMMIT")  # direct SQL does not update API tracking
        provider._connection_manager.disconnect(skip_checkpoint=True)
        provider.connect()
        # A stale transaction_active flag must not block public maintenance on
        # the replacement connection after all actual transactions have ended.
        provider.compact_database()
        assert provider.get_stats()["files"] == 0
    finally:
        provider.disconnect()


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
