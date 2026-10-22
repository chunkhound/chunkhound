"""Real provider handles exclude installation until deterministic final close."""

import asyncio
import concurrent.futures
import os
import threading
import traceback
import weakref
from contextlib import contextmanager
from pathlib import Path

import duckdb
import lancedb
import pytest

import chunkhound_native as native
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.exceptions import DatabaseError
from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.database import Database
from chunkhound.providers.database.duckdb.connection_manager import (
    DuckDBConnectionManager,
)
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.providers.database.lancedb_provider import LanceDBProvider
from chunkhound.providers.database.ownership import DatabaseReaderCloseError
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


@contextmanager
def _failed_inspection_shutdown(monkeypatch):
    """Fault only DuckDB reader shutdown; keep real ATTACH/query behavior."""
    readers = []
    real_close = duckdb.DuckDBPyConnection.close

    def fail_exit(conn, *args):
        readers.append(weakref.ref(conn))
        raise OSError("inspection exit failed")

    def fail_close(conn):
        if any(reader() is conn for reader in readers):
            raise OSError("inspection close failed")
        return real_close(conn)

    with monkeypatch.context() as fault:
        fault.setattr(duckdb.DuckDBPyConnection, "__exit__", fail_exit)
        fault.setattr(duckdb.DuckDBPyConnection, "close", fail_close)
        yield readers


def test_native_handoff_drains_failed_reconnect_reader_without_releasing_lease(
    tmp_path, monkeypatch
):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    owner = provider._database_ownership
    try:
        provider.set_rust_pipeline_in_progress(True)
        provider.release_for_rust_pipeline()
        provider.set_rust_pipeline_in_progress(False)
        with _failed_inspection_shutdown(monkeypatch) as readers:
            with pytest.raises(DatabaseReaderCloseError):
                provider.connect()
            assert readers[0]().execute(
                "SELECT count(*) FROM inspected.main.files"
            ).fetchone() == (0,)
            provider.set_rust_pipeline_in_progress(True)
            with pytest.raises(DatabaseError, match="inspection reader"):
                provider.release_for_rust_pipeline()
            assert readers[0]().execute("SELECT 42").fetchone() == (42,)
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(path), "duckdb")
        provider.release_for_rust_pipeline()
        if readers[0]() is not None:
            with pytest.raises(duckdb.ConnectionException):
                readers[0]().execute("SELECT 42")
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
        provider.set_rust_pipeline_in_progress(False)
        provider.connect()
        assert provider.get_stats()["files"] == 0
        provider.close()
        native.acquire_install_lease(str(path), "duckdb").close()
    finally:
        provider.set_rust_pipeline_in_progress(False)
        provider.close()
        owner.close()


def test_public_close_retries_failed_reader_after_executor_finalization(
    tmp_path, monkeypatch
):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    owner = provider._database_ownership
    try:
        provider.set_rust_pipeline_in_progress(True)
        provider.release_for_rust_pipeline()
        provider.set_rust_pipeline_in_progress(False)
        with _failed_inspection_shutdown(monkeypatch) as readers:
            with pytest.raises(DatabaseReaderCloseError):
                provider.connect()
            with pytest.raises(DatabaseReaderCloseError):
                provider.close()
            assert readers[0]().execute(
                "SELECT count(*) FROM inspected.main.files"
            ).fetchone() == (0,)
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(path), "duckdb")
        provider.close()
        provider.close()
        if readers[0]() is not None:
            with pytest.raises(duckdb.ConnectionException):
                readers[0]().execute("SELECT 42")
        native.acquire_install_lease(str(path), "duckdb").close()
    finally:
        provider.close()
        owner.close()


def test_borrowed_reader_recovery_preserves_live_provider_and_transaction(
    tmp_path, monkeypatch
):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    owner = provider._database_ownership
    manager = provider.connection
    try:
        # A valid closed recovery candidate avoids reading the live primary,
        # whose writable DuckDB handle denies file reads on Windows.
        with duckdb.connect(str(path) + ".compact_backup") as backup:
            backup.execute("CREATE TABLE schema_version AS SELECT 1 AS version")
            backup.execute("CREATE TABLE files(id INTEGER)")
        # Supported partial reconnect: the manager survives executor disconnect.
        provider._execute_in_db_thread_sync("disconnect", True)
        with _failed_inspection_shutdown(monkeypatch) as readers:
            with pytest.raises(DatabaseReaderCloseError) as raised:
                provider.connect()
            assert raised.value.ownership is owner
            assert readers[0]().execute(
                "SELECT count(*) FROM inspected.main.files"
            ).fetchone() == (0,)
        manager.execute("BEGIN")
        manager.execute("CREATE TABLE pending_recovery AS SELECT 42 AS value")
        # Follow the error's public recovery instruction, not raw lease access.
        raised.value.retry_close()
        assert manager.execute("SELECT count(*) FROM files").fetchone() == (0,)
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
        assert manager.execute("SELECT value FROM pending_recovery").fetchone() == (42,)
        manager.execute("COMMIT")
        if readers[0]() is not None:
            with pytest.raises(duckdb.ConnectionException):
                readers[0]().execute("SELECT 42")
        provider.close()
        with pytest.raises(duckdb.ConnectionException):
            manager.execute("SELECT 42")
        native.acquire_install_lease(str(path), "duckdb").close()
    finally:
        provider.close()
        owner.close()


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


def start_call(function, name):
    """Run a public operation without adding another executor lifecycle."""
    result = concurrent.futures.Future()

    def run():
        try:
            result.set_result(function())
        except BaseException as error:
            result.set_exception(error)

    thread = threading.Thread(target=run, name=name)
    thread.start()
    return thread, result


@pytest.mark.parametrize("active_transaction", [False, True])
def test_compaction_refuses_other_provider_and_preserves_durable_writes(
    tmp_path, active_transaction
):
    path = tmp_path / "chunks.db"
    first = DuckDBProvider(path, base_directory=tmp_path)
    second = DuckDBProvider(path, base_directory=tmp_path)
    first.connect()
    second.connect()
    try:
        second.execute_query("CREATE TABLE retained(value INTEGER)")
        if active_transaction:
            second.execute_query("BEGIN")
            second.execute_query("INSERT INTO retained VALUES (42)")
        with pytest.raises(RuntimeError, match="another.*(provider|connection)"):
            first.compact_database()
        if active_transaction:
            assert second.execute_query("SELECT * FROM retained") == [{"value": 42}]
            second.execute_query("COMMIT")
        first.execute_query(
            "UPDATE schema_version SET description='must-survive-checkpoint'"
        )
        first.execute_query("CHECKPOINT")
    finally:
        second.disconnect(skip_checkpoint=True)
        first.close()
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute("SELECT description FROM schema_version").fetchone() == (
            "must-survive-checkpoint",
        )
        if active_transaction:
            assert fresh.execute("SELECT * FROM retained").fetchall() == [(42,)]


@pytest.mark.parametrize("lazy", [False, True])
def test_compaction_waits_for_inflight_open_then_refuses_other_provider(
    tmp_path, monkeypatch, lazy
):
    path = tmp_path / "chunks.db"
    first = DuckDBProvider(path, base_directory=tmp_path)
    second = DuckDBProvider(path, base_directory=tmp_path)
    first.connect()
    opened = threading.Event()
    release = threading.Event()
    real_connect = duckdb.connect

    def paused_connect(database, *args, **kwargs):
        conn = real_connect(database, *args, **kwargs)
        if str(database) == str(path) and not opened.is_set():
            opened.set()  # Real handle exists but open has not returned to our code.
            assert release.wait(5)
        return conn

    monkeypatch.setattr(duckdb, "connect", paused_connect)
    opening, opened_result = start_call(
        (lambda: second.execute_query("SELECT 1")) if lazy else second.connect,
        "opening-provider",
    )
    compacting = None
    try:
        assert opened.wait(5)
        compacting, compacted_result = start_call(first.compact_database, "compactor")
        with pytest.raises(concurrent.futures.TimeoutError):
            compacted_result.result(timeout=0.1)
        release.set()
        opened_result.result(timeout=5)
        with pytest.raises(RuntimeError, match="another.*(provider|connection)"):
            compacted_result.result(timeout=5)
        first.execute_query("UPDATE schema_version SET description='admission-safe'")
        first.execute_query("CHECKPOINT")
    finally:
        release.set()
        opening.join(5)
        if compacting is not None:
            compacting.join(5)
        second.close()
        first.close()
    with real_connect(str(path), read_only=True) as fresh:
        assert fresh.execute("SELECT description FROM schema_version").fetchone() == (
            "admission-safe",
        )


def test_new_provider_cannot_open_during_compaction_swap(tmp_path, monkeypatch):
    path = tmp_path / "chunks.db"
    first = DuckDBProvider(path, base_directory=tmp_path)
    second = DuckDBProvider(path, base_directory=tmp_path)
    first.connect()
    swapping = threading.Event()
    release = threading.Event()
    real_replace = os.replace

    def paused_replace(src, dst, *args, **kwargs):
        if str(src) == str(path) and str(dst).endswith(".compact_backup"):
            swapping.set()
            assert release.wait(5)
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", paused_replace)
    compacting, compacted_result = start_call(first.compact_database, "compactor")
    opening = None
    try:
        assert swapping.wait(5)
        opening, opened_result = start_call(second.connect, "opening-provider")
        with pytest.raises(concurrent.futures.TimeoutError):
            opened_result.result(timeout=0.1)
        release.set()
        compacted_result.result(timeout=5)
        opened_result.result(timeout=5)
        first.execute_query("UPDATE schema_version SET description='swap-safe'")
        first.execute_query("CHECKPOINT")
    finally:
        release.set()
        compacting.join(5)
        if opening is not None:
            opening.join(5)
        second.close()
        first.close()
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute("SELECT description FROM schema_version").fetchone() == (
            "swap-safe",
        )


@pytest.mark.parametrize("phase", ["duckdb-prepare", "duckdb-initialize", "lancedb"])
@pytest.mark.parametrize("close_while_running", [False, True])
def test_connect_deadline_retains_ownership_until_explicit_close(
    tmp_path, monkeypatch, phase, close_while_running
):
    backend = "lancedb" if phase == "lancedb" else "duckdb"
    path = tmp_path / ("lancedb.lancedb" if backend == "lancedb" else "chunks.db")
    cls = LanceDBProvider if backend == "lancedb" else DuckDBProvider
    if backend == "duckdb":
        seed = cls(path, base_directory=tmp_path)
        seed.connect()
        seed.close()
    provider = cls(
        path,
        base_directory=tmp_path,
        config=DatabaseConfig(execute_timeout_seconds=0.5),
    )
    boundary = lancedb if backend == "lancedb" else duckdb
    real_connect = boundary.connect
    opening = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    target_open = 2 if phase == "duckdb-initialize" else 1
    opens = 0

    def paused_connect(database, *args, **kwargs):
        nonlocal opens
        if str(database) in (str(path), path.name) and not kwargs.get("read_only"):
            opens += 1
            if opens == target_open:
                opening.set()
                assert release.wait(5)
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(boundary, "connect", paused_connect)
    caller, result = start_call(provider.connect, "connect-caller")
    result.add_done_callback(lambda _: returned.set())
    closer = None
    try:
        assert opening.wait(5)
        assert returned.wait(2), "connect deadline was replaced by an unbounded close"
        with pytest.raises(TimeoutError):
            result.result()
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), backend)
        if close_while_running:
            closer, closed = start_call(provider.close, "explicit-close")
            with pytest.raises(concurrent.futures.TimeoutError):
                closed.result(timeout=0.1)
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(path), backend)
            release.set()
            closed.result(timeout=5)
        else:
            release.set()
            # A subsequent public connect is queued after the original worker.
            # Even after initialization finishes, no implicit lease release occurs.
            provider.connect()
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(path), backend)
            provider.close()
    finally:
        release.set()
        caller.join(5)
        if closer is not None:
            closer.join(5)
        provider.close()
    native.acquire_install_lease(str(path), backend).close()


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_completed_backend_timeout_is_initialization_failure_not_caller_deadline(
    tmp_path, monkeypatch, backend
):
    path = tmp_path / ("chunks.db" if backend == "duckdb" else "lancedb.lancedb")
    cls = DuckDBProvider if backend == "duckdb" else LanceDBProvider
    provider = cls(path, base_directory=tmp_path)
    boundary = duckdb if backend == "duckdb" else lancedb

    def failed_connect(*args, **kwargs):
        raise TimeoutError("backend initialization failed")

    monkeypatch.setattr(boundary, "connect", failed_connect)
    try:
        with pytest.raises(TimeoutError, match="backend initialization failed"):
            provider.connect()
        native.acquire_install_lease(str(path), backend).close()
    finally:
        provider.close()


@pytest.mark.parametrize("boundary", ["joined-worker", "manager-close"])
def test_concurrent_public_close_coalesces_cleanup(tmp_path, monkeypatch, boundary):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    provider.execute_query("UPDATE schema_version SET description='closed-durably'")
    manager = provider.connection
    entered = threading.Event()
    release = threading.Event()
    real_shutdown = concurrent.futures.ThreadPoolExecutor.shutdown
    real_close = duckdb.DuckDBPyConnection.close

    def paused_shutdown(pool, *args, **kwargs):
        real_shutdown(pool, *args, **kwargs)
        if threading.current_thread().name == "first-close":
            entered.set()
            assert release.wait(5)

    def paused_close(conn):
        if conn is manager:
            if entered.is_set():
                raise RuntimeError("concurrent backend handle close")
            entered.set()
            assert release.wait(5)
        return real_close(conn)

    if boundary == "joined-worker":
        monkeypatch.setattr(
            concurrent.futures.ThreadPoolExecutor, "shutdown", paused_shutdown
        )
    else:
        monkeypatch.setattr(duckdb.DuckDBPyConnection, "close", paused_close)
    first, first_result = start_call(provider.close, "first-close")
    second = None
    try:
        assert entered.wait(5)
        second, second_result = start_call(provider.close, "second-close")
        with pytest.raises(concurrent.futures.TimeoutError):
            second_result.result(timeout=0.1)
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
        release.set()
        first_result.result(timeout=5)
        second_result.result(timeout=5)
    finally:
        release.set()
        first.join(5)
        if second is not None:
            second.join(5)
        monkeypatch.undo()
        provider.close()
    native.acquire_install_lease(str(path), "duckdb").close()
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute("SELECT description FROM schema_version").fetchone() == (
            "closed-durably",
        )


@pytest.mark.parametrize("handle", ["worker", "manager"])
def test_failed_public_close_retains_lease_and_allows_cleanup_retry(
    tmp_path, monkeypatch, handle
):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    provider.execute_query("UPDATE schema_version SET description='retry-durable'")
    manager = provider.connection
    real_close = duckdb.DuckDBPyConnection.close
    failed = False

    def fail_once(conn):
        nonlocal failed
        target = (conn is manager) if handle == "manager" else (conn is not manager)
        if target and not failed:
            failed = True
            raise OSError("backend close failed with handle retained")
        return real_close(conn)

    monkeypatch.setattr(duckdb.DuckDBPyConnection, "close", fail_once)
    try:
        with pytest.raises(OSError, match="handle retained"):
            provider.close()
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
        with pytest.raises(RuntimeError, match="closing or closed"):
            provider.execute_query("SELECT 1")
        provider.close()
        provider.close()
    finally:
        provider.close()
    native.acquire_install_lease(str(path), "duckdb").close()
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute("SELECT description FROM schema_version").fetchone() == (
            "retry-durable",
        )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_retained_lance_regex_error_cannot_keep_table_alive_after_close(
    tmp_path, asynchronous
):
    path = tmp_path / "lancedb.lancedb"
    provider = LanceDBProvider(path, base_directory=tmp_path)
    provider.connect()
    try:
        file_id = provider.insert_file(
            File(path="source.py", mtime=1.0, language=Language.PYTHON, size_bytes=32)
        )
        provider.insert_chunks_batch(
            [
                Chunk(
                    file_id=file_id,
                    code="def function(): return 1",
                    start_line=1,
                    end_line=1,
                    chunk_type=ChunkType.FUNCTION,
                    language=Language.PYTHON,
                    symbol="function",
                )
            ]
        )
        table = weakref.ref(provider._chunks_table)
        with pytest.raises(RuntimeError, match="Regex search failed") as raised:
            if asynchronous:
                asyncio.run(provider.search_regex_async("["))
            else:
                provider.search_regex("[")
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "lancedb")
        provider.close()
        lease = native.acquire_install_lease(str(path), "lancedb")
        try:
            # Keep the public exception, not a backend handle. No GC is needed
            # to make final-close installation admission safe.
            assert raised.value.__cause__ is not None
            assert traceback.extract_tb(raised.value.__cause__.__traceback__)
            assert "regex" in str(raised.value.__cause__).lower()
            assert table() is None, (
                "retained error still owns a usable table with "
                f"{table().count_rows()} rows"
            )
        finally:
            lease.close()
    finally:
        provider.close()


@pytest.mark.parametrize("operation", ["connect", "get_stats"])
def test_failed_lance_open_error_cannot_keep_untransferred_table_alive(
    tmp_path, monkeypatch, operation
):
    path = tmp_path / "lancedb.lancedb"
    seed = LanceDBProvider(path, base_directory=tmp_path)
    seed.connect()
    seed.close()
    provider = LanceDBProvider(path, base_directory=tmp_path)
    real_connect = lancedb.connect
    tables = []

    def failed_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        table = connection.open_table("chunks")
        tables.append(weakref.ref(table))
        try:
            raise OSError("backend open failed after table access")
        except OSError as error:
            raise RuntimeError("Lance open failed") from error

    monkeypatch.setattr(lancedb, "connect", failed_connect)
    try:
        with pytest.raises(RuntimeError, match="Lance open failed") as raised:
            getattr(provider, operation)()
        provider.close()
        native.acquire_install_lease(str(path), "lancedb").close()
        assert raised.value.__cause__ is not None
        assert tables and all(table() is None for table in tables)
    finally:
        provider.close()


def assert_closed_or_collected(handles):
    """A retained public exception must not keep a usable backend handle alive."""
    for reference in handles:
        conn = reference()
        if conn is not None:
            with pytest.raises(duckdb.ConnectionException):
                conn.execute("SELECT 42").fetchone()


@pytest.mark.parametrize(
    "opening", ["lazy", "prepare", "manager", "standalone-manager"]
)
def test_cursor_close_failure_retains_install_exclusion_until_cleanup_retry(
    tmp_path, monkeypatch, opening
):
    path = tmp_path / "chunks.db"
    first = DuckDBProvider(path, base_directory=tmp_path)
    first.connect()
    second = (
        DuckDBConnectionManager(path)
        if opening == "standalone-manager"
        else DuckDBProvider(path, base_directory=tmp_path)
    )
    close_second = (
        second.disconnect if opening == "standalone-manager" else second.close
    )
    real_connect = duckdb.connect
    real_cursor = duckdb.DuckDBPyConnection.cursor
    real_close = duckdb.DuckDBPyConnection.close
    opened = []
    cursors = []
    target_cursor = 2 if opening == "manager" else 1

    def track_connect(database, *args, **kwargs):
        conn = real_connect(database, *args, **kwargs)
        if str(database) == str(path):
            opened.append(weakref.ref(conn))
        return conn

    def track_cursor(conn, *args, **kwargs):
        cursor = real_cursor(conn, *args, **kwargs)
        cursors.append(weakref.ref(cursor))
        return cursor

    def fail_cursor_close(conn):
        if len(cursors) >= target_cursor and conn is cursors[target_cursor - 1]():
            raise OSError("compatible cursor close failed")
        return real_close(conn)

    monkeypatch.setattr(duckdb, "connect", track_connect)
    monkeypatch.setattr(duckdb.DuckDBPyConnection, "cursor", track_cursor)
    monkeypatch.setattr(duckdb.DuckDBPyConnection, "close", fail_cursor_close)
    try:
        # Keep the public exception alive throughout final close and installation
        # admission. Backend tracking uses weak refs, never a test-owned lease.
        with pytest.raises(OSError, match="compatible cursor close failed") as raised:
            if opening == "lazy":
                second.get_stats()
            else:
                second.connect()
        assert cursors[target_cursor - 1]().execute("SELECT 42").fetchone() == (42,)
        first.close()
        with pytest.raises(OSError, match="compatible cursor close failed"):
            close_second()
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
        monkeypatch.setattr(duckdb.DuckDBPyConnection, "close", real_close)
        close_second()
        close_second()
        native.acquire_install_lease(str(path), "duckdb").close()
        assert raised.value is not None
        assert_closed_or_collected(opened + cursors)
    finally:
        monkeypatch.undo()
        close_second()
        first.close()
        # Avoid leaving a known-broken pre-fix handle alive in the RED run.
        for reference in opened + cursors:
            if reference() is not None:
                real_close(reference())


@pytest.mark.parametrize("retained", ["failed-open", "cursor-and-open"])
def test_rust_release_closes_all_retained_handles_or_preserves_lease_for_retry(
    tmp_path, monkeypatch, retained
):
    path = tmp_path / "chunks.db"
    seed = DuckDBProvider(path, base_directory=tmp_path)
    seed.connect()
    if retained == "failed-open":
        seed.close()
    provider = DuckDBProvider(path, base_directory=tmp_path)
    real_connect = duckdb.connect
    real_execute = duckdb.DuckDBPyConnection.execute
    real_cursor = duckdb.DuckDBPyConnection.cursor
    real_close = duckdb.DuckDBPyConnection.close
    handles = []

    def track_connect(database, *args, **kwargs):
        conn = real_connect(database, *args, **kwargs)
        if str(database) == str(path) and not kwargs.get("read_only"):
            handles.append(weakref.ref(conn))
        return conn

    def track_cursor(conn, *args, **kwargs):
        cursor = real_cursor(conn, *args, **kwargs)
        if retained == "cursor-and-open":
            handles.append(weakref.ref(cursor))
        return cursor

    def fail_extension(conn, query, *args, **kwargs):
        if retained == "failed-open" and query == "INSTALL vss" and any(
            conn is ref() for ref in handles
        ):
            raise OSError("extension initialization failed")
        return real_execute(conn, query, *args, **kwargs)

    def fail_retained_close(conn):
        if any(conn is ref() for ref in handles):
            raise OSError("retained backend close failed")
        return real_close(conn)

    monkeypatch.setattr(duckdb, "connect", track_connect)
    monkeypatch.setattr(duckdb.DuckDBPyConnection, "cursor", track_cursor)
    monkeypatch.setattr(duckdb.DuckDBPyConnection, "execute", fail_extension)
    monkeypatch.setattr(duckdb.DuckDBPyConnection, "close", fail_retained_close)
    try:
        with pytest.raises(OSError, match="retained backend close failed") as raised:
            provider.get_stats()
        assert handles
        assert all(ref() is not None for ref in handles)
        for reference in handles:
            assert reference().execute("SELECT 42").fetchone() == (42,)
        seed.close()
        with pytest.raises(DatabaseError, match="retained backend close failed"):
            provider.release_for_rust_pipeline()
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
        monkeypatch.undo()
        provider.release_for_rust_pipeline()
        assert raised.value is not None
        assert_closed_or_collected(handles)
        # Temporary release is not final close; it keeps ownership and a usable
        # executor so Python can reconnect after the native run.
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
        provider.connect()
        assert provider.get_stats()["files"] == 0
        provider.close()
        native.acquire_install_lease(str(path), "duckdb").close()
    finally:
        monkeypatch.undo()
        provider.close()
        seed.close()
        for reference in handles:
            if reference() is not None:
                real_close(reference())
