"""User actions recover the same provider after recoverable failed opens."""

import asyncio
import gc
import subprocess
import textwrap
import threading
import weakref
from pathlib import Path

import duckdb
import lancedb
import pytest

import chunkhound_native as native
from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.database_factory import create_services
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.providers.database.lancedb_provider import LanceDBProvider
from chunkhound.providers.database.lifecycle import DatabaseRecoveryRequiredError
from chunkhound.providers.database.ownership import DatabaseReaderCloseError
from chunkhound.providers.database.serial_executor import DatabaseOperationTimeoutError
from chunkhound.services.rust_pipeline_runner import run_rust_indexing_phase


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_search_reuses_provider_after_backend_open_failure(
    tmp_path, monkeypatch, backend
):
    path = tmp_path / ("chunks.db" if backend == "duckdb" else "lancedb.lancedb")
    cls = DuckDBProvider if backend == "duckdb" else LanceDBProvider
    boundary = duckdb if backend == "duckdb" else lancedb
    provider = cls(path, base_directory=tmp_path)
    real_connect = boundary.connect
    failed = False

    def connect(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise TimeoutError("transient backend open failure")
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(boundary, "connect", connect)
    try:
        with pytest.raises(TimeoutError, match="transient backend open failure"):
            provider.connect()
        native.acquire_install_lease(str(path), backend).close()
        rows, _ = provider.search_regex("never_matches")
        assert rows == []
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), backend)
    finally:
        provider.close()
    with pytest.raises(RuntimeError, match="clos"):
        provider.search_regex("never_matches")
    native.acquire_install_lease(str(path), backend).close()


def test_failed_open_cleanup_excludes_install_until_next_search(tmp_path, monkeypatch):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    real_execute = duckdb.DuckDBPyConnection.execute
    real_close = duckdb.DuckDBPyConnection.close
    fault = {"open": True, "close": True}

    def execute(conn, query, *args, **kwargs):
        if fault["open"] and query == "INSTALL vss":
            raise OSError("initial extension setup failed")
        return real_execute(conn, query, *args, **kwargs)

    def close(conn):
        if fault["close"]:
            raise OSError("retained backend close failed")
        return real_close(conn)

    monkeypatch.setattr(duckdb.DuckDBPyConnection, "execute", execute)
    monkeypatch.setattr(duckdb.DuckDBPyConnection, "close", close)
    try:
        with pytest.raises(Exception):
            provider.connect()
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
        with pytest.raises(Exception, match="cleanup|close|recovery"):
            provider.search_regex("never_matches")
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
        fault.update(open=False, close=False)
        assert provider.search_regex("never_matches")[0] == []
    finally:
        fault.update(open=False, close=False)
        provider.close()
    native.acquire_install_lease(str(path), "duckdb").close()


@pytest.mark.parametrize("sync_owner", [False, True])
@pytest.mark.parametrize("opening", ["manager", "executor"])
@pytest.mark.asyncio
async def test_cancelled_async_action_drains_open_past_sync_deadline(
    tmp_path, monkeypatch, sync_owner, opening
):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(
        path, base_directory=tmp_path,
        config=DatabaseConfig(execute_timeout_seconds=0.2),
    )
    entered = threading.Event()
    release = threading.Event()
    drained = threading.Event()
    real_connect = duckdb.connect
    real_execute = duckdb.DuckDBPyConnection.execute
    opens = 0
    searches = []

    def connect(database, *args, **kwargs):
        nonlocal opens
        if str(database) == str(path) and not kwargs.get("read_only"):
            opens += 1
            if opens == (1 if opening == "manager" else 2):
                entered.set()
                assert release.wait(10)
                try:
                    return real_connect(database, *args, **kwargs)
                finally:
                    drained.set()
        return real_connect(database, *args, **kwargs)

    def execute(conn, query, *args, **kwargs):
        if "regexp_matches" in query:
            searches.append(query)
        return real_execute(conn, query, *args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", connect)
    monkeypatch.setattr(duckdb.DuckDBPyConnection, "execute", execute)
    owner = (
        asyncio.create_task(asyncio.to_thread(provider.connect)) if sync_owner else None
    )
    action = None
    try:
        if owner is not None:
            assert await asyncio.to_thread(entered.wait, 5)
        action = asyncio.create_task(provider.search_regex_async("cancelled-action"))
        if owner is None:
            assert await asyncio.to_thread(entered.wait, 5)
        # Starting the coroutine reaches its readiness wait before cancellation.
        await asyncio.sleep(0)
        action.cancel()
        if owner is not None:
            with pytest.raises(DatabaseOperationTimeoutError):
                await asyncio.wait_for(asyncio.shield(owner), 2)
        done, _ = await asyncio.wait({action}, timeout=0.4)
        assert not done, "cancellation escaped before the backend worker drained"
        assert not drained.is_set()
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(action, 5)
        assert drained.is_set()
        assert searches == [], "the cancelled user action must never execute"
        assert (await provider.search_regex_async("later-action"))[0] == []
        assert searches
    finally:
        release.set()
        if action is not None:
            await asyncio.gather(action, return_exceptions=True)
        if owner is not None:
            await asyncio.gather(owner, return_exceptions=True)
        await asyncio.to_thread(provider.close)
    native.acquire_install_lease(str(path), "duckdb").close()


@pytest.mark.asyncio
async def test_delayed_reader_error_keeps_its_owner_after_later_attempt(
    tmp_path, monkeypatch
):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    owner = provider._database_ownership
    with duckdb.connect(str(path) + ".compact_backup") as backup:
        backup.execute("CREATE TABLE schema_version AS SELECT 1 AS version")
        backup.execute("CREATE TABLE files(id INTEGER)")
    provider._execute_in_db_thread_sync("disconnect", True)
    entered = threading.Event()
    release = threading.Event()
    real_exit = duckdb.DuckDBPyConnection.__exit__
    real_close = duckdb.DuckDBPyConnection.close
    failed_readers = []
    fault = True
    errors = []

    def exit_reader(conn, *args):
        if fault:
            failed_readers.append(conn)
            entered.set()
            assert release.wait(10)
            raise OSError("inspection exit failed")
        return real_exit(conn, *args)

    def close(conn):
        if fault and any(conn is reader for reader in failed_readers):
            raise OSError("inspection close failed")
        return real_close(conn)

    def connect():
        try:
            provider.connect()
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(duckdb.DuckDBPyConnection, "__exit__", exit_reader)
    monkeypatch.setattr(duckdb.DuckDBPyConnection, "close", close)
    caller = threading.Thread(target=connect)
    caller.start()
    joined = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        joined = asyncio.create_task(provider.search_regex_async("joined-action"))
        await asyncio.sleep(0)
        release.set()
        # Deliberately hold this loop's delivery until a later successful attempt
        # and final close clear the provider's current owner.
        caller.join(5)
        assert not caller.is_alive()
        assert len(errors) == 1
        fault = False
        owner.close_readers()
        provider.connect()
        provider.close()
        assert provider._database_ownership is None
        with pytest.raises(DatabaseReaderCloseError) as raised:
            await joined
        assert raised.value.ownership is owner
        assert raised.value is not errors[0]
        raised.value.retry_close()
        native.acquire_install_lease(str(path), "duckdb").close()
    finally:
        fault = False
        release.set()
        caller.join(5)
        if joined is not None:
            await asyncio.gather(joined, return_exceptions=True)
        provider.close()
        owner.close()


@pytest.mark.parametrize("live_peer", [False, True])
@pytest.mark.parametrize("opening", ["manager", "executor"])
def test_failed_open_reports_setup_and_cleanup_separately(
    tmp_path, monkeypatch, live_peer, opening
):
    path = tmp_path / "chunks.db"
    peer = DuckDBProvider(path, base_directory=tmp_path)
    peer.connect()
    if not live_peer:
        peer.close()
    provider = DuckDBProvider(path, base_directory=tmp_path)
    real_execute = duckdb.DuckDBPyConnection.execute
    real_close = duckdb.DuckDBPyConnection.close
    setup_attempts = 0
    setup_failed = False
    fault_enabled = True

    def execute(conn, query, *args, **kwargs):
        nonlocal setup_attempts, setup_failed
        if fault_enabled and query == "INSTALL vss":
            setup_attempts += 1
            if setup_attempts == (1 if opening == "manager" else 2):
                setup_failed = True
                # A legitimate setup cause must not be mistaken for the setup
                # operation itself by picking the deepest exception blindly.
                raise OSError("extension setup failed") from ValueError(
                    "setup input invalid"
                )
        return real_execute(conn, query, *args, **kwargs)

    def close(conn):
        if fault_enabled and setup_failed:
            raise OSError("retained backend close failed")
        return real_close(conn)

    monkeypatch.setattr(duckdb.DuckDBPyConnection, "execute", execute)
    monkeypatch.setattr(duckdb.DuckDBPyConnection, "close", close)
    try:
        with pytest.raises(DatabaseRecoveryRequiredError) as raised:
            provider.connect()
        # This is the user-facing diagnostic, not a search of a nested chain.
        failure = raised.value.failure
        assert failure.original is not None
        assert failure.original.message == "extension setup failed"
        assert failure.original.cause[0].message == "setup input invalid"
        assert failure.cleanup.message == "retained backend close failed"
        assert failure.stage == "failed-open cleanup"
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
        fault_enabled = False
        assert provider.search_regex("never_matches")[0] == []
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
    finally:
        fault_enabled = False
        provider.close()
        peer.close()
    native.acquire_install_lease(str(path), "duckdb").close()


def make_services(root):
    config = Config(
        target_dir=root,
        database={"path": root / "db"},
        embeddings_disabled=True,
        analytics={"enabled": False},
        indexing={"max_concurrent": 1},
        skip_layers={"env", "global", "local_config", "config_file", "cli"},
    )
    return config, create_services(config.database.get_db_path(), config)


async def run_empty_phase(services, config, root):
    return await run_rust_indexing_phase(
        db=services.provider, config=config, embedding_provider=None, progress=None,
        files_to_process=[], directory=root, force_reindex=False, do_cleanup=False,
        diff_task=None, parse_task=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("next_action", ["search", "index"])
async def test_next_action_recovers_rust_reconnect_failure(
    tmp_path, monkeypatch, next_action
):
    config, services = make_services(tmp_path)
    provider = services.provider
    sidecar = Path(str(provider.db_path) + ".root.json")
    provider.ensure_indexed_root_identity(
        requested_root=tmp_path, allow_claim_if_missing=True
    )
    read_text = Path.read_text
    fail = True

    def read(path, *args, **kwargs):
        if path == sidecar and fail:
            raise OSError("trusted reconnect sidecar read failed")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    try:
        with pytest.raises(DatabaseRecoveryRequiredError, match="sidecar read failed"):
            await run_empty_phase(services, config, tmp_path)
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(provider.db_path), "duckdb")
        fail = False
        if next_action == "search":
            rows, _ = await services.search_service.search_regex_async("never_matches")
            assert rows == []
        else:
            result = await services.indexing_coordinator.process_directory(
                tmp_path, patterns=["**/*.py"]
            )
            assert provider.is_connected
            assert result.get("status") != "error", result
        assert services.provider is provider
        assert provider.execute_query("SELECT 42 AS value") == [{"value": 42}]
    finally:
        fail = False
        try:
            await services.search_service.search_regex_async("never_matches")
        finally:
            provider.close()
    native.acquire_install_lease(str(provider.db_path), "duckdb").close()


@pytest.mark.asyncio
async def test_concurrent_cancelled_searches_share_failed_recovery_and_later_retry(
    tmp_path, monkeypatch
):
    config, services = make_services(tmp_path)
    provider = services.provider
    sidecar = Path(str(provider.db_path) + ".root.json")
    provider.ensure_indexed_root_identity(
        requested_root=tmp_path, allow_claim_if_missing=True
    )
    original_read = Path.read_text
    mode = "initial-failure"
    entered, release = threading.Event(), threading.Event()
    attempts = 0
    tasks = []

    def read(path, *args, **kwargs):
        nonlocal attempts
        if path == sidecar and mode != "success":
            if mode == "recovery-failure":
                attempts += 1
                entered.set()
                assert release.wait(20), "test failed to release filesystem boundary"
            raise OSError("reconnect unavailable")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    try:
        with pytest.raises(DatabaseRecoveryRequiredError):
            await run_empty_phase(services, config, tmp_path)
        mode = "recovery-failure"
        for _ in range(3):
            tasks.append(asyncio.create_task(
                services.search_service.search_regex_async("never_matches")
            ))
        assert await asyncio.to_thread(entered.wait, 10)
        for _ in range(3):
            tasks[0].cancel()
            await asyncio.sleep(0.01)
            assert not tasks[0].done()
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(provider.db_path), "duckdb")
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert isinstance(results[0], asyncio.CancelledError)
        assert all(
            isinstance(error, DatabaseRecoveryRequiredError) for error in results[1:]
        )
        assert attempts == 1
        assert str(results[1]) == str(results[2])
        mode = "success"
        rows, _ = await services.search_service.search_regex_async("never_matches")
        assert rows == []
    finally:
        mode = "success"
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await services.search_service.search_regex_async("never_matches")
        finally:
            provider.close()
    native.acquire_install_lease(str(provider.db_path), "duckdb").close()


def test_failed_open_reports_original_and_cleanup_without_retaining_raw_error(
    tmp_path, monkeypatch
):
    provider = DuckDBProvider(tmp_path / "chunks.db", base_directory=tmp_path)
    real_execute = duckdb.DuckDBPyConnection.execute
    real_close = duckdb.DuckDBPyConnection.close
    fail = True
    observed = []

    class Payload:
        pass

    def execute(conn, query, *args, **kwargs):
        if fail and query == "INSTALL vss":
            payload = Payload()
            observed.append(weakref.ref(payload))
            raise OSError(
                "extension setup failure "
                "https://user:password@example.invalid/?token=secret"
            )
        return real_execute(conn, query, *args, **kwargs)

    def close(conn):
        if fail:
            raise OSError("retained handle cleanup failure")
        return real_close(conn)

    monkeypatch.setattr(duckdb.DuckDBPyConnection, "execute", execute)
    monkeypatch.setattr(duckdb.DuckDBPyConnection, "close", close)
    try:
        with pytest.raises(DatabaseRecoveryRequiredError) as caught:
            provider.connect()
        message = str(caught.value)
        assert "extension setup failure" in message
        assert "cleanup failure" in message
        assert "failed-open cleanup" in message
        assert "password" not in message and "example.invalid" not in message
        assert caught.value.__cause__ is not None
        del caught
        gc.collect()
        assert observed and all(ref() is None for ref in observed)
        fail = False
        assert provider.search_regex("never_matches")[0] == []
    finally:
        fail = False
        provider.close()
    native.acquire_install_lease(str(provider.db_path), "duckdb").close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    [
        "coordinator-file",
        "coordinator-directory",
        "provider-file",
        "provider-directory",
    ],
)
async def test_refused_index_action_does_not_claim_root_or_inspect_sources(
    tmp_path, monkeypatch, action
):
    import os

    config = Config(
        target_dir=tmp_path,
        database={"path": tmp_path / "db"},
        embeddings_disabled=True,
        analytics={"enabled": False},
        indexing={"discovery_backend": "python", "parallel_discovery": False},
        skip_layers={"env", "global", "local_config", "config_file", "cli"},
    )
    services = create_services(
        config.database.get_db_path(), config, connect_database=False
    )
    provider = services.provider
    # A legacy database has no root sidecar until an admitted mutation claims it.
    sidecar = Path(str(provider.db_path) + ".root.json")
    assert not sidecar.exists()
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = source_root / "missing.py"
    real_connect = duckdb.connect
    fail = True
    effects = []
    real_link, real_scandir, real_stat = os.link, os.scandir, Path.stat

    def connect(database, *args, **kwargs):
        if fail and str(database) == str(provider.db_path):
            raise OSError("index admission unavailable")
        return real_connect(database, *args, **kwargs)

    def link(src, dst, *args, **kwargs):
        if Path(dst) == sidecar:
            effects.append("root claimed")
        return real_link(src, dst, *args, **kwargs)

    def scandir(path):
        if Path(path) == source_root:
            effects.append("sources scanned")
        return real_scandir(path)

    def stat(path, *args, **kwargs):
        if path == source:
            effects.append("source inspected")
        return real_stat(path, *args, **kwargs)

    # Use supported Python discovery, not a mocked coordinator/discovery helper.
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "0")
    monkeypatch.setattr(duckdb, "connect", connect)
    try:
        with pytest.raises(OSError, match="index admission unavailable"):
            provider.connect()
        monkeypatch.setattr(os, "link", link)
        monkeypatch.setattr(os, "scandir", scandir)
        monkeypatch.setattr(Path, "stat", stat)
        caller = (
            services.indexing_coordinator
            if action.startswith("coordinator") else provider
        )
        error = None
        try:
            if action.endswith("directory"):
                await caller.process_directory(source_root, patterns=["**/*.py"])
            else:
                await caller.process_file(source, skip_embeddings=True)
        except OSError as caught:
            error = caught
        assert effects == [], "refused admission must precede root/source work"
        assert not sidecar.exists()
        assert error is not None and "index admission unavailable" in str(error)
    finally:
        fail = False
        provider.close()
    native.acquire_install_lease(str(provider.db_path), "duckdb").close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["file", "empty-directory"])
async def test_direct_provider_first_index_action_initializes_local_services(
    tmp_path, action
):
    provider = DuckDBProvider(tmp_path / "chunks.db", base_directory=tmp_path)
    try:
        if action == "file":
            # Existing per-file validation remains after admission.
            result = await provider.process_file(tmp_path / "missing.py")
            assert result["status"] == "error"
            assert "File not found" in result["error"]
        else:
            result = await provider.process_directory(tmp_path, patterns=["**/*.py"])
            assert result["status"] == "success"
            assert result["files_processed"] == 0
        assert provider.is_connected, "even an empty action must complete admission"
        assert provider._indexing_coordinator is not None
        assert provider._search_service.search_regex("never_matches")[0] == []
    finally:
        provider.close()
    native.acquire_install_lease(str(provider.db_path), "duckdb").close()


def test_unacknowledged_native_failure_requires_restart_not_retry(tmp_path):
    # Only the native-extension boundary fails. The real read-only token and
    # provider retain exclusion until subprocess exit; never fabricate an ack.
    script = textwrap.dedent('''
        import asyncio
        import sys
        from pathlib import Path
        import chunkhound_native as native
        from chunkhound.core.config.config import Config
        from chunkhound.database_factory import create_services
        from chunkhound.providers.database.lifecycle import (
            DatabaseRecoveryRequiredError,
        )
        from chunkhound.services.rust_pipeline_runner import run_rust_indexing_phase

        clients = []

        class EmbeddingClient:
            def __init__(self):
                self.loop = None
                self.closed = False
                clients.append(self)
            async def embed(self, texts):
                self.loop = asyncio.get_running_loop()
                return [[1.0] * 8 for _ in texts]
            async def shutdown(self):
                self.closed = True

        class FailedNativePipeline:
            def __init__(self, config):
                pass
            def run(self, **kwargs):
                vectors = kwargs["embed_batch_callback"](["callback cache probe"] * 100)
                assert len(vectors) == 100
                raise OSError("native cleanup did not acknowledge completion")

        async def main():
            root = Path(sys.argv[1])
            config = Config(
                target_dir=root, database={"path": root / "db"},
                embedding={
                    "provider": "openai", "model": "hub-test", "output_dims": 8,
                    "api_key": "local-test-key", "max_concurrent_batches": 1,
                },
                analytics={"enabled": False},
                skip_layers={"env", "global", "local_config", "config_file", "cli"},
            )
            services = create_services(config.database.get_db_path(), config)
            from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
            EmbeddingProviderFactory.create_provider = staticmethod(
                lambda config: EmbeddingClient()
            )
            native.IndexingPipeline = FailedNativePipeline
            try:
                await run_rust_indexing_phase(
                    db=services.provider, config=config,
                    embedding_provider=services.search_service._embedding_provider,
                    progress=None, files_to_process=[], directory=root,
                    force_reindex=False, do_cleanup=False,
                    diff_task=None, parse_task=None,
                )
            except DatabaseRecoveryRequiredError as error:
                assert error.failure.restart_required
                assert error.failure.original is not None
                assert "native cleanup" in error.failure.original.message
            else:
                raise AssertionError("native failure was hidden")
            assert len(clients) == 1
            assert clients[0].loop is not None
            assert not clients[0].closed, "no client shutdown without native ack"
            assert not clients[0].loop.is_closed(), "no loop close without native ack"
            for _ in range(2):
                try:
                    await services.search_service.search_regex_async("never_matches")
                except DatabaseRecoveryRequiredError as error:
                    assert "restart" in str(error).lower()
                    assert "not acknowledged" in str(error).lower()
                else:
                    raise AssertionError(
                        "recovery manufactured a native acknowledgment"
                    )
            for _ in range(2):
                try:
                    services.provider.close()
                except RuntimeError as error:
                    assert "not acknowledged" in str(error).lower()
                else:
                    raise AssertionError("close manufactured a native acknowledgment")
                assert not clients[0].closed
                assert not clients[0].loop.is_closed()
                try:
                    native.acquire_install_lease(str(services.provider.db_path), "duckdb")
                except native.DatabaseBusyError:
                    pass
                else:
                    raise AssertionError("unacknowledged owner released exclusion")
        asyncio.run(main())
    ''')
    result = subprocess.run(
        ["uv", "run", "--no-sync", "python", "-c", script, str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    native.acquire_install_lease(str(tmp_path / "db" / "chunks.db"), "duckdb").close()
