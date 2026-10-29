import asyncio
import gc
import hashlib
import json
import shutil
import tarfile
import threading
import traceback
import weakref
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import pytest
from loguru import logger

import chunkhound.hub.errors as hub_errors
import chunkhound_native as native
from chunkhound.core.config.config import Config
from chunkhound.core.config.hub_config import ArchiveLimits
from chunkhound.core.embedding_model_drift import ActiveModel, write_active_model
from chunkhound.hub.client import local_needs_pull, pull
from chunkhound.hub.errors import HubDatabaseMismatch, HubInstallationInterrupted
from chunkhound.hub.inspect import inspect_database
from chunkhound.hub.manifest import configured_key
from chunkhound.providers.database import ownership
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.providers.database.lancedb_provider import LanceDBProvider
from chunkhound.providers.database.ownership import (
    DatabaseOwnership,
    DatabaseReaderCloseError,
)
from tests.hub.activation_support import config_for, publish
from tests.hub.artifacts import build_database
from tests.hub.test_index_format import snapshot


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
async def test_pull_installs_replaces_and_is_searchable(tmp_path, hub_http, backend):
    config = config_for(tmp_path / "checkout", hub_http, backend)
    target = config.database.get_db_path()
    neighbor = target.parent / "notes.txt"
    neighbor.write_text("operator-owned")
    for prefix in ("first", "replacement"):
        source = build_database(tmp_path / prefix, backend, symbol_prefix=prefix)
        publish(hub_http, source, prefix=prefix)
        if prefix == "replacement":
            target.with_name(target.name + ".embedding.json").write_text(
                "old active model"
            )
            if backend == "duckdb":
                target.with_name(target.name + ".root.json").write_text(
                    "old publisher root"
                )
        assert await pull(config)
        assert not target.with_name(target.name + ".embedding.json").exists()
        if backend == "duckdb":
            assert not target.with_name(target.name + ".root.json").exists()
        cls = DuckDBProvider if backend == "duckdb" else LanceDBProvider
        db = cls(target, base_directory=config.target_dir)
        db.connect()
        try:
            rows, _ = db.search_regex(prefix + "_0")
            assert rows
            if prefix == "replacement":
                rows, _ = db.search_regex("first_0")
                assert rows == []
        finally:
            db.disconnect()
    assert neighbor.read_text() == "operator-owned"
    assert not list(target.parent.glob(".chunkhound-pull-*"))


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
async def test_downloaded_mismatch_preserves_existing_target_and_sidecars(
    tmp_path, hub_http, backend
):
    config = config_for(tmp_path / "checkout", hub_http, backend)
    target = build_database(config.database.path, backend)
    sidecars = [
        target.with_name(target.name + suffix)
        for suffix in (".root.json", ".embedding.json")
    ]
    for path in sidecars:
        path.write_text("local state must survive")
    before = {path: snapshot(path) for path in [target, *sidecars]}
    source = build_database(tmp_path / "wrong", backend, model="different")
    publish(hub_http, source)  # Manifest intentionally claims hub-test.
    with pytest.raises(HubDatabaseMismatch):
        await pull(config)
    assert {path: snapshot(path) for path in before} == before


@pytest.mark.parametrize("failure", ["http", "checksum", "archive", "space"])
async def test_preinstall_failures_preserve_local_index(
    tmp_path, hub_http, monkeypatch, failure
):
    config = config_for(tmp_path / "checkout", hub_http)
    target = build_database(config.database.path)
    before = snapshot(target)
    source = build_database(tmp_path / "published")
    manifest, route = publish(hub_http, source)
    if failure == "http":
        hub_http.status = 503
    elif failure == "checksum":
        manifest["embeddings"][0]["versions"][0]["artifact"]["sha256"] = "0" * 64
        hub_http.resources["/catalogue?token=secret"] = json.dumps(manifest).encode()
    elif failure == "archive":
        import hashlib

        body = b"not a gzip archive"
        hub_http.resources[route] = body
        artifact = manifest["embeddings"][0]["versions"][0]["artifact"]
        artifact.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
        hub_http.resources["/catalogue?token=secret"] = json.dumps(manifest).encode()
    else:
        usage = shutil.disk_usage(tmp_path)
        monkeypatch.setattr(shutil, "disk_usage", lambda _: usage._replace(free=0))
    assert not await pull(config)
    assert snapshot(target) == before


@pytest.mark.parametrize("automatic", [False, True])
async def test_busy_pull_does_not_replace_or_open_partial_state(
    tmp_path, hub_http, automatic
):
    config = config_for(tmp_path / "checkout", hub_http)
    target = config.database.get_db_path()
    publish(hub_http, build_database(tmp_path / "published"))
    # A live cooperative holder can own an as-yet absent file.
    held = native.acquire_database_lease(str(target))
    try:
        assert not await pull(config, automatic=automatic)
        assert not target.exists()
    finally:
        held.close()
    assert await pull(config)


async def test_auto_marker_is_fatal_without_requests_but_explicit_recovers(
    tmp_path, hub_http
):
    config = config_for(tmp_path / "checkout", hub_http)
    target = config.database.get_db_path()
    marker = target.with_name(target.name + ".chunkhound-install.json")
    marker.write_bytes(b"{")  # Incomplete marker also means interrupted.
    publish(hub_http, build_database(tmp_path / "published"))
    with pytest.raises(HubInstallationInterrupted):
        await pull(config, automatic=True)
    assert hub_http.requests == []
    assert marker.read_bytes() == b"{"
    assert await pull(config)
    assert not marker.exists()
    assert inspect_database(target, "duckdb").groups


@pytest.mark.parametrize("racing_marker", [False, True])
async def test_auto_rechecks_under_exclusive_ownership(
    tmp_path, hub_http, racing_marker
):
    config = config_for(tmp_path / "checkout", hub_http)
    target = config.database.get_db_path()
    publish(hub_http, build_database(tmp_path / "published"))
    competitor = build_database(tmp_path / "competitor", symbol_prefix="race_winner")
    reached, release = threading.Event(), threading.Event()

    def delay_artifact(route):
        if route.startswith("/base.tar.gz"):
            reached.set()
            assert release.wait(20)

    hub_http.before_get = delay_artifact
    task = asyncio.create_task(pull(config, automatic=True))
    try:
        assert await asyncio.to_thread(reached.wait, 20)
        lease = native.acquire_install_lease(str(target), "duckdb")
        try:
            lease.install(str(competitor))
        finally:
            lease.close()
        before = snapshot(target)
        if racing_marker:
            target.with_name(target.name + ".chunkhound-install.json").write_bytes(b"{")
        release.set()
        if racing_marker:
            with pytest.raises(HubInstallationInterrupted):
                await task
        else:
            assert not await task  # Aligned race winner must not be replaced.
        assert snapshot(target) == before
    finally:
        release.set()
        if not task.done():
            await task


async def test_committed_pull_cleanup_failure_is_only_a_warning(
    tmp_path, hub_http, monkeypatch
):
    config = config_for(tmp_path / "checkout", hub_http)
    publish(hub_http, build_database(tmp_path / "published"))
    original = TemporaryDirectory.cleanup
    messages = []
    sink = logger.add(lambda record: messages.append(str(record)))

    def failed_cleanup(workspace):
        if Path(workspace.name).name.startswith(".chunkhound-pull-"):
            raise OSError("private-path?secret")
        return original(workspace)

    monkeypatch.setattr(TemporaryDirectory, "cleanup", failed_cleanup)
    try:
        assert await pull(config)
    finally:
        logger.remove(sink)
    assert inspect_database(config.database.get_db_path(), "duckdb").groups
    text = "".join(messages)
    assert "staging cleanup" in text.lower()
    assert "private-path" not in text and "?secret" not in text


async def test_repeated_cancellation_waits_for_install_and_workspace_cleanup(
    tmp_path, hub_http, monkeypatch
):
    config = config_for(tmp_path / "checkout", hub_http)
    target = config.database.get_db_path()
    publish(hub_http, build_database(tmp_path / "published"))
    downloaded, entered, release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original_exists = Path.exists

    def block_exclusive_recheck(path):
        # Deterministic filesystem boundary, after initial trigger and download.
        if path == target and downloaded.is_set() and not entered.is_set():
            entered.set()
            assert release.wait(20)
        return original_exists(path)

    def mark_download(route):
        if route.startswith("/base.tar.gz"):
            downloaded.set()

    monkeypatch.setattr(Path, "exists", block_exclusive_recheck)
    hub_http.before_get = mark_download
    task = asyncio.create_task(pull(config, automatic=True))
    try:
        assert await asyncio.to_thread(entered.wait, 20)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert list(target.parent.glob(".chunkhound-pull-*"))
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(target), "duckdb")
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert target.exists()
    assert not list(target.parent.glob(".chunkhound-pull-*"))
    assert inspect_database(target, "duckdb").groups


@pytest.mark.parametrize("boundary", ["initial_inspection", "installation_recheck"])
async def test_cancelled_pull_reports_interruption_after_worker_drains(
    tmp_path, hub_http, monkeypatch, boundary
):
    config = config_for(tmp_path / "private-checkout", hub_http)
    target = config.database.get_db_path()
    marker = target.with_name(target.name + ".chunkhound-install.json")
    entered, release, completed = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    acquire_install = native.acquire_install_lease
    messages = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        filter=lambda record: record["extra"].get("hub_client", False)
        and record["level"].name == "ERROR",
    )

    if boundary == "initial_inspection":
        marker.write_bytes(b"{")
        acquire_database = ownership.acquire_database_lease

        def pause_native_inspection(path):
            try:
                # Real native marker detection raises before any backend opens.
                return acquire_database(path)
            finally:
                entered.set()
                assert release.wait(20)
                completed.set()

        monkeypatch.setattr(
            ownership, "acquire_database_lease", pause_native_inspection
        )
    else:
        publish(hub_http, build_database(tmp_path / "published"))

        def pause_native_install(path, backend):
            # A surviving marker appeared after the initial cold-state check.
            marker.write_bytes(b"{")
            lease = acquire_install(path, backend)
            try:
                entered.set()
                assert release.wait(20)
                return lease
            except BaseException:
                lease.close()
                raise
            finally:
                completed.set()

        monkeypatch.setattr(native, "acquire_install_lease", pause_native_install)

    task = asyncio.create_task(pull(config, automatic=True))
    try:
        try:
            assert await asyncio.to_thread(entered.wait, 20)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                assert not completed.is_set()
                assert messages == []
                if boundary == "installation_recheck":
                    assert list(target.parent.glob(".chunkhound-pull-*"))
                    with pytest.raises(native.DatabaseBusyError):
                        acquire_install(str(target), "duckdb")
                else:
                    assert hub_http.requests == []
        finally:
            release.set()
        with pytest.raises(HubInstallationInterrupted) as failure:
            await task
        assert completed.is_set()
        assert marker.read_bytes() == b"{"
        assert not target.exists()
        assert not list(target.parent.glob(".chunkhound-pull-*"))
        diagnostic = (
            "Hub installation was interrupted; run explicit chunkhound pull again. "
            "Local work may require re-indexing"
        )
        assert messages == [diagnostic]
        assert str(failure.value) == diagnostic
        # Ownership ends only after completion; the marker still refuses readers.
        lease = acquire_install(str(target), "duckdb")
        lease.close()
        with pytest.raises(native.DatabaseInterruptedError):
            native.acquire_database_lease(str(target))
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        logger.remove(sink)


@pytest.mark.parametrize("layout", ["memory", "read_only", "explicit", "legacy"])
async def test_unsupported_install_layout_makes_no_request(tmp_path, hub_http, layout):
    config = config_for(tmp_path / "checkout", hub_http)
    if layout == "memory":
        config.database.path = Path(":memory:")
    elif layout == "read_only":
        config.database.read_only = True
    elif layout == "explicit":
        config.database.path = config.target_dir / "chunks.db"
    else:
        legacy = config.target_dir / "legacy-file"
        legacy.write_bytes(b"legacy")
        config.database.path = legacy
    assert not await pull(config)
    assert hub_http.requests == []


@pytest.mark.parametrize("automatic", [False, True])
async def test_unknown_identity_does_not_probe_services(tmp_path, hub_http, automatic):
    config = config_for(tmp_path / "checkout", hub_http, model="unknown-offline-model")
    config.embedding.output_dims = None
    assert not await pull(config, automatic=automatic)
    assert hub_http.requests == []


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
async def test_two_pulls_and_cold_provider_never_observe_partial_install(
    tmp_path, hub_http, monkeypatch, backend
):
    config = config_for(tmp_path / "checkout", hub_http, backend)
    target = config.database.get_db_path()
    publish(hub_http, build_database(tmp_path / "published", backend))
    downloaded, entered, release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original_exists = Path.exists

    def mark_download(route):
        if route.startswith("/base.tar.gz"):
            downloaded.set()

    def pause_exclusive_recheck(path):
        if path == target and downloaded.is_set() and not entered.is_set():
            entered.set()
            assert release.wait(20)
        return original_exists(path)

    hub_http.before_get = mark_download
    monkeypatch.setattr(Path, "exists", pause_exclusive_recheck)
    first = asyncio.create_task(pull(config, automatic=True))
    cls = DuckDBProvider if backend == "duckdb" else LanceDBProvider
    opener = cls(target, base_directory=config.target_dir)
    try:
        assert await asyncio.to_thread(entered.wait, 20)
        # Second explicit pull downloads/verifies, but cannot install over first.
        assert not await pull(config)
        with pytest.raises(native.DatabaseBusyError):
            opener.connect()
        assert not target.exists()
    finally:
        opener.close()
        release.set()
        await asyncio.gather(first, return_exceptions=True)
    assert first.result() is True
    assert inspect_database(target, backend).groups


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
async def test_failed_explicit_redownload_preserves_interruption_and_partial_state(
    tmp_path, hub_http, backend
):
    config = config_for(tmp_path / "checkout", hub_http, backend)
    target = build_database(config.database.path, backend)
    marker = Path(str(target) + ".chunkhound-install.json")
    marker.write_bytes(b"{")
    before = snapshot(target.parent)
    publish(hub_http, build_database(tmp_path / "published", backend))
    hub_http.status = 503
    assert not await pull(config)
    assert snapshot(target.parent) == before
    with pytest.raises(native.DatabaseInterruptedError):
        inspect_database(target, backend)


def test_asyncio_shutdown_waits_for_install_and_workspace_cleanup(
    tmp_path, hub_http, monkeypatch
):
    """Real asyncio.run shutdown cancels all Tasks, but not installer threads."""
    config = config_for(tmp_path / "checkout", hub_http)
    target = config.database.get_db_path()
    publish(hub_http, build_database(tmp_path / "published"))
    downloaded, entered, release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original_exists = Path.exists
    failures = []

    def block_exclusive_recheck(path):
        if path == target and downloaded.is_set() and not entered.is_set():
            entered.set()
            assert release.wait(20)
        return original_exists(path)

    def mark_download(route):
        if route.startswith("/base.tar.gz"):
            downloaded.set()

    monkeypatch.setattr(Path, "exists", block_exclusive_recheck)
    hub_http.before_get = mark_download

    def probe(loop, task, remaining):
        try:
            assert not task.done(), "pull returned while installer was blocked"
            assert list(target.parent.glob(".chunkhound-pull-*"))
            assert not target.exists()
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(target), "duckdb")
            if remaining:
                for pending in asyncio.all_tasks(loop):
                    pending.cancel()
                loop.call_later(0.02, probe, loop, task, remaining - 1)
                return
        except BaseException as error:
            failures.append(error)
        release.set()

    async def main():
        task = asyncio.create_task(pull(config, automatic=True))
        try:
            assert await asyncio.to_thread(entered.wait, 15)
        except BaseException:
            release.set()
            raise
        loop = asyncio.get_running_loop()
        loop.call_later(0.02, probe, loop, task, 2)
        # Returning invokes real all-Tasks cancellation in asyncio.run.
        return task

    try:
        task = asyncio.run(main())
        if failures:
            raise failures[0]
        assert task.cancelled()
        assert target.exists()
        assert not list(target.parent.glob(".chunkhound-pull-*"))
        assert inspect_database(target, "duckdb").groups
    finally:
        release.set()


@contextmanager
def reader_close_fault(
    monkeypatch, wants_path, *, entered=None, release=None, refuse_cleanup=False
):
    readers = []
    real_exit = duckdb.DuckDBPyConnection.__exit__
    real_close = duckdb.DuckDBPyConnection.close
    paused = False
    held_workspaces = []
    real_cleanup = TemporaryDirectory.cleanup

    def fail_candidate_cleanup(workspace):
        # Emulate a filesystem refusing removal of a still-open candidate.
        # Keep its ORIGINAL lock inode and workspace alive for the real probe.
        if any(Path(workspace.name) in path.parents for _, path in readers):
            held_workspaces.append(workspace)
            raise OSError("retained candidate is still open")
        return real_cleanup(workspace)

    def fail_exit(conn, *args):
        paths = [
            Path(row[0])
            for row in conn.execute(
                "SELECT path FROM system.main.duckdb_databases() "
                "WHERE NOT internal AND path IS NOT NULL"
            ).fetchall()
        ]
        selected = next((path for path in paths if wants_path(path)), None)
        if selected is not None:
            readers.append((weakref.ref(conn), selected))
            raise OSError("raw-reader-detail?token=REVIEW_SECRET")
        return real_exit(conn, *args)

    def fail_close(conn):
        nonlocal paused
        if any(ref() is conn for ref, _ in readers):
            if entered is not None and not paused:
                paused = True
                entered.set()
                assert release.wait(20)
            raise OSError("raw-close-detail?token=REVIEW_SECRET")
        return real_close(conn)

    with monkeypatch.context() as fault:
        fault.setattr(duckdb.DuckDBPyConnection, "__exit__", fail_exit)
        fault.setattr(duckdb.DuckDBPyConnection, "close", fail_close)
        if refuse_cleanup:
            fault.setattr(TemporaryDirectory, "cleanup", fail_candidate_cleanup)
        # Ordinary cleanup is deliberately unpatched unless refusal is requested.
        # Callers must retry reader shutdown before removing any held workspace.
        yield readers, held_workspaces


def retained_reader_error(error):
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, DatabaseReaderCloseError):
            return current
        pending.extend([
            current.__cause__, current.__context__,
            getattr(current, "reader_error", None),
        ])
    raise AssertionError("Expected a retained-reader capability in failure chain")


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("refuse_cleanup", [False, True])
async def test_candidate_workspace_survives_reader_failure_until_retry(
    tmp_path, hub_http, monkeypatch, automatic, refuse_cleanup
):
    config = config_for(tmp_path / "checkout", hub_http)
    target = config.database.get_db_path()
    publish(hub_http, build_database(tmp_path / "published"))
    failure = None
    held = []
    try:
        with reader_close_fault(
            monkeypatch,
            lambda path: any(
                p.name.startswith(".chunkhound-pull-") for p in path.parents
            ),
            refuse_cleanup=refuse_cleanup,
        ) as (readers, held):
            with pytest.raises(hub_errors.HubReaderCloseError) as caught:
                await pull(config, automatic=automatic)
            failure = caught.value
            candidate = readers[0][1]
            workspace = next(
                p for p in candidate.parents if p.name.startswith(".chunkhound-pull-")
            )
            assert candidate.exists()
            assert readers[0][0]().execute("SELECT 42").fetchone() == (42,)
            failure.ownership.assert_covers(candidate)
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(candidate), "duckdb")
            with pytest.raises(DatabaseReaderCloseError):
                failure.retry_close()
            assert candidate.exists()
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(candidate), "duckdb")
            assert not target.exists()
        failure.retry_close()
        failure.retry_close()
        assert not workspace.exists()
        assert not target.exists()
    finally:
        if failure is not None:
            failure.retry_close()
        for retained in held:
            retained.cleanup()


async def test_candidate_workspace_survives_abandoned_error_gc(
    tmp_path, hub_http, monkeypatch
):
    config = config_for(tmp_path / "checkout", hub_http)
    target = config.database.get_db_path()
    publish(hub_http, build_database(tmp_path / "published"))
    original = None
    workspace = None

    async def abandon_wrapper():
        with pytest.raises(hub_errors.HubReaderCloseError) as caught:
            await pull(config)
        failure = caught.value
        original = failure.reader_error
        wrapper_ref = weakref.ref(failure)
        # Only the raw retry capability escapes; no traceback/context keeps the
        # hub error, preparation frame, or its TemporaryDirectory alive.
        pending, seen = [failure], set()
        while pending:
            error = pending.pop()
            if error is None or id(error) in seen:
                continue
            seen.add(id(error))
            pending.extend([error.__cause__, error.__context__])
            if error.__traceback__ is not None:
                traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
            error.__cause__ = None
            error.__context__ = None
        return original, wrapper_ref

    try:
        with reader_close_fault(
            monkeypatch,
            lambda path: any(
                p.name.startswith(".chunkhound-pull-") for p in path.parents
            ),
        ) as (readers, _):
            original, wrapper_ref = await abandon_wrapper()
            candidate = readers[0][1]
            workspace = next(
                p for p in candidate.parents if p.name.startswith(".chunkhound-pull-")
            )
            gc.collect()
            assert wrapper_ref() is None
            assert candidate.exists()
            assert readers[0][0]().execute("SELECT 42").fetchone() == (42,)
            original.ownership.assert_covers(candidate)
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(candidate), "duckdb")
            assert not target.exists()
        original.retry_close()
        native.acquire_install_lease(str(candidate), "duckdb").close()
    finally:
        if original is not None:
            original.retry_close()
            # An abandoned wrapper may retain staging until process exit. Only
            # after proven shutdown may test teardown remove the private path.
            if workspace is not None and workspace.exists():
                shutil.rmtree(workspace)


async def test_candidate_workspace_cleanup_refusal_is_sanitized_and_retryable(
    tmp_path, hub_http, monkeypatch
):
    config = config_for(tmp_path / "checkout", hub_http)
    target = config.database.get_db_path()
    publish(hub_http, build_database(tmp_path / "published"))
    failure = None
    real_rmtree = shutil.rmtree
    try:
        with reader_close_fault(
            monkeypatch,
            lambda path: any(
                p.name.startswith(".chunkhound-pull-") for p in path.parents
            ),
        ) as (readers, _):
            with pytest.raises(hub_errors.HubReaderCloseError) as caught:
                await pull(config)
            failure = caught.value
            original = failure.reader_error
            candidate = readers[0][1]
            workspace = next(
                p for p in candidate.parents if p.name.startswith(".chunkhound-pull-")
            )
            assert candidate.exists()
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(candidate), "duckdb")
        # Direct original retry remains safe; the public retry later finishes
        # deferred staging cleanup without replacing the original capability.
        original.retry_close()
        native.acquire_install_lease(str(candidate), "duckdb").close()
        assert candidate.exists()

        def refuse_workspace(path, *args, **kwargs):
            if Path(path) == workspace:
                raise PermissionError("raw-cleanup-detail?token=REVIEW_SECRET")
            return real_rmtree(path, *args, **kwargs)

        with monkeypatch.context() as fault:
            fault.setattr(shutil, "rmtree", refuse_workspace)
            for _ in range(2):
                with pytest.raises(hub_errors.HubUnavailable) as cleanup_error:
                    failure.retry_close()
                assert str(cleanup_error.value) == (
                    "Hub staging workspace could not be removed"
                )
                assert cleanup_error.value.__suppress_context__
                assert candidate.exists()
                assert failure.reader_error is original
                assert failure.ownership is original.ownership
        failure.retry_close()
        failure.retry_close()
        assert not workspace.exists()
        assert not target.exists()
    finally:
        if failure is not None:
            failure.retry_close()


@pytest.mark.parametrize("boundary,automatic,cancel", [
    ("initial", True, False),
    ("recheck", True, False),
    ("candidate", True, False),
    ("candidate", False, False),
    ("initial", True, True),
    ("recheck", True, True),
])
async def test_pull_reader_failure_is_fatal_retains_owner_and_retries_safely(
    tmp_path, hub_http, monkeypatch, boundary, automatic, cancel
):
    config = config_for(tmp_path / "checkout", hub_http)
    target = config.database.get_db_path()
    before = []
    if boundary != "recheck":
        build_database(config.database.path, model="old-model")
        before.append(snapshot(target))
    competitor = build_database(tmp_path / "competitor", model="old-model")
    publish(hub_http, build_database(tmp_path / "published"))
    sidecar = target.with_name(target.name + ".root.json")
    sidecar.write_text("local root must survive")
    if boundary == "recheck":
        def install_race_winner(route):
            if route.startswith("/base.tar.gz"):
                lease = native.acquire_install_lease(str(target), "duckdb")
                try:
                    lease.install(str(competitor))
                finally:
                    lease.close()
                sidecar.write_text("local root must survive")
                before.append(snapshot(target))
        hub_http.before_get = install_race_winner

    def wants_path(path):
        if boundary == "candidate":
            return any(p.name.startswith(".chunkhound-pull-") for p in path.parents)
        return path.resolve() == target.resolve()

    entered, release = threading.Event(), threading.Event()
    messages = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        filter=lambda record: record["extra"].get("hub_client", False),
    )
    original = None
    task = None
    held_workspaces = []
    try:
        with reader_close_fault(
            monkeypatch, wants_path,
            entered=entered if cancel else None, release=release,
        ) as (readers, held_workspaces):
            task = asyncio.create_task(pull(config, automatic=automatic))
            try:
                if cancel:
                    assert await asyncio.to_thread(entered.wait, 20)
                    for _ in range(3):
                        task.cancel()
                        await asyncio.sleep(0)
                        assert not task.done()
                    assert not any("Cannot close hub" in text for text in messages)
                    with pytest.raises(native.DatabaseBusyError):
                        native.acquire_install_lease(str(target), "duckdb")
                    if boundary == "recheck":
                        assert list(target.parent.glob(".chunkhound-pull-*"))
            finally:
                release.set()
            try:
                await task
            except BaseException as error:
                failure = error
                original = retained_reader_error(error)
            else:
                raise AssertionError("Pull unexpectedly returned after close failure")
            # Run real pull before looking up the new type, so baseline RED is
            # a runtime contract failure rather than an import/collection error.
            assert isinstance(failure, hub_errors.HubReaderCloseError)
            assert failure.reader_error is original
            assert failure.ownership is original.ownership
            held_target = readers[0][1]
            original.ownership.assert_covers(held_target)
            assert readers[0][0]().execute("SELECT 42").fetchone() == (42,)
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(held_target), "duckdb")
            with pytest.raises(DatabaseReaderCloseError):
                failure.retry_close()
            if boundary == "recheck":
                assert not list(target.parent.glob(".chunkhound-pull-*"))
        # All three failing scopes have requested full unwind. Delegate to the
        # ORIGINAL error; constructing another error would lose that policy.
        failure.retry_close()
        failure.retry_close()
        if boundary == "candidate":
            assert not held_target.parent.exists()
        else:
            native.acquire_install_lease(str(held_target), "duckdb").close()
        assert not list(target.parent.glob(".chunkhound-pull-*"))
        assert snapshot(target) == before[0]
        assert sidecar.read_text() == "local root must survive"
        assert len(hub_http.requests) == (0 if boundary == "initial" else 2)
        diagnostic = str(failure)
        assert messages.count(diagnostic) == 1
        assert not any("identity is unreadable" in text for text in messages)
        assert not any("continuing local indexing" in text for text in messages)
        assert "REVIEW_SECRET" not in "\n".join(messages)
        assert "raw-close-detail" not in diagnostic
    finally:
        release.set()
        if task is not None:
            results = await asyncio.gather(task, return_exceptions=True)
            if original is None and isinstance(results[0], BaseException):
                original = retained_reader_error(results[0])
        if original is not None:
            original.retry_close()
        for workspace in held_workspaces:
            workspace.cleanup()
        logger.remove(sink)


async def test_unreadable_local_identity_is_soft_no_request_and_preserves_files(
    tmp_path, hub_http
):
    config = config_for(tmp_path / "checkout", hub_http)
    target = build_database(config.database.path, model="old-model")
    target.with_name(target.name + ".embedding.json").write_text("{")
    before = snapshot(target.parent)
    assert not await pull(config, automatic=True)
    assert hub_http.requests == []
    assert snapshot(target.parent) == before
    native.acquire_install_lease(str(target), "duckdb").close()


def test_local_reader_retry_does_not_release_borrowed_exclusive_owner(
    tmp_path, hub_http, monkeypatch
):
    config = config_for(tmp_path / "checkout", hub_http)
    target = build_database(config.database.path)
    owner = DatabaseOwnership(native.acquire_install_lease(str(target), "duckdb"))
    original = None
    try:
        with reader_close_fault(
            monkeypatch, lambda path: path.resolve() == target.resolve()
        ):
            with pytest.raises(DatabaseReaderCloseError) as raised:
                local_needs_pull(target, configured_key(config), ownership=owner)
            original = raised.value
            assert original.ownership is owner
        original.retry_close()
        owner.assert_covers(target)
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(target), "duckdb")
    finally:
        if original is not None:
            original.retry_close()
        # This test is itself the complete enclosing lifecycle, not a borrower.
        owner.close()
    native.acquire_install_lease(str(target), "duckdb").close()


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize(
    "limit",
    [
        "max_members",
        "max_path_depth",
        "max_path_bytes",
        "max_total_path_bytes",
        "max_extracted_bytes",
        "max_decompressed_bytes",
    ],
)
async def test_public_pull_honors_configured_archive_limits_and_preserves_state(
    tmp_path, hub_http, monkeypatch, automatic, limit
):
    base = config_for(tmp_path / "checkout", hub_http, "lancedb")
    target = build_database(base.database.path, "lancedb", model="old-model")
    sidecar = target.with_name(target.name + ".embedding.json")
    write_active_model(sidecar, ActiveModel("openai", "old-model", 8))
    neighbor = target.parent / "operator.txt"
    neighbor.write_text("keep")
    source = build_database(tmp_path / "published", "lancedb")
    manifest, route = publish(hub_http, source)
    # A real publisher tar.add archive, without the fixture's mode-zero filter.
    ordinary = tmp_path / "ordinary.tar.gz"
    with tarfile.open(ordinary, "w:gz") as tar:
        tar.add(source, arcname=source.name)
    body = ordinary.read_bytes()
    hub_http.resources[route] = body
    manifest["embeddings"][0]["versions"][0]["artifact"].update(
        bytes=len(body), sha256=hashlib.sha256(body).hexdigest()
    )
    hub_http.resources["/catalogue?token=secret"] = json.dumps(manifest).encode()
    monkeypatch.setenv(f"CHUNKHOUND_HUB__ARCHIVE_LIMITS__{limit.upper()}", "2")
    config_path = base.target_dir / ".chunkhound.json"
    config_path.write_text(json.dumps({"hub": {"archive_limits": {limit: 1}}}))
    config = Config(
        target_dir=base.target_dir,
        database=base.database,
        embedding=base.embedding,
        hub={"url": base.hub.url},
        skip_layers={"global", "config_file", "cli"},
    )
    assert getattr(config.hub.archive_limits, limit) == 1
    before = snapshot(target.parent)
    assert not await pull(config, automatic=automatic)
    assert snapshot(target.parent) == before
    assert not list(target.parent.glob(".chunkhound-pull-*"))
    assert len(hub_http.requests) == 2
    # Raising the client limits permits this same ordinary published artifact.
    config.hub.archive_limits = ArchiveLimits()
    assert await pull(config, automatic=automatic)
    assert inspect_database(target, "lancedb").groups
    assert not sidecar.exists()
    assert neighbor.read_text() == "keep"
    assert not list(target.parent.glob(".chunkhound-pull-*"))


async def test_quota_refusal_preserves_interrupted_install_and_unknown_wal(
    tmp_path, hub_http
):
    config = config_for(tmp_path / "checkout", hub_http)
    target = build_database(config.database.path)
    files = [
        target.with_name(target.name + suffix)
        for suffix in [
            ".wal",
            ".root.json",
            ".embedding.json",
            ".chunkhound-install.json",
        ]
    ]
    for path in files:
        path.write_bytes(b"original recovery state")
    publish(hub_http, build_database(tmp_path / "published"))
    before = snapshot(target.parent)
    config.hub.archive_limits = ArchiveLimits(max_extracted_bytes=1)
    # Explicit pull downloads first: quota refusal must never enter recovery.
    assert not await pull(config)
    assert snapshot(target.parent) == before
    assert not list(target.parent.glob(".chunkhound-pull-*"))
