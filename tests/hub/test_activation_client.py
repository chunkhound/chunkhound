import asyncio
import json
import shutil
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from loguru import logger

import chunkhound_native as native
from chunkhound.hub.client import pull
from chunkhound.hub.errors import HubDatabaseMismatch, HubInstallationInterrupted
from chunkhound.hub.inspect import inspect_database
from chunkhound.providers.database import ownership
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.providers.database.lancedb_provider import LanceDBProvider
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
