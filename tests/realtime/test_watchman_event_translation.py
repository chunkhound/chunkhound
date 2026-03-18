from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import chunkhound.services.realtime_indexing_service as realtime_service_module
from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.services.realtime_indexing_service import RealtimeIndexingService
from chunkhound.watchman import WatchmanCliSession, WatchmanSubscriptionScope
from tests.realtime.test_watchman_session_bridge import _write_fake_watchman_cli
from tests.utils.windows_compat import wait_for_indexed

pytestmark = pytest.mark.requires_native_watchman


def _build_watchman_service(target_dir: Path) -> tuple[RealtimeIndexingService, object]:
    db_path = target_dir / ".chunkhound" / "test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config = Config(
        args=SimpleNamespace(path=target_dir),
        database={"path": str(db_path), "provider": "duckdb"},
        indexing={"realtime_backend": "watchman"},
    )
    services = create_services(db_path, config)
    services.provider.connect()
    return RealtimeIndexingService(services, config), services


def _subscription_pdu(*, name: str, exists: bool, is_new: bool) -> dict[str, object]:
    return {
        "subscription": "chunkhound-live-indexing",
        "clock": "c:0:1",
        "files": [
            {
                "name": name,
                "exists": exists,
                "new": is_new,
                "type": "f",
            }
        ],
    }


async def _wait_for_removed(service_provider: object, file_path: Path) -> bool:
    deadline = asyncio.get_running_loop().time() + 5.0
    while asyncio.get_running_loop().time() < deadline:
        record = service_provider.get_file_by_path(str(file_path))
        if record is None:
            return True
        await asyncio.sleep(0.1)
    return False


async def _wait_for_logical_indexed(service_provider: object, file_path: Path) -> bool:
    deadline = asyncio.get_running_loop().time() + 5.0
    while asyncio.get_running_loop().time() < deadline:
        record = service_provider.get_file_by_path(str(file_path))
        if record is not None:
            return True
        await asyncio.sleep(0.1)
    return False


async def _wait_for_pipeline_count(
    service: RealtimeIndexingService, field: str, minimum: int
) -> dict[str, object]:
    deadline = asyncio.get_running_loop().time() + 5.0
    while asyncio.get_running_loop().time() < deadline:
        stats = await service.get_health()
        pipeline = stats.get("pipeline", {})
        if isinstance(pipeline, dict) and int(pipeline.get(field, 0)) >= minimum:
            return stats
        await asyncio.sleep(0.1)
    raise AssertionError(f"Timed out waiting for pipeline.{field} >= {minimum}")


async def _start_isolated_watchman_translation(
    service: RealtimeIndexingService, target_dir: Path
) -> realtime_service_module.WatchmanRealtimeAdapter:
    adapter = realtime_service_module.WatchmanRealtimeAdapter(service)
    adapter._path_filter = realtime_service_module.RealtimePathFilter(
        config=service.config,
        root_path=target_dir,
    )
    service.watch_path = target_dir
    service._service_state = "running"
    service._effective_backend = "watchman"
    service.monitoring_ready.set()
    service._monitoring_ready_at = service._utc_now()
    service.event_consumer_task = asyncio.create_task(service._consume_events())
    service.process_task = asyncio.create_task(service._process_loop())
    return adapter


@pytest.mark.asyncio
async def test_watchman_mount_aware_startup_reuses_primary_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = _write_fake_watchman_cli(tmp_path)
    target_dir = tmp_path / "workspace_root"
    nested_mount = (target_dir / "chunkhound_workspace").resolve()
    target_dir.mkdir(parents=True)
    nested_mount.mkdir(parents=True)
    socket_path = tmp_path / "watchman.sock"
    socket_path.write_text("socket ready\n", encoding="utf-8")
    statefile_path = tmp_path / "watchman.state"
    statefile_path.write_text("state ready\n", encoding="utf-8")
    logfile_path = tmp_path / "watchman.log"
    logfile_path.write_text("log ready\n", encoding="utf-8")
    pidfile_path = tmp_path / "watchman.pid"
    pidfile_path.write_text("123\n", encoding="utf-8")
    service, services = _build_watchman_service(target_dir)
    prepare_delay = 1.0
    prepare_calls: list[int] = []
    operations: list[tuple[object, ...]] = []

    class FakeSidecar:
        def __init__(self) -> None:
            self.paths = SimpleNamespace(
                listener_path=str(tmp_path / "watchman.sock"),
                statefile_path=tmp_path / "watchman.state",
                logfile_path=tmp_path / "watchman.log",
                pidfile_path=tmp_path / "watchman.pid",
                project_root=target_dir,
            )

        async def start(self) -> SimpleNamespace:
            return SimpleNamespace(binary_path=str(tmp_path / "watchman"))

        async def stop(self) -> None:
            operations.append(("sidecar_stop",))

        def get_health(self) -> dict[str, object]:
            return {"watchman_alive": True}

    class TrackingSession(WatchmanCliSession):
        _created_sessions: list[TrackingSession] = []

        def __init__(self, **kwargs: object) -> None:
            kwargs.setdefault("command_prefix", [sys.executable, str(script_path)])
            super().__init__(**kwargs)
            self.session_id = len(self._created_sessions)
            self._created_sessions.append(self)

        async def prepare(self) -> dict[str, bool]:
            prepare_calls.append(self.session_id)
            await asyncio.sleep(prepare_delay)
            return await super().prepare()

        async def watch_project(self, target_path: Path) -> dict[str, object]:
            operations.append(
                ("watch_project", self.session_id, str(target_path.resolve()))
            )
            return await super().watch_project(target_path)

        async def watch_roots(self, roots: list[Path]) -> tuple[Path, ...]:
            operations.append(
                (
                    "watch_roots",
                    self.session_id,
                    tuple(str(root) for root in roots),
                )
            )
            return await super().watch_roots(roots)

        async def subscribe_scopes(
            self,
            *,
            target_path: Path,
            scope_plan: object,
            subscription_name: str | None = None,
        ) -> object:
            resolved_subscription_name = subscription_name or "chunkhound-live-indexing"
            operations.append(
                (
                    "subscribe_scopes",
                    self.session_id,
                    resolved_subscription_name,
                    tuple(item.scope_kind for item in scope_plan.scopes),
                )
            )
            return await super().subscribe_scopes(
                target_path=target_path,
                scope_plan=scope_plan,
                subscription_name=subscription_name,
            )

        async def _run_one_shot_command(
            self, command: list[object]
        ) -> dict[str, object]:
            if command[0] != "version":
                raise AssertionError(
                    f"unexpected one-shot command during startup: {command[0]!r}"
                )
            return await super()._run_one_shot_command(command)

    monkeypatch.setattr(
        realtime_service_module,
        "discover_nested_linux_mount_roots",
        lambda target_path: (nested_mount,),
    )
    monkeypatch.setattr(
        realtime_service_module,
        "discover_nested_windows_junction_scopes",
        lambda target_path: (),
    )
    monkeypatch.setenv(
        "CHUNKHOUND_TEST_WATCHMAN_WATCH_ROOT",
        str(target_dir.resolve()),
    )
    monkeypatch.setattr(
        realtime_service_module,
        "WatchmanCliSession",
        TrackingSession,
    )

    adapter = realtime_service_module.WatchmanRealtimeAdapter(service)
    adapter._sidecar = FakeSidecar()
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    try:
        await adapter.start(target_dir, loop)
        elapsed = loop.time() - started_at

        assert prepare_calls == [0, 1]
        assert len(TrackingSession._created_sessions) == 2
        assert operations == [
            ("watch_project", 0, str(target_dir.resolve())),
            ("watch_roots", 0, (str(nested_mount),)),
            ("subscribe_scopes", 0, "chunkhound-live-indexing", ("primary",)),
            (
                "subscribe_scopes",
                1,
                "chunkhound-live-indexing--chunkhound-workspace",
                ("nested_mount",),
            ),
        ]
        assert elapsed < 3.7
        assert service.monitoring_ready.is_set()
        assert service.watchman_scope_plan is not None
        assert [scope.scope_kind for scope in service.watchman_scope_plan.scopes] == [
            "primary",
            "nested_mount",
        ]
        assert adapter._session is TrackingSession._created_sessions[0]
        assert adapter._sessions == TrackingSession._created_sessions

        startup_snapshot = service._startup_tracker.snapshot()
        assert set(startup_snapshot["phases"]) == {
            "initialize",
            "db_connect",
            "realtime_start",
            "startup_barrier",
            "daemon_publish",
            "watchman_sidecar_start",
            "watchman_watch_project",
            "watchman_scope_discovery",
            "watchman_subscription_setup",
            "watchdog_setup",
            "polling_setup",
        }
        assert (
            startup_snapshot["phases"]["watchman_watch_project"]["state"] == "completed"
        )
        assert (
            startup_snapshot["phases"]["watchman_scope_discovery"]["state"]
            == "completed"
        )
        assert (
            startup_snapshot["phases"]["watchman_subscription_setup"]["state"]
            == "completed"
        )
    finally:
        await adapter.stop()
        services.provider.disconnect()


@pytest.mark.asyncio
async def test_watchman_subscription_pdu_indexes_created_file(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watchman_project"
    watch_dir.mkdir(parents=True)
    service, services = _build_watchman_service(watch_dir)

    try:
        await service.start(watch_dir)
        queue = service.watchman_subscription_queue
        assert queue is not None

        file_path = watch_dir / "src" / "created.py"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("def created():\n    return 1\n", encoding="utf-8")

        queue.put_nowait(
            _subscription_pdu(name="src/created.py", exists=True, is_new=True)
        )

        assert await _wait_for_logical_indexed(services.provider, file_path)
    finally:
        await service.stop()
        services.provider.disconnect()


@pytest.mark.asyncio
async def test_watchman_subscription_pdu_deletes_indexed_file(tmp_path: Path) -> None:
    watch_dir = tmp_path / "watchman_project"
    watch_dir.mkdir(parents=True)
    service, services = _build_watchman_service(watch_dir)

    try:
        await service.start(watch_dir)
        queue = service.watchman_subscription_queue
        assert queue is not None

        file_path = watch_dir / "src" / "deleted.py"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("def deleted():\n    return 1\n", encoding="utf-8")

        await service.add_file(file_path, priority="priority")
        assert await _wait_for_logical_indexed(services.provider, file_path)

        file_path.unlink()
        queue.put_nowait(
            _subscription_pdu(name="src/deleted.py", exists=False, is_new=False)
        )

        assert await _wait_for_removed(services.provider, file_path)
    finally:
        await service.stop()
        services.provider.disconnect()


@pytest.mark.asyncio
async def test_watchman_relative_root_mapping_and_filtering(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "w"
    target_dir = workspace / "p" / "a"
    target_dir.mkdir(parents=True)

    service, services = _build_watchman_service(target_dir)

    try:
        adapter = await _start_isolated_watchman_translation(service, target_dir)

        included = target_dir / "src" / "mapped.py"
        included.parent.mkdir(parents=True, exist_ok=True)
        included.write_text("def mapped():\n    return 1\n", encoding="utf-8")

        excluded = target_dir / "src" / "ignored.xyz"
        excluded.write_text("ignored\n", encoding="utf-8")

        # Exercise translation directly so the event-count assertion stays tied
        # to the per-entry filtering contract rather than to extra sidecar noise.
        adapter._translate_subscription_pdu(
            {
                "subscription": "chunkhound-live-indexing",
                "clock": "c:0:2",
                "files": [
                    {
                        "name": "src/mapped.py",
                        "exists": True,
                        "new": True,
                        "type": "f",
                    },
                    {
                        "name": "src/ignored.xyz",
                        "exists": True,
                        "new": True,
                        "type": "f",
                    },
                ],
            },
            WatchmanSubscriptionScope(
                requested_path=target_dir,
                watch_root=workspace.resolve(),
                relative_root="p/a",
                scope_kind="primary",
            ),
        )

        assert await wait_for_indexed(services.provider, included)
        assert services.provider.get_file_by_path(str(excluded)) is None
        stats = await _wait_for_pipeline_count(service, "filtered_event_count", 1)
        assert stats["pipeline"]["filtered_event_count"] == 1
        assert stats["pipeline"]["last_source_event_path"] == str(excluded)
        assert stats["pipeline"]["last_accepted_event_path"] == str(included)
    finally:
        await service.stop()
        services.provider.disconnect()


@pytest.mark.asyncio
async def test_watchman_multi_scope_translation_deduplicates_across_subscriptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_dir = tmp_path / "workspace_root"
    nested_mount = target_dir / "chunkhound_workspace"
    target_dir.mkdir(parents=True)
    nested_mount.mkdir(parents=True)
    monkeypatch.setattr(
        realtime_service_module,
        "discover_nested_linux_mount_roots",
        lambda target_path: (nested_mount.resolve(),),
    )

    service, services = _build_watchman_service(target_dir)
    add_file_calls: list[tuple[Path, str]] = []

    original_add_file = service.add_file

    async def counting_add_file(file_path: Path, priority: str = "change") -> bool:
        add_file_calls.append((file_path, priority))
        return await original_add_file(file_path, priority)

    monkeypatch.setattr(service, "add_file", counting_add_file)

    try:
        await service.start(target_dir)
        queue = service.watchman_subscription_queue
        assert queue is not None

        file_path = nested_mount / "src" / "mounted.py"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("def mounted():\n    return 1\n", encoding="utf-8")

        queue.put_nowait(
            {
                "subscription": "chunkhound-live-indexing",
                "clock": "c:0:3",
                "files": [
                    {
                        "name": "chunkhound_workspace/src/mounted.py",
                        "exists": True,
                        "new": True,
                        "type": "f",
                    }
                ],
            }
        )
        queue.put_nowait(
            {
                "subscription": "chunkhound-live-indexing--chunkhound-workspace",
                "clock": "c:0:4",
                "files": [
                    {
                        "name": "src/mounted.py",
                        "exists": True,
                        "new": True,
                        "type": "f",
                    }
                ],
            }
        )

        assert await _wait_for_logical_indexed(services.provider, file_path)

        stats = await service.get_health()
        assert stats["watchman_subscription_count"] == 2
        assert stats["watchman_subscription_names"] == [
            "chunkhound-live-indexing",
            "chunkhound-live-indexing--chunkhound-workspace",
        ]
        assert stats["watchman_scopes"] == [
            {
                "subscription_name": "chunkhound-live-indexing",
                "scope_kind": "primary",
                "requested_path": str(target_dir.resolve()),
                "watch_root": str(target_dir.resolve()),
                "relative_root": None,
            },
            {
                "subscription_name": "chunkhound-live-indexing--chunkhound-workspace",
                "scope_kind": "nested_mount",
                "requested_path": str(nested_mount.resolve()),
                "watch_root": str(nested_mount.resolve()),
                "relative_root": None,
            },
        ]
        assert [
            (queued_path, priority)
            for queued_path, priority in add_file_calls
            if priority == "change"
        ] == [(file_path.resolve(), "change")]
        assert stats["pipeline"]["suppressed_duplicate_count"] >= 1
    finally:
        await service.stop()
        services.provider.disconnect()


@pytest.mark.asyncio
async def test_watchman_translation_errors_increment_pipeline_counter(
    tmp_path: Path,
) -> None:
    watch_dir = tmp_path / "watchman_project"
    watch_dir.mkdir(parents=True)
    service, services = _build_watchman_service(watch_dir)

    try:
        await service.start(watch_dir)
        queue = service.watchman_subscription_queue
        assert queue is not None

        queue.put_nowait(
            {
                "subscription": "chunkhound-live-indexing",
                "clock": "c:0:6",
                "files": [
                    {
                        "exists": True,
                        "new": True,
                        "type": "f",
                    }
                ],
            }
        )

        stats = await _wait_for_pipeline_count(service, "translation_error_count", 1)
        assert stats["pipeline"]["translation_error_count"] == 1
        assert "translation warning" in (stats["last_warning"] or "")
    finally:
        await service.stop()
        services.provider.disconnect()


@pytest.mark.asyncio
async def test_watchman_junction_scope_translation_preserves_logical_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_dir = tmp_path / "workspace_root"
    logical_junction = target_dir / "linked_workspace"
    physical_root = tmp_path / "external_workspace"
    target_dir.mkdir(parents=True)
    logical_junction.mkdir(parents=True)
    physical_root.mkdir(parents=True)

    service, services = _build_watchman_service(target_dir)
    original_resolve = realtime_service_module.Path.resolve

    def fake_resolve(self: Path, strict: bool = False) -> Path:
        if self == logical_junction:
            return physical_root
        try:
            relative_to_junction = self.relative_to(logical_junction)
        except ValueError:
            return original_resolve(self, strict=strict)
        return physical_root / relative_to_junction

    try:
        monkeypatch.setattr(realtime_service_module.Path, "resolve", fake_resolve)
        adapter = await _start_isolated_watchman_translation(service, target_dir)
        adapter._path_filter = SimpleNamespace(should_index=lambda _path: True)

        file_path = logical_junction / "src" / "junctioned.py"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("def junctioned():\n    return 1\n", encoding="utf-8")

        assert service._build_mutation("change", file_path).path == file_path

        adapter._translate_subscription_pdu(
            {
                "subscription": "chunkhound-live-indexing--linked-workspace",
                "clock": "c:0:5",
                "files": [
                    {
                        "name": "src/junctioned.py",
                        "exists": True,
                        "new": True,
                        "type": "f",
                    }
                ],
            },
            WatchmanSubscriptionScope(
                requested_path=logical_junction,
                watch_root=physical_root.resolve(),
                relative_root=None,
                scope_kind="nested_junction",
            ),
        )

        assert await _wait_for_logical_indexed(services.provider, file_path)

        stats = await service.get_health()
        assert stats["pipeline"]["last_accepted_event_path"] == str(file_path)
    finally:
        await service.stop()
        services.provider.disconnect()
