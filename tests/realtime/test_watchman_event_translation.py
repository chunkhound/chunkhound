from __future__ import annotations

import asyncio
import shutil
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

import chunkhound.services.realtime_indexing_service as realtime_service_module
from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.services.realtime_indexing_service import RealtimeIndexingService
from chunkhound.watchman import WatchmanScopePlan, WatchmanSubscriptionScope
from tests.helpers.watchman_realtime import (
    build_watchman_service as _build_watchman_service,
)
from tests.helpers.watchman_realtime import (
    start_isolated_watchman_translation as _start_isolated_watchman_translation,
)
from tests.helpers.watchman_realtime import (
    wait_for_logical_indexed as _wait_for_logical_indexed,
)
from tests.helpers.watchman_realtime import (
    wait_for_removed as _wait_for_removed,
)
from tests.utils.windows_compat import wait_for_indexed

pytestmark = pytest.mark.requires_native_watchman


def _subscription_pdu(
    *,
    name: str,
    exists: bool,
    is_new: bool,
    file_type: str = "f",
) -> dict[str, object]:
    return {
        "subscription": "chunkhound-live-indexing",
        "clock": "c:0:1",
        "files": [
            {
                "name": name,
                "exists": exists,
                "new": is_new,
                "type": file_type,
            }
        ],
    }

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


async def _wait_for_realtime_idle(
    service: RealtimeIndexingService,
) -> dict[str, object]:
    deadline = asyncio.get_running_loop().time() + 5.0
    while asyncio.get_running_loop().time() < deadline:
        if (
            service._active_processing_count == 0
            and service.event_queue.qsize() == 0
            and service.file_queue.qsize() == 0
            and not service.pending_files
        ):
            return await service.get_health()
        await asyncio.sleep(0.05)
    raise AssertionError("Timed out waiting for realtime service to become idle")


@pytest.mark.asyncio
async def test_watchman_mount_aware_startup_reuses_primary_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_dir = tmp_path / "workspace_root"
    nested_mount = (target_dir / "chunkhound_workspace").resolve()
    target_dir.mkdir(parents=True)
    nested_mount.mkdir(parents=True)
    service, services = _build_watchman_service(target_dir)
    debug_messages: list[str] = []
    service._debug_sink = debug_messages.append
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

    class TrackingSession:
        _SUBSCRIPTION_QUEUE_MAXSIZE = 1000
        _created_sessions: list[TrackingSession] = []
        _capabilities = {
            "cmd-watch-project": True,
            "relative_root": True,
        }

        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.subscription_queue = asyncio.Queue(
                maxsize=self._SUBSCRIPTION_QUEUE_MAXSIZE
            )
            self.session_id = len(self._created_sessions)
            self._created_sessions.append(self)
            self._subscription_name: str | None = None
            self._subscription_names: tuple[str, ...] = ()
            self._subscription_scopes: dict[str, WatchmanSubscriptionScope] = {}
            self._scope_plan = None

        @staticmethod
        def _sanitize_subscription_suffix(value: str) -> str:
            candidate = value.replace("\\", "/").strip("/")
            if not candidate:
                return ""
            parts: list[str] = []
            for chunk in PurePosixPath(candidate).parts:
                normalized_chunk = "".join(
                    character.lower() if character.isalnum() else "-"
                    for character in chunk
                ).strip("-")
                if normalized_chunk:
                    parts.append(normalized_chunk)
            return "-".join(parts)

        def supports_prepared_session_startup(self) -> bool:
            return True

        async def prepare(self) -> dict[str, bool]:
            prepare_calls.append(self.session_id)
            await asyncio.sleep(prepare_delay)
            return dict(self._capabilities)

        async def watch_project(self, target_path: Path) -> dict[str, object]:
            operations.append(
                ("watch_project", self.session_id, str(target_path.resolve()))
            )
            return {"watch": str(target_path.resolve()), "relative_path": None}

        async def watch_roots(self, roots: list[Path]) -> tuple[Path, ...]:
            operations.append(
                (
                    "watch_roots",
                    self.session_id,
                    tuple(str(root) for root in roots),
                )
            )
            return tuple(roots)

        async def startup_watch_project_once(
            self, target_path: Path
        ) -> dict[str, object]:
            raise AssertionError(
                "unexpected one-shot watch-project during prepared startup: "
                f"{target_path}"
            )

        async def startup_watch_roots_once(
            self, roots: list[Path]
        ) -> tuple[Path, ...]:
            raise AssertionError(
                f"unexpected one-shot watch during prepared startup: {roots!r}"
            )

        async def subscribe_scopes(
            self,
            *,
            target_path: Path,
            scope_plan: object,
            subscription_name: str | None = None,
        ) -> SimpleNamespace:
            del target_path
            resolved_subscription_name = subscription_name or "chunkhound-live-indexing"
            operations.append(
                (
                    "subscribe_scopes",
                    self.session_id,
                    resolved_subscription_name,
                    tuple(item.scope_kind for item in scope_plan.scopes),
                )
            )
            self._scope_plan = scope_plan
            self._subscription_name = resolved_subscription_name
            self._subscription_names = (resolved_subscription_name,)
            self._subscription_scopes = {
                resolved_subscription_name: scope_plan.primary_scope
            }
            return SimpleNamespace(
                scope_plan=scope_plan,
                subscription_name=resolved_subscription_name,
                subscription_names=self._subscription_names,
                capabilities=dict(self._capabilities),
            )

        async def start(
            self,
            *,
            target_path: Path,
            subscription_name: str | None = None,
            scope_plan=None,
            nested_mount_roots=(),
            additional_scopes=(),
        ) -> SimpleNamespace:
            del nested_mount_roots, additional_scopes
            await self.prepare()
            return await self.subscribe_scopes(
                target_path=target_path,
                scope_plan=scope_plan,
                subscription_name=subscription_name,
            )

        async def stop(self) -> None:
            return None

        async def wait_for_unexpected_exit(self) -> str | None:
            await asyncio.Event().wait()
            return None

        def get_health(self) -> dict[str, object]:
            watchman_scopes = []
            for subscription_name in self._subscription_names:
                scope = self._subscription_scopes.get(subscription_name)
                if scope is None:
                    continue
                watchman_scopes.append(
                    {
                        "subscription_name": subscription_name,
                        "scope_kind": scope.scope_kind,
                        "requested_path": str(scope.requested_path),
                        "watch_root": str(scope.watch_root),
                        "relative_root": scope.relative_root,
                    }
                )
            primary_scope = self._scope_plan.primary_scope if self._scope_plan else None
            return {
                "watchman_session_alive": True,
                "watchman_session_pid": None,
                "watchman_session_last_warning": None,
                "watchman_session_last_warning_at": None,
                "watchman_session_last_error": None,
                "watchman_session_last_error_at": None,
                "watchman_session_last_response_at": None,
                "watchman_subscription_last_received_at": None,
                "watchman_session_command_count": len(operations),
                "watchman_subscription_queue_size": self.subscription_queue.qsize(),
                "watchman_subscription_queue_maxsize": self.subscription_queue.maxsize,
                "watchman_subscription_pdu_count": 0,
                "watchman_subscription_pdu_dropped": 0,
                "watchman_subscription_name": self._subscription_name,
                "watchman_subscription_names": list(self._subscription_names),
                "watchman_watch_root": (
                    str(primary_scope.watch_root) if primary_scope else None
                ),
                "watchman_relative_root": (
                    primary_scope.relative_root if primary_scope else None
                ),
                "watchman_scopes": watchman_scopes,
                "watchman_session_capabilities": dict(self._capabilities),
            }

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
        assert any(
            "RT: watchman scope discovery: linux nested mounts count=1" in message
            and str(nested_mount) in message
            for message in debug_messages
        )
        assert any(
            "RT: watchman scope discovery: windows junction scopes count=0"
            in message
            for message in debug_messages
        )
        assert any(
            "RT: watchman scope discovery: watch roots mode=prepared_session "
            "count=1" in message
            and str(nested_mount) in message
            for message in debug_messages
        )
        assert any(
            "RT: watchman scope discovery: scope plan built count=2" in message
            and "['primary', 'nested_mount']" in message
            for message in debug_messages
        )
        assert any(
            "RT: watchman scope discovery: phase total duration=" in message
            for message in debug_messages
        )
    finally:
        await adapter.stop()
        services.provider.disconnect()


@pytest.mark.asyncio
async def test_watchman_start_preserves_logical_root_for_junction_scope_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logical_target = tmp_path / "logical_workspace"
    logical_junction = logical_target / "linked_workspace"
    physical_target = tmp_path / "physical_workspace"
    physical_junction_root = tmp_path / "physical_linked_workspace"
    for path in (
        logical_target,
        logical_junction,
        physical_target,
        physical_junction_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    service, services = _build_watchman_service(logical_target)
    discovery_calls: list[Path] = []
    operations: list[tuple[object, ...]] = []
    original_resolve = realtime_service_module.Path.resolve

    def fake_resolve(self: Path, strict: bool = False) -> Path:
        if self == logical_target:
            return physical_target
        if self == logical_junction:
            return physical_junction_root
        return original_resolve(self, strict=strict)

    class FakeSidecar:
        def __init__(self) -> None:
            self.paths = SimpleNamespace(
                listener_path=str(tmp_path / "watchman.sock"),
                statefile_path=tmp_path / "watchman.state",
                logfile_path=tmp_path / "watchman.log",
                pidfile_path=tmp_path / "watchman.pid",
                project_root=logical_target,
            )

        async def start(self) -> SimpleNamespace:
            return SimpleNamespace(binary_path=str(tmp_path / "watchman"))

        async def stop(self) -> None:
            operations.append(("sidecar_stop",))

        def get_health(self) -> dict[str, object]:
            return {"watchman_alive": True}

    class TrackingSession:
        _SUBSCRIPTION_QUEUE_MAXSIZE = 1000
        _capabilities = {"cmd-watch-project": True, "relative_root": True}
        _created_sessions: list[TrackingSession] = []

        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.subscription_queue = asyncio.Queue(
                maxsize=self._SUBSCRIPTION_QUEUE_MAXSIZE
            )
            self.session_id = len(self._created_sessions)
            self._created_sessions.append(self)
            self._subscription_name: str | None = None
            self._subscription_names: tuple[str, ...] = ()
            self._subscription_scopes: dict[str, WatchmanSubscriptionScope] = {}
            self._scope_plan = None

        def supports_prepared_session_startup(self) -> bool:
            return True

        async def prepare(self) -> dict[str, bool]:
            return dict(self._capabilities)

        async def watch_project(self, target_path: Path) -> dict[str, object]:
            operations.append(("watch_project", self.session_id, str(target_path)))
            return {"watch": str(physical_target), "relative_path": None}

        async def watch_roots(self, roots: list[Path]) -> tuple[Path, ...]:
            operations.append(
                ("watch_roots", self.session_id, tuple(str(root) for root in roots))
            )
            return tuple(roots)

        async def startup_watch_project_once(
            self, target_path: Path
        ) -> dict[str, object]:
            raise AssertionError(
                "unexpected one-shot watch-project during prepared startup: "
                f"{target_path}"
            )

        async def startup_watch_roots_once(
            self, roots: list[Path]
        ) -> tuple[Path, ...]:
            raise AssertionError(
                f"unexpected one-shot watch during prepared startup: {roots!r}"
            )

        async def subscribe_scopes(
            self,
            *,
            target_path: Path,
            scope_plan: object,
            subscription_name: str | None = None,
        ) -> SimpleNamespace:
            del target_path
            resolved_subscription_name = subscription_name or "chunkhound-live-indexing"
            self._scope_plan = scope_plan
            self._subscription_name = resolved_subscription_name
            self._subscription_names = (resolved_subscription_name,)
            self._subscription_scopes = {
                resolved_subscription_name: scope_plan.primary_scope
            }
            operations.append(
                (
                    "subscribe_scopes",
                    self.session_id,
                    resolved_subscription_name,
                    tuple(item.scope_kind for item in scope_plan.scopes),
                )
            )
            return SimpleNamespace(
                scope_plan=scope_plan,
                subscription_name=resolved_subscription_name,
                subscription_names=self._subscription_names,
                capabilities=dict(self._capabilities),
            )

        async def start(
            self,
            *,
            target_path: Path,
            subscription_name: str | None = None,
            scope_plan=None,
            nested_mount_roots=(),
            additional_scopes=(),
        ) -> SimpleNamespace:
            del nested_mount_roots, additional_scopes
            await self.prepare()
            operations.append(("start", self.session_id, str(target_path)))
            return await self.subscribe_scopes(
                target_path=target_path,
                scope_plan=scope_plan,
                subscription_name=subscription_name,
            )

        async def stop(self) -> None:
            return None

        async def wait_for_unexpected_exit(self) -> str | None:
            await asyncio.Event().wait()
            return None

        def get_health(self) -> dict[str, object]:
            watchman_scopes = []
            for subscription_name in self._subscription_names:
                scope = self._subscription_scopes.get(subscription_name)
                if scope is None:
                    continue
                watchman_scopes.append(
                    {
                        "subscription_name": subscription_name,
                        "scope_kind": scope.scope_kind,
                        "requested_path": str(scope.requested_path),
                        "watch_root": str(scope.watch_root),
                        "relative_root": scope.relative_root,
                    }
                )
            primary_scope = self._scope_plan.primary_scope if self._scope_plan else None
            return {
                "watchman_session_alive": True,
                "watchman_session_pid": None,
                "watchman_session_last_warning": None,
                "watchman_session_last_warning_at": None,
                "watchman_session_last_error": None,
                "watchman_session_last_error_at": None,
                "watchman_session_last_response_at": None,
                "watchman_subscription_last_received_at": None,
                "watchman_session_command_count": len(operations),
                "watchman_subscription_queue_size": self.subscription_queue.qsize(),
                "watchman_subscription_queue_maxsize": self.subscription_queue.maxsize,
                "watchman_subscription_pdu_count": 0,
                "watchman_subscription_pdu_dropped": 0,
                "watchman_subscription_name": self._subscription_name,
                "watchman_subscription_names": list(self._subscription_names),
                "watchman_watch_root": (
                    str(primary_scope.watch_root) if primary_scope else None
                ),
                "watchman_relative_root": (
                    primary_scope.relative_root if primary_scope else None
                ),
                "watchman_scopes": watchman_scopes,
                "watchman_session_capabilities": dict(self._capabilities),
            }

    monkeypatch.setattr(realtime_service_module.Path, "resolve", fake_resolve)
    monkeypatch.setattr(
        realtime_service_module,
        "discover_nested_linux_mount_roots",
        lambda target_path: (),
    )

    def discover_junctions(
        target_path: Path,
    ) -> tuple[WatchmanSubscriptionScope, ...]:
        discovery_calls.append(target_path)
        return (
            WatchmanSubscriptionScope(
                requested_path=logical_junction.absolute(),
                watch_root=physical_junction_root,
                relative_root=None,
                scope_kind="nested_junction",
            ),
        )

    monkeypatch.setattr(
        realtime_service_module,
        "discover_nested_windows_junction_scopes",
        discover_junctions,
    )
    monkeypatch.setattr(
        realtime_service_module,
        "PrivateWatchmanSidecar",
        lambda *args, **kwargs: FakeSidecar(),
    )
    monkeypatch.setattr(
        realtime_service_module,
        "WatchmanCliSession",
        TrackingSession,
    )

    try:
        await service.start(logical_target)

        assert discovery_calls == [logical_target.absolute()]
        assert operations[0] == (
            "watch_project",
            0,
            str(logical_target.absolute()),
        )
        assert service.watch_path == logical_target.absolute()
        assert service.watchman_scope_plan is not None
        assert service.watchman_scope_plan.primary_scope == WatchmanSubscriptionScope(
            requested_path=logical_target.absolute(),
            watch_root=physical_target,
            relative_root=None,
            scope_kind="primary",
        )
        assert service.watchman_scope_plan.scopes[1] == WatchmanSubscriptionScope(
            requested_path=logical_junction.absolute(),
            watch_root=physical_junction_root,
            relative_root=None,
            scope_kind="nested_junction",
        )

        stats = await service.get_health()
        assert stats["watchman_scopes"] == [
            {
                "subscription_name": "chunkhound-live-indexing",
                "scope_kind": "primary",
                "requested_path": str(logical_target.absolute()),
                "watch_root": str(physical_target),
                "relative_root": None,
            },
            {
                "subscription_name": "chunkhound-live-indexing--linked-workspace",
                "scope_kind": "nested_junction",
                "requested_path": str(logical_junction.absolute()),
                "watch_root": str(physical_junction_root),
                "relative_root": None,
            },
        ]
    finally:
        await service.stop()
        services.provider.disconnect()


@pytest.mark.asyncio
async def test_watchman_mount_aware_startup_uses_fallback_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_dir = tmp_path / "workspace_root"
    nested_mount = (target_dir / "chunkhound_workspace").resolve()
    target_dir.mkdir(parents=True)
    nested_mount.mkdir(parents=True)
    service, services = _build_watchman_service(target_dir)
    debug_messages: list[str] = []
    service._debug_sink = debug_messages.append
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

    class TrackingSession:
        _SUBSCRIPTION_QUEUE_MAXSIZE = 1000
        _created_sessions: list[TrackingSession] = []
        _capabilities = {
            "cmd-watch-project": True,
            "relative_root": True,
        }

        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.subscription_queue = asyncio.Queue(
                maxsize=self._SUBSCRIPTION_QUEUE_MAXSIZE
            )
            self.session_id = len(self._created_sessions)
            self._created_sessions.append(self)
            self._subscription_name: str | None = None
            self._subscription_names: tuple[str, ...] = ()
            self._subscription_scopes: dict[str, WatchmanSubscriptionScope] = {}
            self._scope_plan = None

        @staticmethod
        def _sanitize_subscription_suffix(value: str) -> str:
            candidate = value.replace("\\", "/").strip("/")
            if not candidate:
                return ""
            parts: list[str] = []
            for chunk in PurePosixPath(candidate).parts:
                normalized_chunk = "".join(
                    character.lower() if character.isalnum() else "-"
                    for character in chunk
                ).strip("-")
                if normalized_chunk:
                    parts.append(normalized_chunk)
            return "-".join(parts)

        def supports_prepared_session_startup(self) -> bool:
            return False

        async def prepare(self) -> dict[str, bool]:
            prepare_calls.append(self.session_id)
            return dict(self._capabilities)

        async def watch_project(self, target_path: Path) -> dict[str, object]:
            raise AssertionError(
                "unexpected persistent watch-project during fallback startup: "
                f"{target_path}"
            )

        async def watch_roots(self, roots: list[Path]) -> tuple[Path, ...]:
            raise AssertionError(
                f"unexpected persistent watch during fallback startup: {roots!r}"
            )

        async def startup_watch_project_once(
            self, target_path: Path
        ) -> dict[str, object]:
            operations.append(
                (
                    "startup_watch_project_once",
                    self.session_id,
                    str(target_path.resolve()),
                )
            )
            return {"watch": str(target_path.resolve()), "relative_path": None}

        async def startup_watch_roots_once(
            self, roots: list[Path]
        ) -> tuple[Path, ...]:
            operations.append(
                (
                    "startup_watch_roots_once",
                    self.session_id,
                    tuple(str(root) for root in roots),
                )
            )
            return tuple(roots)

        async def subscribe_scopes(
            self,
            *,
            target_path: Path,
            scope_plan: object,
            subscription_name: str | None = None,
        ) -> SimpleNamespace:
            del target_path
            resolved_subscription_name = subscription_name or "chunkhound-live-indexing"
            self._scope_plan = scope_plan
            self._subscription_name = resolved_subscription_name
            self._subscription_names = (resolved_subscription_name,)
            self._subscription_scopes = {
                resolved_subscription_name: scope_plan.primary_scope
            }
            return SimpleNamespace(
                scope_plan=scope_plan,
                subscription_name=resolved_subscription_name,
                subscription_names=self._subscription_names,
                capabilities=dict(self._capabilities),
            )

        async def start(
            self,
            *,
            target_path: Path,
            subscription_name: str | None = None,
            scope_plan=None,
            nested_mount_roots=(),
            additional_scopes=(),
        ) -> SimpleNamespace:
            del nested_mount_roots, additional_scopes
            await self.prepare()
            resolved_subscription_name = subscription_name or "chunkhound-live-indexing"
            operations.append(
                (
                    "start",
                    self.session_id,
                    str(target_path.resolve()),
                    resolved_subscription_name,
                    tuple(item.scope_kind for item in scope_plan.scopes),
                )
            )
            return await self.subscribe_scopes(
                target_path=target_path,
                subscription_name=resolved_subscription_name,
                scope_plan=scope_plan,
            )

        async def stop(self) -> None:
            return None

        async def wait_for_unexpected_exit(self) -> str | None:
            await asyncio.Event().wait()
            return None

        def get_health(self) -> dict[str, object]:
            watchman_scopes = []
            for subscription_name in self._subscription_names:
                scope = self._subscription_scopes.get(subscription_name)
                if scope is None:
                    continue
                watchman_scopes.append(
                    {
                        "subscription_name": subscription_name,
                        "scope_kind": scope.scope_kind,
                        "requested_path": str(scope.requested_path),
                        "watch_root": str(scope.watch_root),
                        "relative_root": scope.relative_root,
                    }
                )
            primary_scope = self._scope_plan.primary_scope if self._scope_plan else None
            return {
                "watchman_session_alive": True,
                "watchman_session_pid": 54321,
                "watchman_session_last_warning": None,
                "watchman_session_last_warning_at": None,
                "watchman_session_last_error": None,
                "watchman_session_last_error_at": None,
                "watchman_session_last_response_at": None,
                "watchman_subscription_last_received_at": None,
                "watchman_session_command_count": len(operations),
                "watchman_subscription_queue_size": self.subscription_queue.qsize(),
                "watchman_subscription_queue_maxsize": self.subscription_queue.maxsize,
                "watchman_subscription_pdu_count": 0,
                "watchman_subscription_pdu_dropped": 0,
                "watchman_subscription_name": self._subscription_name,
                "watchman_subscription_names": list(self._subscription_names),
                "watchman_watch_root": (
                    str(primary_scope.watch_root) if primary_scope else None
                ),
                "watchman_relative_root": (
                    primary_scope.relative_root if primary_scope else None
                ),
                "watchman_scopes": watchman_scopes,
                "watchman_session_capabilities": dict(self._capabilities),
            }

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
    monkeypatch.setattr(
        realtime_service_module,
        "WatchmanCliSession",
        TrackingSession,
    )

    adapter = realtime_service_module.WatchmanRealtimeAdapter(service)
    adapter._sidecar = FakeSidecar()
    try:
        await adapter.start(target_dir, asyncio.get_running_loop())

        assert prepare_calls == [0, 1]
        assert len(TrackingSession._created_sessions) == 2
        assert operations == [
            ("startup_watch_project_once", 0, str(target_dir.resolve())),
            ("startup_watch_roots_once", 0, (str(nested_mount),)),
            (
                "start",
                0,
                str(target_dir.resolve()),
                "chunkhound-live-indexing",
                ("primary",),
            ),
            (
                "start",
                1,
                str(nested_mount),
                "chunkhound-live-indexing--chunkhound-workspace",
                ("nested_mount",),
            ),
        ]
        assert service.monitoring_ready.is_set()
        assert service.watchman_scope_plan is not None
        assert [scope.scope_kind for scope in service.watchman_scope_plan.scopes] == [
            "primary",
            "nested_mount",
        ]
        assert adapter._session is TrackingSession._created_sessions[0]
        assert adapter._sessions == TrackingSession._created_sessions
        assert any(
            "RT: watchman scope discovery: watch roots mode=one_shot count=1"
            in message
            and str(nested_mount) in message
            for message in debug_messages
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
async def test_watchman_subscription_pdu_deletes_indexed_directory(
    tmp_path: Path,
) -> None:
    watch_dir = tmp_path / "watchman_project"
    watch_dir.mkdir(parents=True)
    service, services = _build_watchman_service(watch_dir)

    try:
        await service.start(watch_dir)
        queue = service.watchman_subscription_queue
        assert queue is not None

        deleted_dir = watch_dir / "src" / "deleted_dir"
        first_file = deleted_dir / "first.py"
        second_file = deleted_dir / "second.py"
        deleted_dir.mkdir(parents=True, exist_ok=True)
        first_file.write_text("def first():\n    return 1\n", encoding="utf-8")
        second_file.write_text("def second():\n    return 2\n", encoding="utf-8")

        await service.add_file(first_file, priority="priority")
        await service.add_file(second_file, priority="priority")
        assert await _wait_for_logical_indexed(services.provider, first_file)
        assert await _wait_for_logical_indexed(services.provider, second_file)

        shutil.rmtree(deleted_dir)
        queue.put_nowait(
            _subscription_pdu(
                name="src/deleted_dir",
                exists=False,
                is_new=False,
                file_type="d",
            )
        )

        assert await _wait_for_removed(services.provider, first_file)
        assert await _wait_for_removed(services.provider, second_file)
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
        assert await _wait_for_logical_indexed(services.provider, file_path)

        baseline_stats = await _wait_for_realtime_idle(service)
        baseline_suppressed_duplicates = int(
            baseline_stats["pipeline"]["suppressed_duplicate_count"]
        )
        add_file_calls.clear()
        service.reset_file_tracking(file_path)

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

        assert await service.wait_for_file_indexed(file_path)
        stats = await _wait_for_realtime_idle(service)

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
        assert (
            int(stats["pipeline"]["suppressed_duplicate_count"])
            >= baseline_suppressed_duplicates + 1
        )
    finally:
        await service.stop()
        services.provider.disconnect()


@pytest.mark.asyncio
async def test_watchman_multi_scope_translation_routes_colliding_subscription_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_dir = tmp_path / "workspace_root"
    logical_a = target_dir / "foo+bar"
    logical_b = target_dir / "foo_bar"
    physical_a = tmp_path / "physical-a"
    physical_b = tmp_path / "physical-b"
    for path in (target_dir, logical_a, logical_b, physical_a, physical_b):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        realtime_service_module,
        "discover_nested_linux_mount_roots",
        lambda target_path: (),
    )
    monkeypatch.setattr(
        realtime_service_module,
        "discover_nested_windows_junction_scopes",
        lambda target_path: (
            WatchmanSubscriptionScope(
                requested_path=logical_a.resolve(),
                watch_root=physical_a.resolve(),
                relative_root=None,
                scope_kind="nested_junction",
            ),
            WatchmanSubscriptionScope(
                requested_path=logical_b.resolve(),
                watch_root=physical_b.resolve(),
                relative_root=None,
                scope_kind="nested_junction",
            ),
        ),
    )

    service, services = _build_watchman_service(target_dir)

    try:
        await service.start(target_dir)
        queue = service.watchman_subscription_queue
        assert queue is not None

        file_a = logical_a / "src" / "first.py"
        file_b = logical_b / "src" / "second.py"
        file_a.parent.mkdir(parents=True, exist_ok=True)
        file_b.parent.mkdir(parents=True, exist_ok=True)
        file_a.write_text("def first():\n    return 1\n", encoding="utf-8")
        file_b.write_text("def second():\n    return 2\n", encoding="utf-8")

        stats = await service.get_health()
        assert stats["watchman_subscription_count"] == 3
        assert len(stats["watchman_subscription_names"]) == 3
        assert len(set(stats["watchman_subscription_names"])) == 3

        scope_names = {
            scope["requested_path"]: scope["subscription_name"]
            for scope in stats["watchman_scopes"]
        }
        subscription_a = scope_names[str(logical_a.resolve())]
        subscription_b = scope_names[str(logical_b.resolve())]
        assert subscription_a != subscription_b
        assert subscription_a.startswith("chunkhound-live-indexing--foo-bar")
        assert subscription_b.startswith("chunkhound-live-indexing--foo-bar")

        queue.put_nowait(
            {
                "subscription": subscription_a,
                "clock": "c:0:7",
                "files": [
                    {
                        "name": "src/first.py",
                        "exists": True,
                        "new": True,
                        "type": "f",
                    }
                ],
            }
        )
        queue.put_nowait(
            {
                "subscription": subscription_b,
                "clock": "c:0:8",
                "files": [
                    {
                        "name": "src/second.py",
                        "exists": True,
                        "new": True,
                        "type": "f",
                    }
                ],
            }
        )

        assert await _wait_for_logical_indexed(services.provider, file_a)
        assert await _wait_for_logical_indexed(services.provider, file_b)
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
async def test_watchman_scope_mapping_failure_requests_resync(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "workspace_root"
    nested_mount = target_dir / "chunkhound_workspace"
    target_dir.mkdir(parents=True)
    nested_mount.mkdir(parents=True)
    service, services = _build_watchman_service(target_dir)
    callback_calls: list[tuple[str, dict[str, object] | None]] = []
    callback_event = asyncio.Event()

    async def resync_callback(
        reason: str, details: dict[str, object] | None
    ) -> None:
        callback_calls.append((reason, details))
        callback_event.set()

    primary_scope = WatchmanSubscriptionScope(
        requested_path=target_dir.resolve(),
        watch_root=target_dir.resolve(),
        relative_root=None,
        scope_kind="primary",
    )
    nested_scope = WatchmanSubscriptionScope(
        requested_path=nested_mount.resolve(),
        watch_root=nested_mount.resolve(),
        relative_root=None,
        scope_kind="nested_mount",
    )

    try:
        service._resync_callback = resync_callback
        adapter = await _start_isolated_watchman_translation(service, target_dir)
        service._monitor_adapter = adapter
        service.watchman_scope_plan = WatchmanScopePlan(
            scopes=(primary_scope, nested_scope)
        )
        adapter._subscription_scope_map = {
            "chunkhound-live-indexing": primary_scope,
            "chunkhound-live-indexing--chunkhound-workspace": nested_scope,
        }
        adapter._shared_subscription_queue = asyncio.Queue()
        adapter._subscription_consumer_task = asyncio.create_task(
            adapter._consume_subscription_pdus()
        )

        adapter._shared_subscription_queue.put_nowait(
            {
                "subscription": "chunkhound-live-indexing--unknown",
                "clock": "c:0:6",
                "files": [],
            }
        )

        await asyncio.wait_for(callback_event.wait(), timeout=5.0)
        stats = await _wait_for_pipeline_count(service, "translation_error_count", 1)

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "realtime_loss_of_sync"
        assert callback_calls[0][1] == {
            "backend": "watchman",
            "loss_of_sync_reason": "scope_mapping_failure",
            "subscription": "chunkhound-live-indexing--unknown",
            "translation_failure_kind": "unknown_scope",
            "translation_failure_message": (
                "Watchman subscription PDU did not map to a known scope"
            ),
            "translation_warning": (
                "Watchman event translation warning: "
                "Watchman subscription PDU did not map to a known scope"
            ),
            "clock": "c:0:6",
        }
        assert stats["pipeline"]["translation_error_count"] == 1
        assert stats["watchman_loss_of_sync"]["translation_failure_count"] == 1
        assert stats["watchman_loss_of_sync"]["last_reason"] == "scope_mapping_failure"
        assert stats["watchman_loss_of_sync"]["last_details"] == callback_calls[0][1]
        assert (
            "did not map to a known scope" in (stats["last_warning"] or "")
        )
    finally:
        await service.stop()
        services.provider.disconnect()


@pytest.mark.asyncio
async def test_watchman_translation_failure_requests_resync(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "workspace_root"
    target_dir.mkdir(parents=True)
    service, services = _build_watchman_service(target_dir)
    callback_calls: list[tuple[str, dict[str, object] | None]] = []
    callback_event = asyncio.Event()

    async def resync_callback(
        reason: str, details: dict[str, object] | None
    ) -> None:
        callback_calls.append((reason, details))
        callback_event.set()

    primary_scope = WatchmanSubscriptionScope(
        requested_path=target_dir.resolve(),
        watch_root=target_dir.resolve(),
        relative_root=None,
        scope_kind="primary",
    )

    try:
        service._resync_callback = resync_callback
        adapter = await _start_isolated_watchman_translation(service, target_dir)
        service._monitor_adapter = adapter
        service.watchman_scope_plan = WatchmanScopePlan(scopes=(primary_scope,))
        adapter._subscription_scope_map = {
            "chunkhound-live-indexing": primary_scope,
        }
        adapter._shared_subscription_queue = asyncio.Queue()
        adapter._subscription_consumer_task = asyncio.create_task(
            adapter._consume_subscription_pdus()
        )

        adapter._shared_subscription_queue.put_nowait(
            {
                "subscription": "chunkhound-live-indexing",
                "clock": "c:0:7",
                "files": [
                    {
                        "exists": True,
                        "new": True,
                        "type": "f",
                    }
                ],
            }
        )

        await asyncio.wait_for(callback_event.wait(), timeout=5.0)
        stats = await _wait_for_pipeline_count(service, "translation_error_count", 1)

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "realtime_loss_of_sync"
        assert callback_calls[0][1] == {
            "backend": "watchman",
            "loss_of_sync_reason": "translation_failure",
            "subscription": "chunkhound-live-indexing",
            "translation_failure_kind": "invalid_entry_name",
            "translation_failure_message": (
                "Skipping Watchman subscription entry without a valid name"
            ),
            "translation_warning": (
                "Watchman event translation warning: "
                "Skipping Watchman subscription entry without a valid name"
            ),
            "clock": "c:0:7",
        }
        assert stats["pipeline"]["translation_error_count"] == 1
        assert stats["watchman_loss_of_sync"]["translation_failure_count"] == 1
        assert stats["watchman_loss_of_sync"]["last_reason"] == "translation_failure"
        assert stats["watchman_loss_of_sync"]["last_details"] == callback_calls[0][1]
        assert (
            "without a valid name" in (stats["last_warning"] or "")
        )
    finally:
        await service.stop()
        services.provider.disconnect()


@pytest.mark.asyncio
async def test_watchman_start_preserves_logical_root_for_scope_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logical_root = tmp_path / "logical_workspace"
    logical_junction = logical_root / "linked_workspace"
    physical_root = tmp_path / "physical_workspace"
    physical_junction = tmp_path / "physical_linked_workspace"
    logical_root.mkdir(parents=True)
    logical_junction.mkdir(parents=True)
    physical_root.mkdir(parents=True)
    physical_junction.mkdir(parents=True)
    service, services = _build_watchman_service(logical_root)
    original_resolve = realtime_service_module.Path.resolve
    planned_paths: list[tuple[Path, Path, Path]] = []

    def fake_resolve(self: Path, strict: bool = False) -> Path:
        if self == logical_root:
            return physical_root
        if self == logical_junction:
            return physical_junction
        return original_resolve(self, strict=strict)

    class FakeWatchmanAdapter:
        backend_name = "watchman"

        async def start(
            self, target_path: Path, loop: asyncio.AbstractEventLoop
        ) -> None:
            del loop
            plan = realtime_service_module.build_watchman_scope_plan(
                target_path,
                {"watch": str(physical_root)},
                additional_scopes=(
                    WatchmanSubscriptionScope(
                        requested_path=logical_junction,
                        watch_root=physical_junction,
                        relative_root=None,
                        scope_kind="nested_junction",
                    ),
                ),
            )
            service.watchman_scope_plan = plan
            service._set_effective_backend("watchman")
            service.monitoring_ready.set()
            service._monitoring_ready_at = service._utc_now()
            planned_paths.append(
                (
                    plan.primary_scope.requested_path,
                    plan.primary_scope.watch_root,
                    plan.scopes[1].requested_path,
                )
            )

        async def stop(self) -> None:
            return None

        def get_health(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(realtime_service_module.Path, "resolve", fake_resolve)
    monkeypatch.setattr(
        service,
        "_build_monitor_adapter",
        lambda: FakeWatchmanAdapter(),
    )

    try:
        await service.start(logical_root)

        assert planned_paths == [
            (
                logical_root.absolute(),
                physical_root,
                logical_junction.absolute(),
            )
        ]
        assert service.watch_path == logical_root.absolute()
        assert service.watchman_scope_plan is not None
        assert service.watchman_scope_plan.primary_scope.requested_path == (
            logical_root.absolute()
        )
        assert service.watchman_scope_plan.scopes[1].requested_path == (
            logical_junction.absolute()
        )
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
        adapter._scope_path_filters[str(logical_junction)] = (
            realtime_service_module.RealtimePathFilter(
                config=service.config,
                root_path=logical_junction,
            )
        )

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


@pytest.mark.asyncio
async def test_watchman_junction_scope_filter_uses_logical_scope_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_dir = tmp_path / "workspace_root"
    logical_junction = target_dir / "linked_workspace"
    physical_root = tmp_path / "external_workspace"
    db_path = target_dir / ".chunkhound" / "test.db"
    target_dir.mkdir(parents=True)
    logical_junction.mkdir(parents=True)
    physical_root.mkdir(parents=True)
    config = Config(
        args=SimpleNamespace(path=target_dir),
        target_dir=target_dir,
        database={"path": str(db_path), "provider": "duckdb"},
        indexing={
            "include": ["**/*.py"],
            "exclude": ["generated/**"],
            "realtime_backend": "watchman",
        },
    )
    services = create_services(db_path, config)
    services.provider.connect()
    service = RealtimeIndexingService(services, config)
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
        adapter._scope_path_filters[str(logical_junction)] = (
            realtime_service_module.RealtimePathFilter(
                config=service.config,
                root_path=logical_junction,
            )
        )

        blocked_file = logical_junction / "generated" / "blocked.py"
        blocked_file.parent.mkdir(parents=True, exist_ok=True)
        blocked_file.write_text("def blocked():\n    return 1\n", encoding="utf-8")

        adapter._translate_subscription_pdu(
            {
                "subscription": "chunkhound-live-indexing--linked-workspace",
                "clock": "c:0:6",
                "files": [
                    {
                        "name": "generated/blocked.py",
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

        stats = await _wait_for_pipeline_count(service, "filtered_event_count", 1)
        assert service.event_queue.empty()
        assert stats["pipeline"]["last_source_event_path"] == str(blocked_file)
        assert stats["pipeline"]["last_accepted_event_path"] is None
    finally:
        await service.stop()
        services.provider.disconnect()
