from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.services.realtime_indexing_service import RealtimeIndexingService

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


async def _wait_for_watchman_reconnect_state(
    service: RealtimeIndexingService,
    expected_state: str,
    *,
    timeout: float = 10.0,
) -> dict[str, object]:
    async def _poll() -> dict[str, object]:
        while True:
            stats = await service.get_health()
            reconnect = stats.get("watchman_reconnect")
            if isinstance(reconnect, dict) and reconnect.get("state") == expected_state:
                return stats
            await asyncio.sleep(0.05)

    return await asyncio.wait_for(_poll(), timeout=timeout)


@pytest.mark.asyncio
async def test_watchman_fresh_instance_requests_resync_without_incremental_translation(
    tmp_path: Path,
) -> None:
    watch_dir = tmp_path / "watchman_project"
    watch_dir.mkdir(parents=True)
    service, services = _build_watchman_service(watch_dir)
    callback_calls: list[tuple[str, dict[str, object] | None]] = []
    callback_event = asyncio.Event()

    async def resync_callback(
        reason: str, details: dict[str, object] | None
    ) -> None:
        callback_calls.append((reason, details))
        callback_event.set()

    service._resync_callback = resync_callback

    try:
        await service.start(watch_dir)
        queue = service.watchman_subscription_queue
        assert queue is not None
        baseline_loss_of_sync = (await service.get_health())["watchman_loss_of_sync"]

        queue.put_nowait(
            {
                "subscription": "chunkhound-live-indexing",
                "clock": "c:0:2",
                "is_fresh_instance": True,
                "files": [
                    {
                        "name": "src/fresh.py",
                        "exists": True,
                        "new": True,
                        "type": "f",
                    }
                ],
            }
        )

        await asyncio.wait_for(callback_event.wait(), timeout=5.0)
        stats = await service.get_health()

        assert callback_calls == [
            (
                "realtime_loss_of_sync",
                {
                    "backend": "watchman",
                    "loss_of_sync_reason": "fresh_instance",
                    "subscription": "chunkhound-live-indexing",
                    "clock": "c:0:2",
                },
            )
        ]
        assert stats["event_queue"]["accepted"] == 0
        assert (
            stats["watchman_loss_of_sync"]["count"]
            == baseline_loss_of_sync["count"] + 1
        )
        assert (
            stats["watchman_loss_of_sync"]["fresh_instance_count"]
            == baseline_loss_of_sync["fresh_instance_count"] + 1
        )
        assert stats["watchman_loss_of_sync"]["recrawl_count"] == baseline_loss_of_sync[
            "recrawl_count"
        ]
        assert (
            stats["watchman_loss_of_sync"]["disconnect_count"]
            == baseline_loss_of_sync["disconnect_count"]
        )
        assert stats["watchman_loss_of_sync"]["last_reason"] == "fresh_instance"
        assert stats["watchman_loss_of_sync"]["last_details"] == {
            "backend": "watchman",
            "loss_of_sync_reason": "fresh_instance",
            "subscription": "chunkhound-live-indexing",
            "clock": "c:0:2",
        }
        assert stats["watchman_loss_of_sync"]["last_at"] is not None
    finally:
        await service.stop()
        services.provider.disconnect()


@pytest.mark.asyncio
async def test_watchman_recrawl_warning_requests_resync_without_incremental_translation(
    tmp_path: Path,
) -> None:
    watch_dir = tmp_path / "watchman_project"
    watch_dir.mkdir(parents=True)
    service, services = _build_watchman_service(watch_dir)
    callback_calls: list[tuple[str, dict[str, object] | None]] = []
    callback_event = asyncio.Event()

    async def resync_callback(
        reason: str, details: dict[str, object] | None
    ) -> None:
        callback_calls.append((reason, details))
        callback_event.set()

    service._resync_callback = resync_callback

    try:
        await service.start(watch_dir)
        queue = service.watchman_subscription_queue
        assert queue is not None
        baseline_loss_of_sync = (await service.get_health())["watchman_loss_of_sync"]

        queue.put_nowait(
            {
                "subscription": "chunkhound-live-indexing",
                "clock": "c:0:3",
                "warning": "Recrawled this watch due to dropped events",
                "files": [
                    {
                        "name": "src/recrawl.py",
                        "exists": True,
                        "new": False,
                        "type": "f",
                    }
                ],
            }
        )

        await asyncio.wait_for(callback_event.wait(), timeout=5.0)
        stats = await service.get_health()

        assert callback_calls == [
            (
                "realtime_loss_of_sync",
                {
                    "backend": "watchman",
                    "loss_of_sync_reason": "recrawl",
                    "subscription": "chunkhound-live-indexing",
                    "clock": "c:0:3",
                    "warning": "Recrawled this watch due to dropped events",
                },
            )
        ]
        assert stats["event_queue"]["accepted"] == 0
        assert (
            stats["watchman_loss_of_sync"]["count"]
            == baseline_loss_of_sync["count"] + 1
        )
        assert (
            stats["watchman_loss_of_sync"]["fresh_instance_count"]
            == baseline_loss_of_sync["fresh_instance_count"]
        )
        assert (
            stats["watchman_loss_of_sync"]["recrawl_count"]
            == baseline_loss_of_sync["recrawl_count"] + 1
        )
        assert (
            stats["watchman_loss_of_sync"]["disconnect_count"]
            == baseline_loss_of_sync["disconnect_count"]
        )
        assert stats["watchman_loss_of_sync"]["last_reason"] == "recrawl"
        assert stats["watchman_loss_of_sync"]["last_details"] == {
            "backend": "watchman",
            "loss_of_sync_reason": "recrawl",
            "subscription": "chunkhound-live-indexing",
            "clock": "c:0:3",
            "warning": "Recrawled this watch due to dropped events",
        }
    finally:
        await service.stop()
        services.provider.disconnect()


@pytest.mark.asyncio
async def test_watchman_unexpected_session_exit_requests_resync_and_restores_monitoring(
    tmp_path: Path,
) -> None:
    watch_dir = tmp_path / "watchman_project"
    watch_dir.mkdir(parents=True)
    service, services = _build_watchman_service(watch_dir)
    callback_calls: list[tuple[str, dict[str, object] | None]] = []
    callback_event = asyncio.Event()

    async def resync_callback(
        reason: str, details: dict[str, object] | None
    ) -> None:
        callback_calls.append((reason, details))
        callback_event.set()

    service._resync_callback = resync_callback

    try:
        await service.start(watch_dir)
        adapter = service._monitor_adapter
        session = getattr(adapter, "_session", None)
        process = getattr(session, "_process", None)
        assert process is not None
        baseline_loss_of_sync = (await service.get_health())["watchman_loss_of_sync"]

        process.terminate()

        await asyncio.wait_for(callback_event.wait(), timeout=5.0)
        stats = await _wait_for_watchman_reconnect_state(service, "restored")

        assert callback_calls
        assert stats["watchman_session_alive"] is True
        assert stats["watchman_connection_state"] == "connected"
        assert (
            stats["watchman_loss_of_sync"]["count"]
            >= baseline_loss_of_sync["count"] + 1
        )
        assert (
            stats["watchman_loss_of_sync"]["disconnect_count"]
            >= baseline_loss_of_sync["disconnect_count"] + 1
        )
        assert stats["watchman_reconnect"]["attempt_count"] >= 1
        assert stats["watchman_reconnect"]["last_result"] == "restored"
        assert stats["service_state"] == "running"
    finally:
        await service.stop()
        services.provider.disconnect()


@pytest.mark.asyncio
async def test_watchman_failed_reconnect_state_is_observable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watch_dir = tmp_path / "watchman_project"
    watch_dir.mkdir(parents=True)
    service, services = _build_watchman_service(watch_dir)
    callback_event = asyncio.Event()

    async def resync_callback(
        reason: str, details: dict[str, object] | None
    ) -> None:
        assert reason == "realtime_loss_of_sync"
        assert details is not None
        callback_event.set()

    service._resync_callback = resync_callback

    try:
        await service.start(watch_dir)
        adapter = service._monitor_adapter
        assert adapter is not None
        monkeypatch.setattr(adapter, "_RECONNECT_RETRY_DELAY_SECONDS", 0.01)

        async def failing_establish(*args, **kwargs) -> None:
            raise RuntimeError("simulated reconnect failure")

        monkeypatch.setattr(adapter, "_establish_monitoring", failing_establish)

        session = getattr(adapter, "_session", None)
        process = getattr(session, "_process", None)
        assert process is not None

        process.terminate()

        await asyncio.wait_for(callback_event.wait(), timeout=5.0)
        stats = await _wait_for_watchman_reconnect_state(service, "failed")

        assert stats["watchman_connection_state"] in {"disconnected", "sidecar_only"}
        assert (
            stats["watchman_reconnect"]["attempt_count"]
            == adapter._RECONNECT_MAX_ATTEMPTS
        )
        assert stats["watchman_reconnect"]["last_result"] == "failed"
        assert "simulated reconnect failure" in (
            stats["watchman_reconnect"]["last_error"] or ""
        )
        assert stats["service_state"] == "degraded"
    finally:
        await service.stop()
        services.provider.disconnect()


@pytest.mark.asyncio
async def test_service_stop_during_reconnect_teardown_does_not_orphan_cli_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watch_dir = tmp_path / "watchman_project"
    watch_dir.mkdir(parents=True)
    service, services = _build_watchman_service(watch_dir)
    stop_task: asyncio.Task[None] | None = None
    old_pid: int | None = None

    try:
        await service.start(watch_dir)
        adapter = service._monitor_adapter
        assert adapter is not None
        session = getattr(adapter, "_session", None)
        process = getattr(session, "_process", None)
        stdin = getattr(process, "stdin", None)
        assert process is not None
        assert stdin is not None
        old_pid = process.pid

        teardown_waiting = asyncio.Event()
        allow_wait_closed = asyncio.Event()

        monkeypatch.setattr(stdin, "close", lambda: None)

        async def blocked_wait_closed() -> None:
            teardown_waiting.set()
            await allow_wait_closed.wait()

        monkeypatch.setattr(stdin, "wait_closed", blocked_wait_closed)

        adapter._begin_reconnect_cycle()
        await asyncio.wait_for(teardown_waiting.wait(), timeout=5.0)
        assert adapter._reconnect_task is not None

        stop_task = asyncio.create_task(service.stop())
        await asyncio.sleep(0)
        allow_wait_closed.set()
        await asyncio.wait_for(stop_task, timeout=5.0)
        stop_task = None

        assert not psutil.pid_exists(old_pid)
    finally:
        if stop_task is not None and not stop_task.done():
            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task
        if old_pid is not None and psutil.pid_exists(old_pid):
            try:
                lingering = psutil.Process(old_pid)
                lingering.kill()
                lingering.wait(timeout=3.0)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                pass
        await service.stop()
        services.provider.disconnect()
