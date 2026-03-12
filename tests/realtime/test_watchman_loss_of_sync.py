from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

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
async def test_watchman_unexpected_session_exit_requests_resync(
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
        await asyncio.sleep(0.3)
        stats = await service.get_health()

        assert callback_calls[0][0] == "realtime_loss_of_sync"
        assert callback_calls[0][1] is not None
        assert callback_calls[0][1]["backend"] == "watchman"
        assert callback_calls[0][1]["loss_of_sync_reason"] == "disconnect"
        assert callback_calls[0][1]["watchman_session_alive"] is False
        assert stats["watchman_session_alive"] is False
        assert (
            stats["watchman_loss_of_sync"]["count"]
            == baseline_loss_of_sync["count"] + 1
        )
        assert (
            stats["watchman_loss_of_sync"]["disconnect_count"]
            == baseline_loss_of_sync["disconnect_count"] + 1
        )
        assert stats["watchman_loss_of_sync"]["last_reason"] == "disconnect"
        assert stats["service_state"] == "degraded"
    finally:
        await service.stop()
        services.provider.disconnect()
