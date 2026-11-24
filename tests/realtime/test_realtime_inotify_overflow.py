"""A kernel inotify queue overflow requests a realtime resync.

watchdog 7.1.1 emits EventsLostEvent after it rebuilds the watches a kernel
IN_Q_OVERFLOW dropped. ChunkHound accepts that event and asks for a resync.
A later content write is still a content change. watchdog 6 has no such event.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chunkhound.core.config.config import Config
from chunkhound.services.realtime.events import (
    SimpleEventHandler,
    watchdog_event_filter,
)
from chunkhound.services.realtime.inotify_overflow import (
    InotifyOverflowDetection,
    inotify_overflow_detection,
)
from chunkhound.services.realtime.service import RealtimeIndexingService


def test_events_lost_is_in_the_linux_filter_only() -> None:
    """Linux admits EventsLostEvent. Other platforms keep it out of the filter."""
    watchdog_events = pytest.importorskip("watchdog.events")
    events_lost = getattr(watchdog_events, "EventsLostEvent", None)
    if events_lost is None:
        pytest.skip("watchdog 6 has no EventsLostEvent")

    classes = watchdog_event_filter()
    if sys.platform.startswith("linux"):
        assert events_lost in classes
        assert inotify_overflow_detection() == "active"
    else:
        assert events_lost not in classes
        assert inotify_overflow_detection() == "not_applicable"


def test_events_lost_requests_a_resync_and_a_write_stays_a_change(
    tmp_path: Path,
) -> None:
    """EventsLostEvent calls the overflow callback. A content write still queues."""
    watchdog_events = pytest.importorskip("watchdog.events")
    events_lost_cls = getattr(watchdog_events, "EventsLostEvent", None)
    if events_lost_cls is None:
        pytest.skip("watchdog 6 has no EventsLostEvent")

    calls: list[None] = []
    seen: list[str] = []
    handler = SimpleEventHandler(
        None,
        Config(target_dir=tmp_path),
        None,
        root_path=tmp_path,
        source_event_callback=lambda kind, _path: seen.append(kind),
        events_lost_callback=lambda: calls.append(None),
    )
    source = tmp_path / "module.py"
    source.write_text("x = 1\n", encoding="utf-8")

    handler.dispatch(watchdog_events.FileModifiedEvent(str(source)))
    handler.dispatch(events_lost_cls(str(tmp_path), is_directory=True))

    assert seen == ["modified"]
    assert calls == [None]


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="inotify is Linux-only"
)
def test_linux_observer_watches_and_reports_events_lost(tmp_path: Path) -> None:
    """The real Linux observer starts, and its handler accepts EventsLostEvent."""
    watchdog_events = pytest.importorskip("watchdog.events")
    events_lost_cls = getattr(watchdog_events, "EventsLostEvent", None)
    if events_lost_cls is None:
        pytest.skip("watchdog 6 drops the overflow record and emits no event")

    import chunkhound.services.realtime.startup as startup_module

    calls: list[None] = []
    observer, detection = startup_module._create_watchdog_observer()
    handler = SimpleEventHandler(
        None,
        Config(target_dir=tmp_path),
        None,
        root_path=tmp_path,
        events_lost_callback=lambda: calls.append(None),
    )
    observer.schedule(
        handler,
        str(tmp_path),
        recursive=True,
        event_filter=watchdog_event_filter(),
    )
    observer.start()
    try:
        assert detection == "active"
        assert observer.is_alive()
        handler.dispatch(events_lost_cls(str(tmp_path), is_directory=True))
        assert calls == [None]
    finally:
        observer.stop()
        observer.join(timeout=1.0)


@pytest.mark.asyncio
async def test_overflow_requests_a_loss_of_sync_resync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chunkhound.services.realtime.startup as startup_module

    class FakeObserver:
        def schedule(self, handler: Any, path: str, **kwargs: Any) -> None: ...

        def start(self) -> None: ...

        def is_alive(self) -> bool:
            return True

        def stop(self) -> None: ...

        def join(self, timeout: float | None = None) -> None: ...

    def fake_create() -> tuple[FakeObserver, InotifyOverflowDetection]:
        return FakeObserver(), "active"

    monkeypatch.setattr(startup_module, "_create_watchdog_observer", fake_create)
    config = Config(target_dir=tmp_path, indexing={"realtime_backend": "watchdog"})
    service = RealtimeIndexingService(SimpleNamespace(provider=None), config)

    async def resync_callback(reason: str, details: Any) -> None:
        return None

    service._resync_callback = resync_callback
    resync_requested = asyncio.Event()
    original_request_resync = service.request_resync

    async def recording_request_resync(
        reason: str, details: dict[str, Any] | None = None
    ) -> bool:
        result = await original_request_resync(reason, details)
        resync_requested.set()
        return result

    monkeypatch.setattr(service, "request_resync", recording_request_resync)
    loop = asyncio.get_running_loop()
    try:
        bootstrap = service._bootstrap_fs_monitor(tmp_path, loop, threading.Event())
        assert bootstrap is not None
        service._adopt_watchdog_monitor(
            bootstrap.observer,
            bootstrap.event_handler,
            tmp_path,
            bootstrap.inotify_overflow_detection,
        )

        watchdog_events = pytest.importorskip("watchdog.events")
        events_lost_cls = getattr(watchdog_events, "EventsLostEvent", None)
        if events_lost_cls is None:
            pytest.skip("watchdog 6 has no EventsLostEvent")
        bootstrap.event_handler.dispatch(
            events_lost_cls(str(tmp_path), is_directory=True)
        )
        await asyncio.wait_for(resync_requested.wait(), timeout=5.0)

        health = await service.get_health()
        assert health["resync"]["request_count"] == 1
        assert health["resync"]["last_reason"] == "realtime_loss_of_sync"
        details = health["resync"]["last_details"]
        assert details["loss_of_sync_reason"] == "inotify_queue_overflow"
        overflow = health["inotify_overflow"]
        assert overflow["detection"] == "active"
        assert overflow["count"] == 1
        assert overflow["last_at"] is not None
    finally:
        await service.stop()
