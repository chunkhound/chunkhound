"""Read-only filesystem events never reach the realtime event queue.

On Linux, watchdog's inotify emitter reports every open and close of a file. A
plain read (ChunkHound's own start-up scan, or tsc/vitest/lint over a worktree)
then produced `opened` + `closed_no_write` per file, each of which took one of
the event queue's slots before the consumer dropped it; a scan of a tree larger
than the queue overflowed it and queued a full resync.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chunkhound.core.config.config import Config
from chunkhound.services.realtime.events import SimpleEventHandler
from chunkhound.services.realtime.service import RealtimeIndexingService


def _event(event_type: str, path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        event_type=event_type, is_directory=False, src_path=str(path)
    )


@pytest.mark.asyncio
async def test_read_only_events_take_no_queue_slot(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("x = 1\n")
    config = Config(target_dir=tmp_path)
    queue: asyncio.Queue[tuple[str, Path]] = asyncio.Queue(
        maxsize=RealtimeIndexingService._EVENT_QUEUE_MAXSIZE
    )
    results: list[tuple[str, Path, bool, str | None]] = []
    seen: list[tuple[str, Path]] = []
    handler = SimpleEventHandler(
        queue,
        config=config,
        loop=asyncio.get_running_loop(),
        root_path=tmp_path,
        queue_result_callback=lambda *args: results.append(args),
        source_event_callback=lambda *args: seen.append(args),
        filtered_event_callback=lambda *args: seen.append(args),
    )

    burst = RealtimeIndexingService._EVENT_QUEUE_MAXSIZE + 500
    for _ in range(burst):
        handler.dispatch(_event("opened", source))
        handler.dispatch(_event("closed_no_write", source))
        handler.dispatch(_event("closed", source))
    handler.dispatch(_event("modified", source))
    await asyncio.sleep(0)  # run the call_soon_threadsafe enqueues

    assert queue.qsize() == 1
    assert queue.get_nowait() == ("modified", source.resolve())
    assert [r for r in results if not r[2]] == []
    assert [s for s in seen if s[0] != "modified"] == []


def test_watches_are_scheduled_without_read_only_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watchdog_events = pytest.importorskip("watchdog.events")
    import chunkhound.services.realtime.startup as startup_module

    scheduled: list[dict[str, Any]] = []

    class FakeObserver:
        def schedule(self, handler: Any, path: str, **kwargs: Any) -> None:
            scheduled.append(kwargs)

        def start(self) -> None: ...

        def is_alive(self) -> bool:
            return True

        def stop(self) -> None: ...

        def join(self, timeout: float | None = None) -> None: ...

    monkeypatch.setattr(startup_module, "_create_watchdog_observer", FakeObserver)
    config = Config(target_dir=tmp_path, indexing={"realtime_backend": "watchdog"})
    service = RealtimeIndexingService(SimpleNamespace(provider=None), config)
    loop = asyncio.new_event_loop()
    try:
        observer, handler = service._bootstrap_fs_monitor(
            tmp_path, loop, threading.Event()
        )
        service.observer = observer
        service.event_handler = handler
        loop.run_until_complete(service._add_directory_watch(str(tmp_path / "new")))
    finally:
        loop.close()

    assert len(scheduled) == 2
    read_only = {
        watchdog_events.FileOpenedEvent,
        watchdog_events.FileClosedEvent,
        watchdog_events.FileClosedNoWriteEvent,
    }
    content = {
        watchdog_events.FileCreatedEvent,
        watchdog_events.FileModifiedEvent,
        watchdog_events.FileDeletedEvent,
        watchdog_events.FileMovedEvent,
        watchdog_events.DirCreatedEvent,
        watchdog_events.DirDeletedEvent,
        watchdog_events.DirMovedEvent,
    }
    if hasattr(watchdog_events, "FileAttribEvent"):  # watchdog 7: can be a write
        content.add(watchdog_events.FileAttribEvent)
    for kwargs in scheduled:
        event_filter = set(kwargs.get("event_filter") or ())
        assert content <= event_filter
        assert not (read_only & event_filter)
