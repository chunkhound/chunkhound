"""Read-only filesystem events never reach the realtime event queue.

On Linux, watchdog's inotify emitter reports every open and close of a file. A
plain read (ChunkHound's own start-up scan, or tsc/vitest/lint over a worktree)
then produced `opened` + `closed_no_write` per file, each of which took one of
the event queue's slots before the consumer dropped it; a scan of a tree larger
than the queue overflowed it and queued a full resync.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
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


ProcessFile = Callable[..., Awaitable[dict[str, object]]]


def _service_with_handler(
    root: Path, process_file: ProcessFile
) -> tuple[RealtimeIndexingService, SimpleEventHandler]:
    """A running service and a handler wired as startup.py wires it.

    Embedding follow-ups are skipped, and the duplicate-event window is long
    enough that every repeated write in a test lands inside it.
    """
    config = Config(target_dir=root, indexing={"include": ["*.py"]})
    services = SimpleNamespace(
        provider=SimpleNamespace(),
        indexing_coordinator=SimpleNamespace(process_file=process_file),
    )
    service = RealtimeIndexingService(services, config)
    service._EVENT_DEDUP_WINDOW_SECONDS = 60.0
    add_file = service.add_file

    async def add_file_without_embed(file_path: Path, priority: str = "change") -> bool:
        if priority == "embed":
            return True
        return await add_file(file_path, priority)

    service.add_file = add_file_without_embed  # type: ignore[method-assign]
    service._service_state = "running"
    service._effective_backend = "watchdog"
    service.monitoring_ready.set()
    service._monitoring_ready_at = service._utc_now()
    handler = SimpleEventHandler(
        service.event_queue,
        config=config,
        loop=asyncio.get_running_loop(),
        root_path=root,
        queue_result_callback=service._handle_queue_result,
        source_event_callback=service._record_source_event,
        filtered_event_callback=service._record_filtered_event,
        admission_callback=service._should_admit_realtime_event,
    )
    return service, handler


@contextlib.asynccontextmanager
async def _consuming(service: RealtimeIndexingService) -> AsyncIterator[None]:
    """Run the event consumer and the process loop for the block."""
    tasks = [
        asyncio.create_task(service._consume_events()),
        asyncio.create_task(service._process_loop()),
    ]
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_write_after_an_indexing_read_is_indexed(tmp_path: Path) -> None:
    """With reads dropped, two writes reach the consumer as modified, modified.

    The second lands while the first is being processed, after that run read
    the file, and inside the duplicate-event window. The open and close events
    that used to separate the two are gone, so only the consumer's rule for a
    path being processed keeps the second write from being dropped as a
    duplicate.
    """
    source = tmp_path / "module.py"
    source.write_text("x = 1\n")
    processed: list[str] = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_done = asyncio.Event()

    async def process_file(
        file_path: Path, skip_embeddings: bool = False
    ) -> dict[str, object]:
        processed.append(file_path.read_text())
        if len(processed) == 1:
            first_started.set()
            await release_first.wait()
        else:
            second_done.set()
        return {"status": "processed", "chunks": 1, "embeddings": 0}

    service, handler = _service_with_handler(tmp_path, process_file)

    async def deliver(*event_types: str) -> None:
        for event_type in event_types:
            handler.dispatch(_event(event_type, source))
        await asyncio.sleep(0)  # run the call_soon_threadsafe enqueues

    async with _consuming(service):
        await deliver("opened", "modified", "closed")  # the first write
        await asyncio.wait_for(first_started.wait(), timeout=5.0)
        await deliver("opened", "closed_no_write")  # that run's read
        source.write_text("x = 2\n")
        await deliver("opened", "modified", "closed")  # the second write
        await asyncio.wait_for(service.event_queue.join(), timeout=3.0)

        release_first.set()
        await asyncio.wait_for(second_done.wait(), timeout=5.0)
    assert processed == ["x = 1\n", "x = 2\n"]


@pytest.mark.asyncio
async def test_a_burst_of_saves_is_indexed_once(tmp_path: Path) -> None:
    """Saves closer together than the debounce delay are indexed in one pass.

    With reads dropped, every save after the first reaches the consumer as a
    duplicate `modified` and is suppressed. It must still push the pending
    debounce back, as a save does when it reaches add_file, or the burst is
    indexed part-way through and then again.
    """
    source = tmp_path / "module.py"
    source.write_text("x = 0\n")
    processed: list[str] = []
    handlers: list[SimpleEventHandler] = []
    indexed = asyncio.Event()

    async def process_file(
        file_path: Path, skip_embeddings: bool = False
    ) -> dict[str, object]:
        processed.append(file_path.read_text())
        indexed.set()
        for event_type in ("opened", "closed_no_write"):  # this run's read
            handlers[0].dispatch(_event(event_type, source))
        return {"status": "processed", "chunks": 1, "embeddings": 0}

    service, handler = _service_with_handler(tmp_path, process_file)
    handlers.append(handler)
    service._debounce_delay = 1.5
    saves, gap = 8, 0.25  # 1.75 s of saves; no gap comes near the delay

    async with _consuming(service):
        for i in range(1, saves + 1):
            source.write_text(f"x = {i}\n")
            for event_type in ("opened", "modified", "closed"):
                handler.dispatch(_event(event_type, source))
            await asyncio.sleep(gap)
        await asyncio.wait_for(indexed.wait(), timeout=10.0)
        # Long enough for a second pass to show up if one was queued.
        await asyncio.sleep(service._debounce_delay + 0.5)
    assert processed == [f"x = {saves}\n"]


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
