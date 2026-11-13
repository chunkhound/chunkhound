"""Every in-place write is indexed, and a plain read costs nothing.

On Windows, watchdog 7 reports a FILE_ACTION_MODIFIED as `attrib` whenever the
record's change time is ahead of its write time, a race an ordinary append
usually loses: on NTFS most in-place writes arrive as `attrib` alone, and an
`attrib` the consumer does not act on is a write that is never indexed. And
while watchdog subscribes to last-access changes, every read of a file not
read for an hour raises a notification too (7: `attrib`; 6: `modified`), each
taking an event-queue slot; the start-up scan reads every file.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.core.config.config import Config
from chunkhound.services.realtime.events import (
    SimpleEventHandler,
    watchdog_event_filter,
)
from chunkhound.services.realtime.service import RealtimeIndexingService


def _handler(
    tmp_path: Path, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop
) -> tuple[SimpleEventHandler, list[tuple]]:
    accepted: list[tuple] = []
    handler = SimpleEventHandler(
        queue,
        config=Config(target_dir=tmp_path),
        loop=loop,
        root_path=tmp_path,
        queue_result_callback=lambda *args: accepted.append(args),
    )
    return handler, accepted


@pytest.mark.asyncio
async def test_attrib_event_is_queued_as_a_change(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("x = 1\n")
    queue: asyncio.Queue[tuple[str, Path]] = asyncio.Queue(
        maxsize=RealtimeIndexingService._EVENT_QUEUE_MAXSIZE
    )
    handler, accepted = _handler(tmp_path, queue, asyncio.get_running_loop())

    handler.dispatch(
        SimpleNamespace(event_type="attrib", is_directory=False, src_path=str(source))
    )
    await asyncio.sleep(0)  # run the call_soon_threadsafe enqueue

    assert queue.get_nowait() == ("modified", source.resolve())
    assert [(event_type, ok) for event_type, _, ok, _ in accepted] == [
        ("modified", True)
    ]


def _age_access_time(path: Path, hours: float = 3.0) -> None:
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns - int(hours * 3600 * 1e9), st.st_mtime_ns))


async def _drain_until(
    queue: asyncio.Queue, predicate, timeout: float
) -> list[tuple[str, Path]]:
    """Collect queue items until one matches `predicate` or `timeout` passes."""
    items: list[tuple[str, Path]] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            item = await asyncio.wait_for(queue.get(), 0.1)
        except asyncio.TimeoutError:
            continue
        items.append(item)
        if predicate(item):
            break
    return items


@pytest.mark.skipif(sys.platform != "win32", reason="ReadDirectoryChangesW only")
@pytest.mark.asyncio
async def test_windows_read_raises_no_event_and_write_is_a_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    winapi = pytest.importorskip("watchdog.observers.winapi")
    import chunkhound.services.realtime.startup as startup_module

    # The helper narrows a process-wide watchdog global; restore it afterwards.
    monkeypatch.setattr(
        winapi, "WATCHDOG_FILE_NOTIFY_FLAGS", winapi.WATCHDOG_FILE_NOTIFY_FLAGS
    )
    source = tmp_path / "module.py"
    source.write_text("x = 1\n" * 50)
    sentinel = tmp_path / "sentinel.py"
    # Aged before the watch exists: os.utime itself is a change notification.
    _age_access_time(source)
    aged_atime = source.stat().st_atime_ns

    queue: asyncio.Queue[tuple[str, Path]] = asyncio.Queue()
    handler, _ = _handler(tmp_path, queue, asyncio.get_running_loop())
    observer = startup_module._create_watchdog_observer()
    observer.schedule(
        handler, str(tmp_path), recursive=True, event_filter=watchdog_event_filter()
    )
    observer.start()
    try:
        # A watch reports nothing until ReadDirectoryChangesW is armed: prove it
        # is armed with a write of our own, then start from an empty queue.
        sentinel.write_text("y = 1\n")
        armed = await _drain_until(
            queue, lambda item: item[1] == sentinel.resolve(), timeout=5.0
        )
        assert armed and armed[-1][1] == sentinel.resolve()
        await asyncio.sleep(0.5)
        while not queue.empty():
            queue.get_nowait()

        with open(source, encoding="utf-8") as f:
            f.read()
        if source.stat().st_atime_ns == aged_atime:
            pytest.skip("last-access updates are off on this volume")
        after_read = await _drain_until(queue, lambda item: False, timeout=1.5)
        assert after_read == []

        with open(source, "a", encoding="utf-8") as f:
            f.write("z = 2\n")
        after_write = await _drain_until(
            queue, lambda item: item[1] == source.resolve(), timeout=3.0
        )
        assert ("modified", source.resolve()) in after_write
    finally:
        observer.stop()
        observer.join(timeout=5)
