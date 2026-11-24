"""An access is not a content change on Windows, Linux, and macOS.

The watchdog observer is the one ChunkHound starts. A read must not take an
event-queue slot, and a later write must. On Windows a read often does not
touch last-access time, so the test also sets that timestamp on its own.
Linux inotify does not watch IN_ACCESS, and macOS FSEvents does not report a
read as a content change. An atime-only utime on those platforms is generic
metadata (chmod and touch use the same signal) and stays a change.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

from chunkhound.core.config.config import Config
from chunkhound.services.realtime.events import (
    SimpleEventHandler,
    watchdog_event_filter,
)
from chunkhound.services.realtime.service import RealtimeIndexingService
from tests.utils.windows_compat import get_fs_event_timeout

pytestmark = pytest.mark.skipif(
    sys.platform not in {"win32", "linux", "darwin"},
    reason="covered on Windows, Linux, and macOS",
)

# A missing event has to be waited out. The sibling Windows test uses this bound;
# the full filesystem timeout is for the sentinel and the write.
_SILENCE_TIMEOUT_SECONDS = 1.5


def _handler(
    tmp_path: Path, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop
) -> SimpleEventHandler:
    return SimpleEventHandler(
        queue,
        config=Config(target_dir=tmp_path),
        loop=loop,
        root_path=tmp_path,
    )


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


def _set_access_time_only(path: Path) -> None:
    """Set last-access through SetFileTime and leave last-write unchanged.

    ``os.utime`` on Windows also updates file attributes, so ReadDirectoryChangesW
    reports it through FILE_NOTIFY_CHANGE_ATTRIBUTES. SetFileTime of the access
    time alone is the last-access notification.
    """
    import ctypes
    from ctypes import wintypes

    class _FileTime(ctypes.Structure):
        _fields_ = (
            ("dwLowDateTime", wintypes.DWORD),
            ("dwHighDateTime", wintypes.DWORD),
        )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    invalid = ctypes.c_void_p(-1).value
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.GetFileTime.restype = wintypes.BOOL
    kernel32.SetFileTime.restype = wintypes.BOOL
    kernel32.CloseHandle.restype = wintypes.BOOL

    before = path.stat()
    handle = kernel32.CreateFileW(
        str(path),
        0x100,  # FILE_WRITE_ATTRIBUTES
        0x01 | 0x02 | 0x04,
        None,
        3,  # OPEN_EXISTING
        0x80,  # FILE_ATTRIBUTE_NORMAL
        None,
    )
    if handle == invalid:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        accessed = _FileTime()
        if not kernel32.GetFileTime(handle, None, ctypes.byref(accessed), None):
            raise ctypes.WinError(ctypes.get_last_error())
        ticks = (accessed.dwHighDateTime << 32) | accessed.dwLowDateTime
        ticks += 3600 * 10_000_000
        accessed.dwLowDateTime = ticks & 0xFFFFFFFF
        accessed.dwHighDateTime = (ticks >> 32) & 0xFFFFFFFF
        if not kernel32.SetFileTime(handle, None, ctypes.byref(accessed), None):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(handle)

    after = path.stat()
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_atime_ns != before.st_atime_ns


def _assert_windows_notify_filter_omits_last_access(observer: object) -> None:
    """When this watchdog accepts notify_filter, the emitter uses it.

    watchdog 6 has no emitter attribute and clears the module global instead.
    That path is covered by the read and SetFileTime checks below.
    """
    import watchdog.observers.winapi as winapi

    emitters = list(getattr(observer, "emitters", ()))
    assert len(emitters) == 1
    notify_filter = getattr(emitters[0], "notify_filter", None)
    if not isinstance(notify_filter, int):
        return
    last_access = winapi.FILE_NOTIFY_CHANGE_LAST_ACCESS
    assert winapi.WATCHDOG_FILE_NOTIFY_FLAGS & last_access
    assert notify_filter & last_access == 0
    assert notify_filter == (winapi.WATCHDOG_FILE_NOTIFY_FLAGS & ~last_access)


@pytest.mark.asyncio
async def test_access_is_not_a_content_change_and_a_write_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read and a Windows access-time update stay quiet. A write does not."""
    import chunkhound.services.realtime.startup as startup_module

    source = tmp_path / "module.py"
    sentinel = tmp_path / "sentinel.py"
    source.write_text("x = 1\n")
    if sys.platform == "win32":
        import watchdog.observers.winapi as winapi

        # watchdog 6 clears a process-wide global. Restore the library default.
        monkeypatch.setattr(
            winapi, "WATCHDOG_FILE_NOTIFY_FLAGS", winapi.WATCHDOG_FILE_NOTIFY_FLAGS
        )
        # Aged before the watch exists: os.utime itself would notify.
        before = source.stat()
        os.utime(
            source,
            ns=(before.st_atime_ns - 3_600_000_000_000, before.st_mtime_ns),
        )

    queue: asyncio.Queue[tuple[str, Path]] = asyncio.Queue(
        maxsize=RealtimeIndexingService._EVENT_QUEUE_MAXSIZE
    )
    handler = _handler(tmp_path, queue, asyncio.get_running_loop())
    observer = startup_module._create_watchdog_observer()
    observer.schedule(
        handler, str(tmp_path), recursive=True, event_filter=watchdog_event_filter()
    )
    if sys.platform == "win32":
        _assert_windows_notify_filter_omits_last_access(observer)
    observer.start()
    try:
        timeout = get_fs_event_timeout()
        sentinel.write_text("y = 1\n")
        armed = await _drain_until(
            queue, lambda item: item[1] == sentinel.resolve(), timeout
        )
        assert armed and armed[-1][1] == sentinel.resolve()
        await asyncio.sleep(0.5)
        while not queue.empty():
            queue.get_nowait()

        with source.open("rb") as handle:
            handle.read()
        after_read = await _drain_until(
            queue, lambda item: False, _SILENCE_TIMEOUT_SECONDS
        )
        assert after_read == []

        if sys.platform == "win32":
            _set_access_time_only(source)
            after_access = await _drain_until(
                queue, lambda item: False, _SILENCE_TIMEOUT_SECONDS
            )
            assert after_access == []

        with source.open("a", encoding="utf-8") as handle:
            handle.write("z = 2\n")
        after_write = await _drain_until(
            queue, lambda item: item[1] == source.resolve(), timeout
        )
        assert ("modified", source.resolve()) in after_write
    finally:
        observer.stop()
        observer.join(5)
