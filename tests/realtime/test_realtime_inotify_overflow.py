"""A kernel inotify queue overflow requests a realtime resync.

When the kernel inotify queue overflows it drops events and queues one
IN_Q_OVERFLOW record. watchdog 7.1 rebuilds its watches but emits no event, so
without a hook ChunkHound never learns that changes were lost.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chunkhound.core.config.config import Config
from chunkhound.services.realtime.service import RealtimeIndexingService


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="inotify is Linux-only"
)
def test_kernel_overflow_reaches_the_service() -> None:
    inotify_c = pytest.importorskip("watchdog.observers.inotify_c")
    if not hasattr(inotify_c, "InotifyFD"):
        pytest.skip("watchdog < 7 has no inotify overflow hook")
    import chunkhound.services.realtime.startup as startup_module

    calls: list[None] = []
    observer, detection = startup_module._create_watchdog_observer(
        lambda: calls.append(None)
    )
    observer.start()
    try:
        assert detection == "active"
        observer._inotify_fd.handle_event(  # type: ignore[attr-defined]
            inotify_c.InotifyEvent(
                -1, inotify_c.InotifyConstants.IN_Q_OVERFLOW, 0, b""
            )
        )
        assert len(calls) == 1
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

    captured: list[Callable[[], None]] = []

    def fake_create(on_overflow: Callable[[], None]) -> tuple[FakeObserver, str]:
        captured.append(on_overflow)
        return FakeObserver(), "active"

    monkeypatch.setattr(startup_module, "_create_watchdog_observer", fake_create)
    config = Config(target_dir=tmp_path, indexing={"realtime_backend": "watchdog"})
    service = RealtimeIndexingService(SimpleNamespace(provider=None), config)

    async def resync_callback(reason: str, details: Any) -> None:
        return None

    service._resync_callback = resync_callback
    loop = asyncio.get_running_loop()
    try:
        observer, handler, detection = service._bootstrap_fs_monitor(
            tmp_path, loop, threading.Event()
        )
        service._adopt_watchdog_monitor(observer, handler, tmp_path, detection)

        thread = threading.Thread(target=captured[0])
        thread.start()
        thread.join()
        for _ in range(200):
            if service._resync_request_count == 1:
                break
            await asyncio.sleep(0.01)

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
