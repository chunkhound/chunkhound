from __future__ import annotations

import asyncio
import functools
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ..context import RealtimeServiceContext, coerce_realtime_context

if TYPE_CHECKING:
    from ..startup import RealtimeObserver


@functools.cache
def _rescan_aware_observer_class() -> type[Any]:
    """Define the macOS observer once, keeping watchdog imports lazy.

    macOS FSEvents can silently LOSE events: native batches carry
    `must_scan_subdirs` (kernel/user dropped) which watchdog ignores. Counting
    them lets realtime supervision trigger a reconciliation rescan.
    """
    from watchdog.observers.api import DEFAULT_OBSERVER_TIMEOUT
    from watchdog.observers.fsevents import FSEventsEmitter, FSEventsObserver

    class RescanAwareFSEventsEmitter(FSEventsEmitter):
        # Supervision reads this counter; the class default keeps it simple.
        dropped_event_batches = 0

        # NativeEvent-bearing signature is FSEventsEmitter's own override shape
        # (watchdog's fsevents.py carries the same ignore).
        def queue_events(  # type: ignore[override]
            self, timeout: float, events: list[Any]
        ) -> None:
            if any(getattr(event, "must_scan_subdirs", False) for event in events):
                self.dropped_event_batches += 1
            super().queue_events(timeout, events)

    class RescanAwareFSEventsObserver(FSEventsObserver):
        def __init__(self, *, timeout: float = DEFAULT_OBSERVER_TIMEOUT) -> None:
            super().__init__(timeout=timeout)
            self._emitter_class = RescanAwareFSEventsEmitter

    return RescanAwareFSEventsObserver


def create_watchdog_observer() -> RealtimeObserver:
    """Build the watchdog observer for this platform (lazy watchdog import)."""
    if sys.platform != "darwin":
        from watchdog.observers import Observer

        return Observer()

    return cast("RealtimeObserver", _rescan_aware_observer_class()())


class WatchdogRealtimeAdapter:
    """Watchdog-backed monitor with polling fallback."""

    backend_name = "watchdog"

    def __init__(self, context: RealtimeServiceContext | object) -> None:
        self._context = coerce_realtime_context(context)

    async def start(self, watch_path: Path, loop: asyncio.AbstractEventLoop) -> None:
        self._context.set_effective_backend(self.backend_name)
        self._context.start_startup_phase("watchdog_setup")
        await self._context.setup_watchdog_with_timeout(watch_path, loop)

    async def stop(self) -> None:
        await self._context.cancel_watchdog_setup_task()
        await self._context.cancel_watchdog_bootstrap_future()
        await self._context.stop_observer()
        await self._context.cancel_polling_task()

    def get_health(self) -> dict[str, Any]:
        observer_alive = False
        if self._context.observer and self._context.observer.is_alive():
            observer_alive = True
        elif (
            self._context.using_polling
            and self._context.polling_task
            and not self._context.polling_task.done()
        ):
            observer_alive = True
        return {"observer_alive": observer_alive}
