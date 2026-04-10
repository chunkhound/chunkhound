from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..context import RealtimeServiceContext, coerce_realtime_context


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
