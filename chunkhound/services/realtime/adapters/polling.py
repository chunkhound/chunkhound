from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..context import RealtimeServiceContext, coerce_realtime_context


class PollingRealtimeAdapter:
    """Explicit polling backend."""

    backend_name = "polling"

    def __init__(self, context: RealtimeServiceContext | object) -> None:
        self._context = coerce_realtime_context(context)

    async def start(self, watch_path: Path, loop: asyncio.AbstractEventLoop) -> None:
        del loop
        self._context.start_startup_phase("polling_setup")
        await self._context.start_polling_backend(
            watch_path,
            reason="Configured realtime backend is polling",
            emit_warning=False,
        )

    async def stop(self) -> None:
        await self._context.cancel_polling_task()

    def get_health(self) -> dict[str, Any]:
        return {
            "observer_alive": bool(
                self._context.polling_task and not self._context.polling_task.done()
            )
        }
