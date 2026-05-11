"""Helpers for tracking owned transient realtime tasks."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any


class OwnedTaskSet:
    """Track short-lived tasks so shutdown can cancel them deterministically."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()

    def create_task(self, awaitable: Awaitable[Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(awaitable)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def cancel_all(self) -> None:
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
