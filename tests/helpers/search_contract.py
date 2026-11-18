"""Public search-contract helpers for realtime integration tests.

Tests poll observable MCP search output instead of internal indexing flags.
This keeps them valid across queue, debounce, and storage refactors.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from chunkhound.mcp_server.tools import EMPTY_SEARCH_MESSAGE, execute_tool
from chunkhound.services.realtime.service import RealtimeIndexingService
from tests.utils.windows_compat import get_fs_event_timeout

# Platform fs-event settle: 45s Windows CI, 5s POSIX CI, 3s local (issue #254).
SEARCH_SETTLE_TIMEOUT_SECONDS = get_fs_event_timeout()
SEARCH_POLL_INTERVAL_SECONDS = 0.2
# Recovery-contract waits: generous bounded caps on real outcomes, never on
# interior tick counts. Fast cadence comes from config, not from these caps.
HEALTH_WAIT_TIMEOUT_SECONDS = 10.0
HEALTH_POLL_INTERVAL_SECONDS = 0.05
_REALTIME_SERVICE: ContextVar[RealtimeIndexingService | None] = ContextVar(
    "search_contract_realtime_service", default=None
)


@contextmanager
def realtime_search_diagnostics(service: RealtimeIndexingService) -> Iterator[None]:
    """Attach public health to timeouts without changing existing search callers."""
    token = _REALTIME_SERVICE.set(service)
    try:
        yield
    finally:
        _REALTIME_SERVICE.reset(token)


async def _timeout_diagnostics() -> str:
    service = _REALTIME_SERVICE.get()
    if service is None:
        return ""
    try:
        health = await service.get_health()
    except Exception as error:
        return f"; realtime health unavailable: {type(error).__name__}: {error}"
    return f"; realtime health={json.dumps(health, sort_keys=True, default=str)}"


async def search_tool(
    services: Any,
    query: str,
    *,
    search_type: str = "regex",
    embedding_manager: Any = None,
    page_size: int = 10,
    offset: int = 0,
) -> str:
    """Call the public ``search`` tool and return its markdown contract."""
    result = await execute_tool(
        "search",
        services,
        embedding_manager,
        {
            "type": search_type,
            "query": query,
            "page_size": page_size,
            "offset": offset,
        },
    )
    if not isinstance(result, str):
        raise TypeError(f"search tool returned {type(result).__name__}, expected str")
    return result


async def wait_for_search(
    services: Any,
    query: str,
    predicate: Callable[[str], bool],
    *,
    search_type: str = "regex",
    embedding_manager: Any = None,
    timeout: float = SEARCH_SETTLE_TIMEOUT_SECONDS,
    poll_interval: float = SEARCH_POLL_INTERVAL_SECONDS,
    page_size: int = 10,
) -> str:
    """Poll the public search tool until ``predicate`` accepts the output."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last_result = ""

    while True:
        last_result = await search_tool(
            services,
            query,
            search_type=search_type,
            embedding_manager=embedding_manager,
            page_size=page_size,
        )
        if predicate(last_result):
            return last_result
        if loop.time() >= deadline:
            raise AssertionError(
                f"Search contract not satisfied within {timeout:.1f}s for "
                f"{search_type} query {query!r}; last output={last_result[:300]!r}"
                f"{await _timeout_diagnostics()}"
            )
        await asyncio.sleep(poll_interval)


def results_contain(results: str, text: str) -> bool:
    """True when the rendered search output contains ``text``."""
    return text in results


async def wait_for_health(
    service: RealtimeIndexingService,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout: float = HEALTH_WAIT_TIMEOUT_SECONDS,
    poll_interval: float = HEALTH_POLL_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Await a realtime-health state transition via the public health surface."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last: dict[str, Any] = {}
    while True:
        last = await service.get_health()
        if predicate(last):
            return last
        if loop.time() >= deadline:
            raise AssertionError(
                f"Health contract not satisfied within {timeout:.1f}s; "
                f"last health={json.dumps(last, default=str)[:600]}"
            )
        await asyncio.sleep(poll_interval)


def results_are_empty(results: str) -> bool:
    """True when the rendered search output reports no matches."""
    return results.startswith(EMPTY_SEARCH_MESSAGE)
