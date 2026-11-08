"""Shared CLI/MCP websearch runner with a split discovery/research budget."""

from __future__ import annotations

import asyncio
import urllib.error
from collections.abc import AsyncIterator, Callable
from typing import Any

from chunkhound.utils.websearch_core import websearch_timeout

_DISCOVERY_TIMEOUT_CAP_SECONDS = 60.0


class EmptyWebSearchError(Exception):
    """DuckDuckGo returned no rows."""


class NoPagesFetchedError(Exception):
    """Every fetched URL failed before research saw a body."""


class WebSearchTimeoutError(TimeoutError):
    """Phase timeout; distinct from wait_for so 3.11 does not remap it."""


def websearch_discovery_timeout(total: float) -> float:
    """Cap time spent on query expansion + DDG so it cannot consume research."""
    return min(_DISCOVERY_TIMEOUT_CAP_SECONDS, max(0.0, total / 4.0))


def _timeout(phase: str, seconds: float) -> WebSearchTimeoutError:
    if phase == "discovery":
        return WebSearchTimeoutError(
            f"websearch timed out after {seconds:.0f}s waiting for search results"
        )
    return WebSearchTimeoutError(f"websearch timed out after {seconds:.0f}s")


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    """Retrieve a detached task's result so late failures stay quiet."""
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


async def _cancel_and_wait(task: asyncio.Task[Any], *, deadline: float) -> None:
    """Cancel a sibling task, detaching it when the total budget is spent."""
    if task.done():
        _consume_task_result(task)
        return
    loop = asyncio.get_running_loop()
    task.cancel()
    remaining = max(0.0, deadline - loop.time())
    try:
        await asyncio.wait({task}, timeout=remaining)
    finally:
        if task.done():
            _consume_task_result(task)
        else:
            task.add_done_callback(_consume_task_result)


async def run_websearch(
    query: str,
    *,
    limit: int,
    config: Any,
    embedding_manager: Any,
    llm_manager: Any,
    previous_query: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
    progress: Any = None,
    failure_callback: Callable[[str, urllib.error.URLError], None] | None = None,
    timeout_s: float | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Expand, search, fetch, and research under one total wall-clock budget.

    Discovery (expansion + DuckDuckGo) is capped at a quarter of the total,
    at most 60s. Fetch overlaps research under whatever time remains. Research
    stays on the same task as the page async generator: wrapping only
    research in wait_for moves iteration to a child task and hangs on 3.10.
    """
    from chunkhound.services.web_research_service import Page, research_web_pages
    from chunkhound.utils.websearch_core import fetch_pages, search_multi
    from chunkhound.utils.websearch_expansion import expand_web_queries

    total = websearch_timeout() if timeout_s is None else timeout_s
    loop = asyncio.get_running_loop()
    deadline = loop.time() + total

    async def _run() -> tuple[dict[str, Any], list[str]]:
        discovery_s = websearch_discovery_timeout(total)

        async def discover() -> list[tuple[str, str, str]]:
            queries = await expand_web_queries(
                query, llm_manager, previous_query=previous_query
            )
            return await search_multi(
                queries,
                limit,
                progress_callback,
                failure_callback=failure_callback,
            )

        discovery_task = asyncio.create_task(discover())
        try:
            done, _pending = await asyncio.wait({discovery_task}, timeout=discovery_s)
            if not done:
                raise _timeout("discovery", discovery_s)
            results = discovery_task.result()
        finally:
            await _cancel_and_wait(discovery_task, deadline=deadline)

        if not results:
            raise EmptyWebSearchError(
                f"No results found for {query!r} — DDG HTML structure may have changed"
            )
        if progress_callback is not None:
            progress_callback(f"Found {len(results)} results, fetching content...")

        warnings: list[str] = []
        fetched_pages = fetch_pages(
            [url for _, url, _ in results],
            progress_callback,
            warnings.append,
        )
        try:
            first_page = await anext(fetched_pages, None)
            if first_page is None:
                detail = "".join(f"\n  - {w}" for w in warnings)
                raise NoPagesFetchedError(
                    f"No pages could be fetched for {query!r}{detail}"
                )

            async def pages() -> AsyncIterator[Page]:
                yield first_page
                async for page in fetched_pages:
                    yield page

            result = await research_web_pages(
                query,
                pages(),
                config,
                embedding_manager,
                llm_manager,
                progress=progress,
                warning_callback=warnings.append,
                previous_query=previous_query,
            )
            return result, warnings
        except (
            NoPagesFetchedError,
            WebSearchTimeoutError,
            asyncio.CancelledError,
        ):
            raise
        except Exception as exc:
            exc.websearch_warnings = list(warnings)  # type: ignore[attr-defined]
            raise
        finally:
            await fetched_pages.aclose()

    try:
        return await asyncio.wait_for(_run(), timeout=total)
    except WebSearchTimeoutError:
        raise
    except (asyncio.TimeoutError, TimeoutError):
        # On 3.11+ asyncio.TimeoutError is the builtin, so provider/socket
        # timeouts are indistinguishable by type. Only the budget decides.
        if loop.time() >= deadline:
            raise _timeout("research", total) from None
        raise
