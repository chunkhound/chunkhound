"""User-visible error contracts for in-process MCP websearch."""

import asyncio
import urllib.error
from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.mcp_server import tools
from chunkhound.mcp_server.common import MCPError
from chunkhound.services import web_research_service
from chunkhound.utils import websearch_core, websearch_expansion


@pytest.fixture
def patched(monkeypatch):
    search = AsyncMock(
        return_value=[("Title", "https://example.invalid/page", "snippet")]
    )
    monkeypatch.setattr(websearch_core, "search_multi", search)
    monkeypatch.setattr(
        websearch_expansion,
        "expand_web_queries",
        AsyncMock(return_value=["query"]),
    )

    async def pages(*args, **kwargs):
        yield "https://example.invalid/page", ".md", "# Page"

    monkeypatch.setattr(websearch_core, "fetch_pages", pages)

    async def research(query, page_stream, *args, **kwargs):
        async for _ in page_stream:
            pass
        return {"answer": "ANSWER"}

    monkeypatch.setattr(web_research_service, "research_web_pages", research)
    return search, research


async def invoke(limit: int = 30) -> str:
    return await tools.websearch_impl(
        embedding_manager=MagicMock(),
        llm_manager=MagicMock(),
        config=MagicMock(),
        query="query",
        limit=limit,
    )


@pytest.mark.asyncio
async def test_urlerror_from_search_raises_mcperror(monkeypatch, patched) -> None:
    monkeypatch.setattr(
        websearch_core,
        "search_multi",
        AsyncMock(side_effect=urllib.error.URLError("boom")),
    )

    with pytest.raises(MCPError, match="Web search failed: boom"):
        await invoke()


@pytest.mark.asyncio
async def test_empty_results_raises_mcperror(monkeypatch, patched) -> None:
    monkeypatch.setattr(websearch_core, "search_multi", AsyncMock(return_value=[]))

    with pytest.raises(MCPError, match="No results found"):
        await invoke()


@pytest.mark.asyncio
async def test_partial_fetch_warnings_render_bullets(monkeypatch, patched) -> None:
    async def pages(urls, progress_callback=None, warning_callback=None):
        warning_callback("first\ncontinued")
        yield urls[0], ".md", "# Page"

    monkeypatch.setattr(websearch_core, "fetch_pages", pages)

    answer = await invoke()

    assert answer.startswith("ANSWER")
    assert "> - first\n> continued" in answer


@pytest.mark.asyncio
async def test_research_failure_raises_mcperror(monkeypatch, patched) -> None:
    monkeypatch.setattr(
        web_research_service,
        "research_web_pages",
        AsyncMock(side_effect=ValueError("bad research")),
    )

    with pytest.raises(MCPError, match="Web research failed: bad research"):
        await invoke()


@pytest.mark.asyncio
async def test_timeout_raises_mcperror(monkeypatch, patched) -> None:
    async def blocked(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(web_research_service, "research_web_pages", blocked)
    monkeypatch.setattr(websearch_core, "websearch_timeout", lambda: 0.01)

    with pytest.raises(MCPError, match="websearch timed out"):
        await invoke()


@pytest.mark.asyncio
async def test_cancellation_propagates_and_cancels_research(
    monkeypatch, patched
) -> None:
    cancelled = asyncio.Event()

    async def blocked(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(web_research_service, "research_web_pages", blocked)
    task = asyncio.create_task(invoke())
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_limit_clamped_to_range(patched) -> None:
    search, _ = patched

    await invoke(limit=1_000)

    assert search.await_args.args[1] == 100


@pytest.mark.asyncio
async def test_websearch_mcp_previous_query_reaches_expansion_and_research(
    monkeypatch, patched
) -> None:
    captured: dict[str, object] = {}

    async def capturing_expand(query, llm_manager, previous_query=None):
        captured["previous_query"] = previous_query
        return [query]

    async def capturing_research(query, pages, *args, **kwargs):
        captured["research_previous_query"] = kwargs.get("previous_query")
        async for _ in pages:
            pass
        return {"answer": "ANSWER"}

    monkeypatch.setattr(websearch_expansion, "expand_web_queries", capturing_expand)
    monkeypatch.setattr(web_research_service, "research_web_pages", capturing_research)

    await tools.websearch_impl(
        embedding_manager=MagicMock(),
        llm_manager=MagicMock(),
        config=MagicMock(),
        query="q",
        previous_query="prior topic",
    )

    assert captured["previous_query"] == "prior topic"
    assert captured["research_previous_query"] == "prior topic"


@pytest.mark.asyncio
async def test_websearch_mcp_empty_previous_query_treated_as_none(
    monkeypatch, patched
) -> None:
    captured: dict[str, object] = {}

    async def capturing_expand(query, llm_manager, previous_query=None):
        captured["previous_query"] = previous_query
        return [query]

    monkeypatch.setattr(websearch_expansion, "expand_web_queries", capturing_expand)

    await tools.websearch_impl(
        embedding_manager=MagicMock(),
        llm_manager=MagicMock(),
        config=MagicMock(),
        query="q",
        previous_query="",
    )

    assert captured["previous_query"] is None
