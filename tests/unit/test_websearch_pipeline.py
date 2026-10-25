"""Contracts for the split websearch discovery/research budget."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from chunkhound.utils.websearch_pipeline import (
    run_websearch,
    websearch_discovery_timeout,
)


def test_discovery_timeout_is_a_quarter_capped_at_60() -> None:
    assert websearch_discovery_timeout(600) == 60.0
    assert websearch_discovery_timeout(80) == 20.0
    assert websearch_discovery_timeout(0.04) == 0.01


@pytest.mark.asyncio
async def test_discovery_timeout_does_not_wait_for_full_budget(monkeypatch) -> None:
    async def blocked(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(
        "chunkhound.utils.websearch_expansion.expand_web_queries", blocked
    )

    with pytest.raises(TimeoutError, match="waiting for search results"):
        await run_websearch(
            "q",
            limit=1,
            config=None,
            embedding_manager=None,
            llm_manager=None,
            timeout_s=0.08,
        )


@pytest.mark.asyncio
async def test_research_timeout_uses_remaining_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        "chunkhound.utils.websearch_expansion.expand_web_queries",
        AsyncMock(return_value=["q"]),
    )
    monkeypatch.setattr(
        "chunkhound.utils.websearch_core.search_multi",
        AsyncMock(return_value=[("T", "https://example.invalid/p", "s")]),
    )

    async def pages(*args, **kwargs):
        yield "https://example.invalid/p", ".md", "# p"

    async def blocked(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr("chunkhound.utils.websearch_core.fetch_pages", pages)
    monkeypatch.setattr(
        "chunkhound.services.web_research_service.research_web_pages", blocked
    )

    with pytest.raises(TimeoutError, match="websearch timed out after 0s"):
        await run_websearch(
            "q",
            limit=1,
            config=None,
            embedding_manager=None,
            llm_manager=None,
            timeout_s=0.08,
        )


@pytest.mark.asyncio
async def test_caller_cancel_cancels_research(monkeypatch) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    monkeypatch.setattr(
        "chunkhound.utils.websearch_expansion.expand_web_queries",
        AsyncMock(return_value=["q"]),
    )
    monkeypatch.setattr(
        "chunkhound.utils.websearch_core.search_multi",
        AsyncMock(return_value=[("T", "https://example.invalid/p", "s")]),
    )

    async def pages(*args, **kwargs):
        yield "https://example.invalid/p", ".md", "# p"

    async def blocked(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr("chunkhound.utils.websearch_core.fetch_pages", pages)
    monkeypatch.setattr(
        "chunkhound.services.web_research_service.research_web_pages", blocked
    )

    task = asyncio.create_task(
        run_websearch(
            "q",
            limit=1,
            config=None,
            embedding_manager=None,
            llm_manager=None,
            timeout_s=30,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
