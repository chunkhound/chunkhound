"""Contracts for the split websearch discovery/research budget."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from chunkhound.utils.websearch_pipeline import (
    NoPagesFetchedError,
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
async def test_discovery_timeout_returns_promptly_and_cancels_discovery(
    monkeypatch,
) -> None:
    cancelled = asyncio.Event()

    async def blocked(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(
        "chunkhound.utils.websearch_expansion.expand_web_queries", blocked
    )

    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(TimeoutError, match="waiting for search results"):
        await run_websearch(
            "q",
            limit=1,
            config=None,
            embedding_manager=None,
            llm_manager=None,
            timeout_s=0.08,
        )

    assert loop.time() - started < 0.15
    await asyncio.wait_for(cancelled.wait(), timeout=1)


@pytest.mark.asyncio
async def test_all_fetch_failures_stop_before_research(monkeypatch) -> None:
    monkeypatch.setattr(
        "chunkhound.utils.websearch_expansion.expand_web_queries",
        AsyncMock(return_value=["q"]),
    )
    monkeypatch.setattr(
        "chunkhound.utils.websearch_core.search_multi",
        AsyncMock(return_value=[("T", "https://example.invalid/p", "s")]),
    )

    async def no_pages(*args, **kwargs):
        if False:
            yield

    research = AsyncMock()
    monkeypatch.setattr("chunkhound.utils.websearch_core.fetch_pages", no_pages)
    monkeypatch.setattr(
        "chunkhound.services.web_research_service.research_web_pages", research
    )

    with pytest.raises(NoPagesFetchedError, match="No pages could be fetched"):
        await run_websearch(
            "q",
            limit=1,
            config=None,
            embedding_manager=None,
            llm_manager=None,
            timeout_s=30,
        )

    research.assert_not_awaited()


@pytest.mark.asyncio
async def test_first_page_fetch_timeout_returns_promptly_and_closes_stream(
    monkeypatch,
) -> None:
    close_started = asyncio.Event()
    close_finished = asyncio.Event()

    monkeypatch.setattr(
        "chunkhound.utils.websearch_expansion.expand_web_queries",
        AsyncMock(return_value=["q"]),
    )
    monkeypatch.setattr(
        "chunkhound.utils.websearch_core.search_multi",
        AsyncMock(return_value=[("T", "https://example.invalid/p", "s")]),
    )

    async def blocked_pages(*args, **kwargs):
        try:
            await asyncio.Event().wait()
            yield "https://example.invalid/p", ".md", "# p"
        finally:
            close_started.set()
            await asyncio.sleep(0.1)
            close_finished.set()

    monkeypatch.setattr("chunkhound.utils.websearch_core.fetch_pages", blocked_pages)

    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(TimeoutError, match="websearch timed out after 0s"):
        await run_websearch(
            "q",
            limit=1,
            config=None,
            embedding_manager=None,
            llm_manager=None,
            timeout_s=0.05,
        )

    assert loop.time() - started < 0.15
    await asyncio.wait_for(close_started.wait(), timeout=1)
    await asyncio.wait_for(close_finished.wait(), timeout=1)


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
async def test_timeout_does_not_wait_for_slow_page_cleanup(monkeypatch) -> None:
    close_started = asyncio.Event()
    close_finished = asyncio.Event()

    monkeypatch.setattr(
        "chunkhound.utils.websearch_expansion.expand_web_queries",
        AsyncMock(return_value=["q"]),
    )
    monkeypatch.setattr(
        "chunkhound.utils.websearch_core.search_multi",
        AsyncMock(return_value=[("T", "https://example.invalid/p", "s")]),
    )

    async def pages(*args, **kwargs):
        try:
            yield "https://example.invalid/p", ".md", "# p"
            await asyncio.Event().wait()
        finally:
            close_started.set()
            await asyncio.sleep(0.2)
            close_finished.set()

    async def blocked(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr("chunkhound.utils.websearch_core.fetch_pages", pages)
    monkeypatch.setattr(
        "chunkhound.services.web_research_service.research_web_pages", blocked
    )

    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(TimeoutError, match="websearch timed out after 0s"):
        await run_websearch(
            "q",
            limit=1,
            config=None,
            embedding_manager=None,
            llm_manager=None,
            timeout_s=0.05,
        )

    assert loop.time() - started < 0.15
    await asyncio.wait_for(close_started.wait(), timeout=1)
    await asyncio.wait_for(close_finished.wait(), timeout=1)


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


@pytest.mark.asyncio
async def test_no_pages_fetched_error_carries_fetch_failure_reasons(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "chunkhound.utils.websearch_expansion.expand_web_queries",
        AsyncMock(return_value=["q"]),
    )
    monkeypatch.setattr(
        "chunkhound.utils.websearch_core.search_multi",
        AsyncMock(return_value=[("T", "https://example.invalid/p", "s")]),
    )

    async def failing(urls, progress, warn):
        warn("Failed to fetch https://example.invalid/p: blocked by SSRF guard")
        if False:
            yield

    monkeypatch.setattr("chunkhound.utils.websearch_core.fetch_pages", failing)

    with pytest.raises(NoPagesFetchedError, match="blocked by SSRF guard"):
        await run_websearch(
            "q",
            limit=1,
            config=None,
            embedding_manager=None,
            llm_manager=None,
            timeout_s=30,
        )


@pytest.mark.asyncio
async def test_provider_timeout_is_not_reported_as_budget_timeout(
    monkeypatch,
) -> None:
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

    async def provider_timeout(*args, **kwargs):
        raise TimeoutError("provider read timed out")

    monkeypatch.setattr("chunkhound.utils.websearch_core.fetch_pages", pages)
    monkeypatch.setattr(
        "chunkhound.services.web_research_service.research_web_pages",
        provider_timeout,
    )

    with pytest.raises(TimeoutError, match="provider read timed out") as info:
        await run_websearch(
            "q",
            limit=1,
            config=None,
            embedding_manager=None,
            llm_manager=None,
            timeout_s=30,
        )

    assert hasattr(info.value, "websearch_warnings")
