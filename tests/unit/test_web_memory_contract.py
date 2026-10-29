"""Bounded-memory and live-source contracts for web research/fetching."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.embeddings import LocalEmbeddingResult
from chunkhound.services import web_research_service as web
from chunkhound.services.research.shared.file_reader import FileReader
from chunkhound.utils import websearch_core as core


def _embedding_manager():
    manager = MagicMock()

    async def embed(texts):
        await asyncio.sleep(0)
        return LocalEmbeddingResult(
            embeddings=[[1.0, 0.0] for _ in texts],
            model="bounded-web-test",
            provider="test",
            dims=2,
            embedding_space_identity="test-space",
        )

    manager.embed_texts = AsyncMock(side_effect=embed)
    return manager


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("cache_budget", [0, 5000])
async def test_evicted_pages_support_concurrent_late_search_and_async_source(
    monkeypatch, changed, cache_budget
):
    # Unicode makes a character-count budget insufficient. At most one page
    # fits; budget zero also protects the oversized-entry/no-fallback contract.
    monkeypatch.setattr(web, "_SOURCE_CACHE_BYTES", cache_budget)
    count = 12
    refetched = []
    revision = False
    captured = {}

    def body(index):
        return f"source {index}: " + "漢字龍" * 150

    async def pages():
        for index in range(count):
            yield f"https://example.invalid/{index}", ".md", body(index)
            await asyncio.sleep(0)

    async def refetch(urls, **kwargs):
        assert len(urls) == 1
        url = urls[0]
        refetched.append(url)
        await asyncio.sleep(0)
        index = int(url.rsplit("/", 1)[1])
        yield url, ".md", body(index) + (" changed" if revision else "")

    monkeypatch.setattr(web, "fetch_pages", refetch)

    async def deep_research(query, previous_query=None):
        nonlocal revision
        services = captured["db_services"]
        provider = services.provider
        results = await asyncio.gather(
            *(
                services.search_service.search_semantic(f"query {i}", page_size=20)
                for i in range(6)
            )
        )
        results.append(
            await services.search_service.search_semantic("late", page_size=20)
        )
        for matches, _ in results:
            assert len(matches) == count
            assert {row["content"] for row in matches} == {
                body(i) for i in range(count)
            }
        assert refetched, (
            "evicted pages must be refetched, not replayed from retained chunks"
        )
        assert provider._source_bytes <= cache_budget
        assert provider.get_transient_file_content("https://example.invalid/0") is None
        assert provider.transient_source_budget_bytes == 1_000_000
        # The actual research reader must still resolve evicted sources and
        # obey a corpus-independent evidence budget, including long lines.
        provider.transient_source_budget_bytes = 800
        llm_manager = MagicMock()
        llm_manager.get_utility_provider().estimate_tokens.side_effect = len
        evidence = await FileReader(services).read_files_with_budget(
            results[0][0], llm_manager, max_tokens=500
        )
        assert evidence
        assert sum(len(text.encode("utf-8")) for text in evidence.values()) <= 800
        assert sum(len(text) for text in evidence.values()) <= 500
        assert body(0)[:20] in evidence["https://example.invalid/0"]
        revision = changed
        if changed:
            with pytest.raises(
                RuntimeError, match="Web source changed since first read"
            ):
                await services.search_service.search_semantic("changed replay")
        else:
            assert await provider.get_transient_file_content_async(
                "https://example.invalid/0"
            ) == body(0)
        return {"answer": "done"}

    def create(**kwargs):
        captured.update(kwargs)
        return MagicMock(deep_research=deep_research)

    monkeypatch.setattr(web.ResearchServiceFactory, "create", create)
    if changed:
        # Expanded research searches may swallow individual search errors.
        # A revision mismatch must still fail the overall research session.
        with pytest.raises(RuntimeError, match="Web source changed since first read"):
            await web.research_web_pages(
                "query", pages(), MagicMock(), _embedding_manager(), MagicMock()
            )
    else:
        assert await web.research_web_pages(
            "query", pages(), MagicMock(), _embedding_manager(), MagicMock()
        ) == {"answer": "done"}


@pytest.mark.asyncio
async def test_research_does_not_fetch_before_search_and_closes_suspended_input(
    monkeypatch,
):
    requested = 0
    closed = asyncio.Event()
    captured = {}

    async def pages():
        nonlocal requested
        try:
            for index in range(1000):
                requested += 1
                yield f"https://example.invalid/{index}", ".md", "body"
        finally:
            closed.set()

    async def deep_research(query, previous_query=None):
        assert requested == 0
        stream = captured["db_services"].search_service._chunk_stream()
        try:
            await anext(stream)
            assert requested == 100  # One embedding batch, not all 1000 pages.
            await asyncio.sleep(0)
            assert requested == 100
            raise asyncio.CancelledError()
        finally:
            await stream.aclose()

    def create(**kwargs):
        captured.update(kwargs)
        return MagicMock(deep_research=deep_research)

    monkeypatch.setattr(web.ResearchServiceFactory, "create", create)
    with pytest.raises(asyncio.CancelledError):
        await web.research_web_pages(
            "query", pages(), MagicMock(), _embedding_manager(), MagicMock()
        )
    assert closed.is_set()


@pytest.mark.asyncio
async def test_fetch_pages_slow_consumer_bounds_fetch_ahead_and_closes_pending(
    monkeypatch,
):
    started = []
    finished = []
    browser_closed = asyncio.Event()
    release = asyncio.Event()

    @asynccontextmanager
    async def browser(*args):
        try:
            yield None
        finally:
            browser_closed.set()

    async def fetch(url, browser):
        started.append(url)
        try:
            if url != "0":
                await release.wait()
            return ".md", f"body {url}", {}
        finally:
            finished.append(url)

    monkeypatch.setattr(core, "_managed_browser", browser)
    monkeypatch.setattr(core, "fetch_url_to_content", fetch)
    pages = core.fetch_pages([str(i) for i in range(100)])
    assert await anext(pages) == ("0", ".md", "body 0")
    await asyncio.sleep(0.01)
    assert len(started) == core._MAX_FETCH_CONCURRENCY
    assert finished == ["0"]
    await pages.aclose()
    assert sorted(started) == sorted(finished)
    assert browser_closed.is_set()


@pytest.mark.asyncio
async def test_fetch_pages_completed_window_does_not_drain_without_consumer(
    monkeypatch,
):
    fetched = []

    @asynccontextmanager
    async def browser(*args):
        yield None

    async def fetch(url, browser):
        fetched.append(url)
        return ".md", f"body {url}", {}

    monkeypatch.setattr(core, "_managed_browser", browser)
    monkeypatch.setattr(core, "fetch_url_to_content", fetch)
    pages = core.fetch_pages([str(i) for i in range(30)])
    first = await anext(pages)
    await asyncio.sleep(0.01)
    assert len(fetched) == core._MAX_FETCH_CONCURRENCY
    second = await anext(pages)
    await asyncio.sleep(0.01)
    assert len(fetched) == core._MAX_FETCH_CONCURRENCY + 1
    rest = [page async for page in pages]
    assert {page[0] for page in [first, second, *rest]} == {str(i) for i in range(30)}
