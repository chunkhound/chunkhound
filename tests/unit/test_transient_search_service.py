"""Contracts for streaming transient semantic search."""

import asyncio
import math
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.embeddings import LocalEmbeddingResult
from chunkhound.services.transient_search_service import TransientSearchService
from chunkhound.services.vector_cache import VectorCache
from tests.unit.test_diff_aware_search_service import make_chunk, make_original


def stream_chunks(chunks, batch_size: int = 100) -> object:
    async def stream() -> AsyncIterator[list]:
        for start in range(0, len(chunks), batch_size):
            await asyncio.sleep(0)
            yield chunks[start : start + batch_size]

    return stream


def embedding_manager() -> MagicMock:
    manager = MagicMock()

    async def embed(texts: list[str]) -> LocalEmbeddingResult:
        vectors = []
        for text in texts:
            if text in {"query", "target"} or text.endswith("4999"):
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return LocalEmbeddingResult(
            embeddings=vectors,
            model="test-model",
            provider="test",
            dims=2,
        )

    manager.embed_texts = AsyncMock(side_effect=embed)
    return manager


@pytest.mark.asyncio
async def test_streams_5000_chunks_into_bounded_top_page() -> None:
    chunks = [
        make_chunk(f"symbol-{i}", start_line=i + 1, code=f"chunk {i}")
        for i in range(5_000)
    ]
    service = TransientSearchService(
        make_original(),
        stream_chunks(chunks),
        "diff",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    results, pagination = await service.search_semantic("query", page_size=10)

    assert len(results) == 10
    assert results[0]["content"] == "chunk 4999"
    assert pagination["total"] <= 50


@pytest.mark.asyncio
async def test_second_search_reuses_cached_chunk_embeddings() -> None:
    chunks = [make_chunk("one", code="one"), make_chunk("two", code="two")]
    manager = embedding_manager()
    service = TransientSearchService(
        make_original(),
        stream_chunks(chunks),
        "diff",
        manager,
        VectorCache(max_entries=10),
    )

    await service.search_semantic("query")
    await service.search_semantic("query")

    embedded_batches = [call.args[0] for call in manager.embed_texts.await_args_list]
    assert embedded_batches == [["query"], ["one", "two"], ["query"]]


@pytest.mark.asyncio
async def test_disabled_cache_preserves_ranking() -> None:
    chunks = [make_chunk("one", code="one"), make_chunk("target", code="target")]

    async def ranked(cache: VectorCache) -> list[str]:
        service = TransientSearchService(
            make_original(),
            stream_chunks(chunks),
            "diff",
            embedding_manager(),
            cache,
        )
        results, _ = await service.search_semantic("query")
        return [result["content"] for result in results]

    assert await ranked(VectorCache(max_entries=0)) == await ranked(
        VectorCache(max_entries=10)
    )


@pytest.mark.asyncio
async def test_concurrent_searches_do_not_cross_results() -> None:
    chunks = [make_chunk("left", code="left"), make_chunk("right", code="right")]
    manager = MagicMock()

    async def embed(texts: list[str]) -> LocalEmbeddingResult:
        await asyncio.sleep(0)
        vectors = []
        for text in texts:
            angle = 0.0 if text in {"left", "left query"} else math.pi / 2
            vectors.append([math.cos(angle), math.sin(angle)])
        return LocalEmbeddingResult(
            embeddings=vectors, model="test", provider="test", dims=2
        )

    manager.embed_texts = AsyncMock(side_effect=embed)
    service = TransientSearchService(
        make_original(),
        stream_chunks(chunks, batch_size=1),
        "diff",
        manager,
        VectorCache(max_entries=10),
    )

    left, right = await asyncio.gather(
        service.search_semantic("left query", page_size=1),
        service.search_semantic("right query", page_size=1),
    )

    assert left[0][0]["content"] == "left"
    assert right[0][0]["content"] == "right"
