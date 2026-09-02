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
    assert pagination == {
        "offset": 0,
        "page_size": 10,
        "has_more": True,
        "next_offset": 10,
        "total": 5_000,
    }

    final_results, final_page = await service.search_semantic(
        "query", page_size=10, offset=4_990
    )

    assert len(final_results) == 10
    assert final_page == {
        "offset": 4_990,
        "page_size": 10,
        "has_more": False,
        "next_offset": None,
        "total": 5_000,
    }


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


@pytest.mark.asyncio
async def test_peak_live_vectors_stays_within_heap_plus_batch() -> None:
    chunks = [
        make_chunk(f"symbol-{i}", start_line=i + 1, code=f"chunk {i}")
        for i in range(5_000)
    ]
    service = TransientSearchService(
        make_original(),
        stream_chunks(chunks, batch_size=100),
        "diff",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    await service.search_semantic("query", page_size=10)

    assert service.peak_live_vectors <= 50 + 100


@pytest.mark.asyncio
async def test_equal_scores_prefer_earlier_ordinal() -> None:
    chunks = [
        make_chunk("first", code="target-one"),
        make_chunk("second", code="target-two"),
    ]
    manager = MagicMock()

    async def embed(texts: list[str]) -> LocalEmbeddingResult:
        return LocalEmbeddingResult(
            embeddings=[[1.0, 0.0] for _ in texts],
            model="test-model",
            provider="test",
            dims=2,
        )

    manager.embed_texts = AsyncMock(side_effect=embed)
    service = TransientSearchService(
        make_original(),
        stream_chunks(chunks),
        "diff",
        manager,
        VectorCache(max_entries=0),
    )

    results, _ = await service.search_semantic("query", page_size=10)

    assert [result["symbol"] for result in results] == ["first", "second"]


@pytest.mark.asyncio
async def test_cache_does_not_reuse_vectors_across_embedding_spaces() -> None:
    cache = VectorCache(max_entries=10)
    chunks = [make_chunk("one", code="shared-text")]

    async def manager_for(provider: str, dims: int, vector: list[float]) -> MagicMock:
        manager = MagicMock()

        async def embed(texts: list[str]) -> LocalEmbeddingResult:
            return LocalEmbeddingResult(
                embeddings=[vector[:dims] for _ in texts],
                model="model",
                provider=provider,
                dims=dims,
            )

        manager.embed_texts = AsyncMock(side_effect=embed)
        return manager

    first = await manager_for("alpha", 2, [1.0, 0.0])
    second = await manager_for("beta", 3, [1.0, 0.0, 0.0])
    await TransientSearchService(
        make_original(), stream_chunks(chunks), "diff", first, cache
    ).search_semantic("query")
    await TransientSearchService(
        make_original(), stream_chunks(chunks), "diff", second, cache
    ).search_semantic("query")

    assert first.embed_texts.await_count == 2
    assert second.embed_texts.await_count == 2


@pytest.mark.asyncio
async def test_lower_scoring_diff_wins_same_file_line() -> None:
    chunks = [make_chunk("fn", file_path="src/foo.py", start_line=10, code="diff-body")]
    original = make_original(
        [
            {
                "file_path": "src/foo.py",
                "start_line": 10,
                "content": "db-body",
                "score": 0.99,
                "similarity": 0.99,
            }
        ]
    )
    manager = MagicMock()

    async def embed(texts: list[str]) -> LocalEmbeddingResult:
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if text == "query" else [0.0, 1.0])
        return LocalEmbeddingResult(
            embeddings=vectors, model="test", provider="test", dims=2
        )

    manager.embed_texts = AsyncMock(side_effect=embed)
    service = TransientSearchService(
        original, stream_chunks(chunks), "both", manager, VectorCache(max_entries=0)
    )

    results, _ = await service.search_semantic("query", page_size=10)

    assert len(results) == 1
    assert results[0]["content"] == "diff-body"


@pytest.mark.asyncio
async def test_both_mode_cancels_sibling_on_failure() -> None:
    cancelled = asyncio.Event()
    original = make_original()

    async def hang(**kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    original.search_semantic = hang

    async def boom() -> AsyncIterator[list]:
        raise RuntimeError("stream failed")
        yield []  # pragma: no cover

    service = TransientSearchService(
        original, boom, "both", embedding_manager(), VectorCache(max_entries=0)
    )

    with pytest.raises(RuntimeError, match="stream failed"):
        await service.search_semantic("query")

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_hybrid_freezes_initial_source_total() -> None:
    original = make_original()
    calls = 0

    async def growing_search(**kwargs):
        nonlocal calls
        calls += 1
        page_size = kwargs["page_size"]
        total = page_size + 1
        results = [
            {
                "chunk_id": f"db-{index}",
                "file_path": f"db-{index}.py",
                "similarity": 1.0 - index / 100,
            }
            for index in range(page_size)
        ]
        return results, {
            "offset": 0,
            "page_size": page_size,
            "has_more": True,
            "next_offset": page_size,
            "total": total,
        }

    original.search_semantic = AsyncMock(side_effect=growing_search)
    service = TransientSearchService(
        original,
        stream_chunks([]),
        "db",
        embedding_manager(),
    )

    results, pagination = await service.search_hybrid("query", page_size=1)

    assert calls == 2
    assert len(results) == 1
    assert pagination["total"] == 3
