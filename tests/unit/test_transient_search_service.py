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

    # The last page of this 5,000-chunk corpus sits beyond the pagination
    # window (see test_deep_offset_rejected_beyond_max_result_window): exact
    # deep-offset ranking would require holding the whole corpus in the heap,
    # which defeats the O(K) memory bound this service exists to provide.
    with pytest.raises(ValueError, match="offset\\+page_size"):
        await service.search_semantic("query", page_size=10, offset=4_990)


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
async def test_deep_offset_rejected_beyond_max_result_window() -> None:
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

    with pytest.raises(ValueError, match="offset\\+page_size"):
        await service.search_semantic("query", page_size=10, offset=1_995)


@pytest.mark.asyncio
async def test_deep_offset_within_window_keeps_heap_bounded() -> None:
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

    # Largest offset still inside the window: heap capacity is bounded by the
    # window ceiling, not by the 5,000-chunk corpus size.
    await service.search_semantic("query", page_size=10, offset=1_990)

    assert service.peak_live_vectors <= 2_000 + 100


@pytest.mark.asyncio
async def test_both_mode_deep_offset_rejected_beyond_max_result_window() -> None:
    chunks = [
        make_chunk(f"symbol-{i}", start_line=i + 1, code=f"chunk {i}")
        for i in range(5_000)
    ]
    service = TransientSearchService(
        make_original(),
        stream_chunks(chunks, batch_size=100),
        "both",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    with pytest.raises(ValueError, match="offset\\+page_size"):
        await service.search_semantic("query", page_size=10, offset=1_995)


@pytest.mark.asyncio
async def test_search_transient_times_out_on_stalled_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled/pathologically slow source must fail with a clear timeout
    instead of hanging the caller indefinitely -- the old MAX_DIFF_CHUNKS cap
    and its paired embedding timeout are gone, so this is the only remaining
    circuit breaker against an unbounded diff/web search call."""
    import chunkhound.services.transient_search_service as tss_module

    monkeypatch.setattr(tss_module, "_TRANSIENT_SEARCH_TIMEOUT_SECONDS", 0.05)

    async def stalled_stream() -> AsyncIterator[list]:
        await asyncio.Event().wait()
        yield []  # pragma: no cover

    service = TransientSearchService(
        make_original(),
        stalled_stream,
        "diff",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    with pytest.raises(TimeoutError, match="timed out"):
        await service.search_semantic("query", page_size=10)


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
async def test_late_winner_preserves_tie_order_across_page_boundaries() -> None:
    chunks = [
        make_chunk(f"symbol-{i}", start_line=i + 1, code=f"chunk {i}")
        for i in range(60)
    ] + [make_chunk("winner", start_line=61, code="target")]
    service = TransientSearchService(
        make_original(), stream_chunks(chunks), "diff", embedding_manager()
    )

    contents = []
    for offset in range(0, 61, 10):
        results, _ = await service.search_semantic("query", page_size=10, offset=offset)
        contents.extend(result["content"] for result in results)

    assert contents == ["target"] + [f"chunk {i}" for i in range(60)]


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
@pytest.mark.parametrize("source", ["diff", "both"])
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("page_size, last_offset", [(10, 1990), (30, 1950)])
async def test_large_corpus_pagination_stops_at_supported_window(
    source, hybrid, page_size, last_offset
) -> None:
    chunks = [
        make_chunk(f"symbol-{i}", start_line=i + 1, code=f"chunk {i}")
        for i in range(2100)
    ]
    service = TransientSearchService(
        make_original(), stream_chunks(chunks), source, embedding_manager()
    )
    search = service.search_hybrid if hybrid else service.search_semantic

    first, first_page = await search("query", page_size=page_size)
    last, last_page = await search("query", page_size=page_size, offset=last_offset)

    assert len(first) == len(last) == page_size
    assert first_page["has_more"] is True
    assert isinstance(first_page["total"], int)
    assert last_page["has_more"] is False
    assert last_page["next_offset"] is None


@pytest.mark.asyncio
async def test_both_mode_has_no_continuation_for_filtered_out_corpus() -> None:
    chunks = [
        make_chunk(f"symbol-{i}", start_line=i + 1, code=f"chunk {i}")
        for i in range(2100)
    ]
    service = TransientSearchService(
        make_original(), stream_chunks(chunks), "both", embedding_manager()
    )

    results, page = await service.search_semantic("query", threshold=0.5)

    assert results == []
    assert page["total"] == 0
    assert page["has_more"] is False
    assert page["next_offset"] is None


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


@pytest.mark.asyncio
async def test_both_mode_heap_limit_not_compounded() -> None:
    """Regression guard: the diff-side heap in "both" mode must stay bounded
    by the fixed snapshot size, not compound the overfetch formula twice
    (previously up to page_size * _OVERFETCH_FACTOR**2 == 50,000 for this
    page_size, contradicting the documented O(K) memory bound)."""
    chunks = [
        make_chunk(f"symbol-{i}", start_line=i + 1, code=f"chunk {i}")
        for i in range(5_000)
    ]
    service = TransientSearchService(
        make_original(),
        stream_chunks(chunks, batch_size=100),
        "both",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    await service.search_semantic("query", page_size=2_000)

    assert service.peak_live_vectors <= 2_000 + 100


@pytest.mark.asyncio
async def test_both_mode_returns_requested_page_size_not_fetch_size() -> None:
    chunks = [
        make_chunk(f"symbol-{i}", start_line=i + 1, code=f"chunk {i}")
        for i in range(200)
    ]
    service = TransientSearchService(
        make_original(),
        stream_chunks(chunks, batch_size=50),
        "both",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    results, pagination = await service.search_semantic("query", page_size=2)

    assert len(results) == 2
    assert pagination["page_size"] == 2


@pytest.mark.asyncio
async def test_both_mode_db_fetch_size_stable_across_offsets() -> None:
    """The candidate pool requested from the DB side must not grow with the
    caller's offset. TransientSearchService is constructed fresh on every
    MCP/CLI call (verified: no instance persists across pages), so pagination
    consistency across pages depends entirely on every call for the same
    query fetching an identically-sized candidate pool from the DB."""
    recorded_page_sizes: list[int] = []

    async def recording_search(**kwargs: object) -> tuple[list, dict]:
        recorded_page_sizes.append(kwargs["page_size"])  # type: ignore[arg-type]
        return [], {
            "offset": 0,
            "page_size": kwargs["page_size"],
            "has_more": False,
            "next_offset": None,
            "total": 0,
        }

    def make_service() -> TransientSearchService:
        original = make_original()
        original.search_semantic = AsyncMock(side_effect=recording_search)
        return TransientSearchService(
            original,
            stream_chunks([]),
            "both",
            embedding_manager(),
            VectorCache(max_entries=0),
        )

    await make_service().search_semantic("query", page_size=10, offset=0)
    await make_service().search_semantic("query", page_size=10, offset=190)

    assert len(recorded_page_sizes) == 2
    assert recorded_page_sizes[0] == recorded_page_sizes[1]


@pytest.mark.asyncio
async def test_both_mode_pagination_fields_unaffected_by_fixed_fetch_size() -> None:
    chunks = [make_chunk("fn", code="diff-body")]
    original = make_original(
        [
            {
                "file_path": "src/bar.py",
                "start_line": 1,
                "content": "db-body",
                "score": 0.5,
                "similarity": 0.5,
            }
        ]
    )
    service = TransientSearchService(
        original,
        stream_chunks(chunks),
        "both",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    results, pagination = await service.search_semantic("query", page_size=10, offset=0)

    # Total must reflect the real merged count (one diff result, one db
    # result), not the fixed internal snapshot/candidate-pool size.
    assert pagination["total"] == 2
    assert pagination["has_more"] is False
    assert pagination["next_offset"] is None
