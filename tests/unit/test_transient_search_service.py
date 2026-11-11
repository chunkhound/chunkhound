"""Contracts for streaming transient semantic search."""

import asyncio
import math
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.embeddings import LocalEmbeddingResult
from chunkhound.interfaces.embedding_provider import RerankResult
from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider
from chunkhound.services.transient_search_service import TransientSearchService
from chunkhound.services.vector_cache import VectorCache
from chunkhound.utils.hashing import compute_embedding_space_identity
from tests.unit.transient_search_helpers import (
    make_chunk,
    make_original,
    stream_chunks,
)


def embedding_manager() -> MagicMock:
    class Provider:
        embedding_space_identity = compute_embedding_space_identity("test-provider")

    manager = MagicMock()
    manager.get_provider.return_value = Provider()

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
            embedding_space_identity="test-space",
        )

    manager.embed_texts = AsyncMock(side_effect=embed)
    manager.embed_query_texts = manager.embed_texts
    return manager


def reranking_manager(
    results: list[RerankResult] | BaseException,
) -> tuple[MagicMock, MagicMock]:
    manager = embedding_manager()
    provider = MagicMock()
    provider.supports_reranking.return_value = True
    if isinstance(results, BaseException):
        provider.rerank = AsyncMock(side_effect=results)
    else:
        provider.rerank = AsyncMock(return_value=results)
    manager.get_provider.return_value = provider
    return manager, provider


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
async def test_result_limit_controls_transient_pagination_window() -> None:
    chunks = [
        make_chunk(f"symbol-{i}", start_line=i + 1, code=f"chunk {i}")
        for i in range(100)
    ]
    service = TransientSearchService(
        make_original(),
        stream_chunks(chunks),
        "diff",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    results, pagination = await service.search_semantic(
        "query", page_size=10, offset=60, result_limit=100
    )

    assert len(results) == 10
    assert pagination["total"] == 100


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
@pytest.mark.parametrize(
    ("second_endpoint", "second_deployment"),
    [
        ("https://two.example/v1", "deployment-a"),
        ("https://one.example/v1", "deployment-b"),
    ],
)
async def test_shared_cache_isolates_embedding_provider_instances(
    second_endpoint: str, second_deployment: str
) -> None:
    cache = VectorCache(max_entries=10)
    chunks = [make_chunk("one", code="one")]

    class Provider:
        def __init__(self, endpoint: str, deployment: str) -> None:
            self.embedding_space_identity = compute_embedding_space_identity(
                "test-v1", endpoint, deployment
            )

    def manager(endpoint: str, deployment: str, chunk_vector: list[float]) -> MagicMock:
        embedding_provider = Provider(endpoint, deployment)
        embedding_manager = MagicMock()
        embedding_manager.get_provider.return_value = embedding_provider

        async def embed(texts: list[str]) -> LocalEmbeddingResult:
            vectors = [
                [1.0, 0.0] if text == "query" else chunk_vector for text in texts
            ]
            return LocalEmbeddingResult(
                embeddings=vectors,
                model="shared-model",
                provider="shared-provider",
                dims=2,
                embedding_space_identity=embedding_provider.embedding_space_identity,
            )

        embedding_manager.embed_texts = AsyncMock(side_effect=embed)
        embedding_manager.embed_query_texts = embedding_manager.embed_texts
        return embedding_manager

    first_manager = manager("https://one.example/v1", "deployment-a", [1.0, 0.0])
    second_manager = manager(second_endpoint, second_deployment, [0.0, 1.0])

    first_service = TransientSearchService(
        make_original(), stream_chunks(chunks), "diff", first_manager, cache
    )
    second_service = TransientSearchService(
        make_original(), stream_chunks(chunks), "diff", second_manager, cache
    )

    first_results, _ = await first_service.search_semantic("query")
    second_results, _ = await second_service.search_semantic("query")

    assert first_results[0]["similarity"] == pytest.approx(1.0)
    assert second_results[0]["similarity"] == pytest.approx(0.0)
    assert [call.args[0] for call in second_manager.embed_texts.await_args_list] == [
        ["query"],
        ["one"],
    ]


@pytest.mark.asyncio
async def test_embedding_space_change_between_query_and_chunks_fails() -> None:
    chunks = [make_chunk("one", code="one")]
    manager = MagicMock()
    identities = iter(["space-a", "space-b"])

    async def embed(texts: list[str]) -> LocalEmbeddingResult:
        return LocalEmbeddingResult(
            embeddings=[[1.0, 0.0] for _ in texts],
            model="shared-model",
            provider="test-provider",
            dims=2,
            embedding_space_identity=next(identities),
        )

    manager.embed_texts = AsyncMock(side_effect=embed)
    manager.embed_query_texts = manager.embed_texts
    service = TransientSearchService(
        make_original(), stream_chunks(chunks), "diff", manager, VectorCache(10)
    )

    with pytest.raises(RuntimeError, match="changed embedding space"):
        await service.search_semantic("query")
    assert len(service._vector_cache) == 0


@pytest.mark.asyncio
async def test_shared_cache_isolates_real_azure_provider_endpoints() -> None:
    cache = VectorCache(max_entries=10)
    chunks = [make_chunk("one", code="one")]

    def manager(endpoint: str, chunk_vector: list[float]) -> MagicMock:
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
            azure_endpoint=endpoint,
            azure_deployment="shared-deployment",
            api_version="2024-02-01",
        )
        embedding_manager = MagicMock()
        embedding_manager.get_provider.return_value = provider

        async def embed(texts: list[str]) -> LocalEmbeddingResult:
            vectors = [
                [1.0, 0.0] if text == "query" else chunk_vector for text in texts
            ]
            return LocalEmbeddingResult(
                embeddings=vectors,
                model="text-embedding-3-small",
                provider="azure_openai",
                dims=2,
                embedding_space_identity=provider.embedding_space_identity,
            )

        embedding_manager.embed_texts = AsyncMock(side_effect=embed)
        embedding_manager.embed_query_texts = embedding_manager.embed_texts
        return embedding_manager

    first_manager = manager("https://one.openai.azure.com", [1.0, 0.0])
    second_manager = manager("https://two.openai.azure.com", [0.0, 1.0])
    first_service = TransientSearchService(
        make_original(), stream_chunks(chunks), "diff", first_manager, cache
    )
    second_service = TransientSearchService(
        make_original(), stream_chunks(chunks), "diff", second_manager, cache
    )

    first_results, _ = await first_service.search_semantic("query")
    second_results, _ = await second_service.search_semantic("query")

    assert first_results[0]["similarity"] == pytest.approx(1.0)
    assert second_results[0]["similarity"] == pytest.approx(0.0)
    assert [call.args[0] for call in second_manager.embed_texts.await_args_list] == [
        ["query"],
        ["one"],
    ]


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
async def test_bulk_search_scans_source_once_and_matches_scalar_ranking() -> None:
    chunks = [make_chunk("left", code="left"), make_chunk("right", code="right")]
    stream_calls = 0

    async def counted_stream() -> AsyncIterator[list]:
        nonlocal stream_calls
        stream_calls += 1
        yield chunks

    manager = MagicMock()

    async def embed(texts: list[str]) -> LocalEmbeddingResult:
        vectors = [
            [1.0, 0.0] if text in {"left", "left query"} else [0.0, 1.0]
            for text in texts
        ]
        return LocalEmbeddingResult(
            embeddings=vectors,
            model="test",
            provider="test",
            dims=2,
            embedding_space_identity="test-space",
        )

    manager.embed_texts = AsyncMock(side_effect=embed)
    manager.embed_query_texts = manager.embed_texts
    service = TransientSearchService(
        make_original(), counted_stream, "diff", manager, VectorCache(max_entries=0)
    )

    results = await service.search_semantic_many(
        ["left query", "right query"], page_size=1
    )

    assert stream_calls == 1
    assert [result[0][0]["content"] for result in results] == ["left", "right"]
    assert [call.args[0] for call in manager.embed_texts.await_args_list] == [
        ["left query", "right query"],
        ["left", "right"],
    ]


@pytest.mark.asyncio
async def test_bulk_search_empty_query_embeddings_match_scalar_empty_results() -> None:
    manager = MagicMock()
    manager.embed_texts = AsyncMock(
        return_value=LocalEmbeddingResult(
            embeddings=[],
            model="test",
            provider="test",
            dims=0,
            embedding_space_identity="test-space",
        )
    )
    manager.embed_query_texts = manager.embed_texts
    service = TransientSearchService(
        make_original(),
        stream_chunks([make_chunk("one")]),
        "diff",
        manager,
        VectorCache(max_entries=0),
    )

    results = await service.search_semantic_many(["one", "two"])

    assert results == [
        (
            [],
            {
                "offset": 0,
                "page_size": 10,
                "has_more": False,
                "next_offset": None,
                "total": 0,
            },
        ),
        (
            [],
            {
                "offset": 0,
                "page_size": 10,
                "has_more": False,
                "next_offset": None,
                "total": 0,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_bulk_both_mode_preserves_per_query_db_merge() -> None:
    chunks = [make_chunk("left", file_path="shared.py", code="left")]
    original = make_original()

    async def db_search(**kwargs):
        query = kwargs["query"]
        results = [
            {
                "chunk_id": f"db-{query}",
                "file_path": "shared.py" if query == "left query" else "db.py",
                "start_line": 1,
                "content": f"database {query}",
                "similarity": 0.9,
                "score": 0.9,
            }
        ]
        return results, {"total": 1, "has_more": False}

    original.search_semantic = AsyncMock(side_effect=db_search)
    service = TransientSearchService(
        original,
        stream_chunks(chunks),
        "both",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    results = await service.search_semantic_many(
        ["left query", "right query"], page_size=10
    )

    assert [item["content"] for item in results[0][0]] == ["left"]
    assert {item["content"] for item in results[1][0]} == {
        "left",
        "database right query",
    }


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
            embeddings=vectors,
            model="test",
            provider="test",
            dims=2,
            embedding_space_identity="test-space",
        )

    manager.embed_texts = AsyncMock(side_effect=embed)
    manager.embed_query_texts = manager.embed_texts
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
async def test_both_mode_rejects_offset_outside_candidate_window() -> None:
    service = TransientSearchService(
        make_original(),
        stream_chunks([]),
        "both",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    with pytest.raises(ValueError, match=r"both-mode candidate window \(50\)"):
        await service.search_semantic("query", page_size=10, offset=50)


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
            embedding_space_identity="test-space",
        )

    manager.embed_texts = AsyncMock(side_effect=embed)
    manager.embed_query_texts = manager.embed_texts
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
                embedding_space_identity="test-space",
            )

        manager.embed_texts = AsyncMock(side_effect=embed)
        manager.embed_query_texts = manager.embed_texts
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
async def test_both_mode_combined_rerank_promotes_low_cosine_diff() -> None:
    chunks = [make_chunk("diff", file_path="diff.py", code="low cosine")]
    original = make_original(
        [
            {
                "chunk_id": "db-high",
                "file_path": "db.py",
                "start_line": 1,
                "content": "high cosine",
                "similarity": 0.9,
                "score": 0.9,
            }
        ]
    )
    manager, provider = reranking_manager(
        [RerankResult(index=1, score=0.95), RerankResult(index=0, score=0.1)]
    )
    service = TransientSearchService(
        original, stream_chunks(chunks), "both", manager, VectorCache(max_entries=0)
    )

    results, _ = await service.search_semantic("query", threshold=0.5)

    assert [result["content"] for result in results] == ["low cosine"]
    assert results[0]["similarity"] == pytest.approx(0.0)
    assert results[0]["score"] == pytest.approx(0.95)
    provider.rerank.assert_awaited_once_with(
        query="query", documents=["high cosine", "low cosine"], top_k=2
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rerank_results", "message"),
    [
        ([RerankResult(index=0, score=0.5)], "every candidate"),
        (
            [RerankResult(index=0, score=0.5), RerankResult(index=0, score=0.4)],
            "duplicate",
        ),
        (
            [RerankResult(index=0, score=0.5), RerankResult(index=2, score=0.4)],
            "invalid candidate index",
        ),
        (
            [RerankResult(index=0, score=0.5), RerankResult(index=1, score=math.nan)],
            "non-finite",
        ),
    ],
)
async def test_both_mode_rejects_invalid_combined_rerank(
    rerank_results: list[RerankResult], message: str
) -> None:
    manager, _ = reranking_manager(rerank_results)
    service = TransientSearchService(
        make_original(
            [
                {
                    "chunk_id": "db",
                    "file_path": "db.py",
                    "start_line": 1,
                    "content": "db",
                    "score": 0.8,
                }
            ]
        ),
        stream_chunks([make_chunk("diff", code="diff")]),
        "both",
        manager,
        VectorCache(max_entries=0),
    )

    with pytest.raises(RuntimeError, match=message):
        await service.search_semantic("query")


@pytest.mark.asyncio
async def test_both_mode_propagates_combined_rerank_failure() -> None:
    manager, _ = reranking_manager(RuntimeError("reranker unavailable"))
    service = TransientSearchService(
        make_original(), stream_chunks([make_chunk("diff")]), "both", manager
    )

    with pytest.raises(RuntimeError, match="reranker unavailable"):
        await service.search_semantic("query")


@pytest.mark.asyncio
async def test_both_mode_forced_single_hop_keeps_cosine_and_skips_rerank() -> None:
    manager, provider = reranking_manager([RerankResult(index=0, score=0.1)])
    service = TransientSearchService(
        make_original(),
        stream_chunks([make_chunk("target", code="target")]),
        "both",
        manager,
        VectorCache(max_entries=0),
    )

    results, _ = await service.search_semantic(
        "query", threshold=0.5, force_strategy="single_hop"
    )

    assert results[0]["score"] == pytest.approx(1.0)
    provider.rerank.assert_not_awaited()


@pytest.mark.asyncio
async def test_bulk_both_mode_final_rerank_preserves_per_query_failures() -> None:
    manager = embedding_manager()
    provider = MagicMock()
    provider.supports_reranking.return_value = True

    async def rerank(query: str, documents: list[str], top_k: int):
        if query == "broken":
            raise RuntimeError("query rerank failed")
        return [
            RerankResult(index=index, score=float(len(documents) - index))
            for index in range(len(documents))
        ]

    provider.rerank = AsyncMock(side_effect=rerank)
    manager.get_provider.return_value = provider
    service = TransientSearchService(
        make_original(),
        stream_chunks([make_chunk("diff", code="target")]),
        "both",
        manager,
        VectorCache(max_entries=0),
    )

    results = await service.search_semantic_many(["query", "broken"])

    assert not isinstance(results[0], BaseException)
    assert isinstance(results[1], RuntimeError)
    assert "query rerank failed" in str(results[1])


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
            embeddings=vectors,
            model="test",
            provider="test",
            dims=2,
            embedding_space_identity="test-space",
        )

    manager.embed_texts = AsyncMock(side_effect=embed)
    manager.embed_query_texts = manager.embed_texts
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
@pytest.mark.parametrize("page_size, last_offset", [(10, 1990), (30, 1950)])
async def test_large_corpus_pagination_stops_at_supported_window(
    page_size, last_offset
) -> None:
    chunks = [
        make_chunk(f"symbol-{i}", start_line=i + 1, code=f"chunk {i}")
        for i in range(2100)
    ]
    service = TransientSearchService(
        make_original(), stream_chunks(chunks), "diff", embedding_manager()
    )

    first, first_page = await service.search_semantic("query", page_size=page_size)
    last, last_page = await service.search_semantic(
        "query", page_size=page_size, offset=last_offset
    )

    assert len(first) == len(last) == page_size
    assert first_page["has_more"] is True
    assert isinstance(first_page["total"], int)
    assert last_page["has_more"] is False
    assert last_page["next_offset"] is None


@pytest.mark.asyncio
async def test_both_mode_pagination_stops_at_overfetch_pool() -> None:
    chunks = [
        make_chunk(f"symbol-{i}", start_line=i + 1, code=f"chunk {i}")
        for i in range(2100)
    ]
    service = TransientSearchService(
        make_original(), stream_chunks(chunks), "both", embedding_manager()
    )

    first, first_page = await service.search_semantic("query", page_size=10)
    last, last_page = await service.search_semantic("query", page_size=10, offset=40)

    assert len(first) == len(last) == 10
    assert first_page["has_more"] is True
    assert first_page["total"] == 50
    assert last_page["has_more"] is False
    assert last_page["next_offset"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("page_size", [3, 7, 10])
async def test_both_mode_next_offset_is_always_accepted(page_size) -> None:
    """Following ``next_offset`` must never lead to a rejected request."""
    chunks = [
        make_chunk(f"symbol-{i}", start_line=i + 1, code=f"chunk {i}")
        for i in range(60)
    ]
    service = TransientSearchService(
        make_original(),
        stream_chunks(chunks),
        "both",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    offset: int | None = 0
    while offset is not None:
        _, page = await service.search_semantic(
            "query", page_size=page_size, offset=offset
        )
        [(_, many_page)] = await service.search_semantic_many(
            ["query"], page_size=page_size, offset=offset
        )
        assert many_page["next_offset"] == page["next_offset"]
        offset = page["next_offset"]


@pytest.mark.asyncio
async def test_both_mode_rejects_partial_page_outside_candidate_window() -> None:
    service = TransientSearchService(
        make_original(), stream_chunks([]), "both", embedding_manager()
    )

    with pytest.raises(ValueError, match=r"offset\+page_size \(55\).+window \(50\)"):
        await service.search_semantic("query", page_size=10, offset=45)


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
    """Both-mode heap is overfetch-sized (50 for page_size=10), not the
    2000-hit paging window and not page_size * overfetch^2."""
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

    await service.search_semantic("query", page_size=10)

    # Default overfetch is max(10 * 5, 50) = 50, not the 2000-hit window.
    assert service.peak_live_vectors <= 50 + 100


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
    await make_service().search_semantic("query", page_size=10, offset=40)

    assert recorded_page_sizes == [50, 50]


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


@pytest.mark.asyncio
async def test_hybrid_both_warns_about_large_reranker_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[tuple[str, tuple[object, ...]]] = []

    def capture_warning(message: str, *args: object) -> None:
        warnings.append((message, args))

    monkeypatch.setattr(
        "chunkhound.services.transient_search_service.logger.warning",
        capture_warning,
    )
    service = TransientSearchService(
        make_original(),
        stream_chunks([]),
        "both",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    await service.search_hybrid("query")

    assert len(warnings) == 1
    message, args = warnings[0]
    assert "can be slow or fail" in message
    assert args == (500,)


@pytest.mark.asyncio
async def test_hybrid_both_includes_diff_db_and_regex_results() -> None:
    chunks = [make_chunk("diff", file_path="diff.py", code="target")]
    original = make_original(
        [
            {
                "chunk_id": "db-semantic",
                "file_path": "db.py",
                "start_line": 1,
                "content": "database semantic",
                "similarity": 0.9,
                "score": 0.9,
            }
        ],
        [
            {
                "chunk_id": "db-regex",
                "file_path": "regex.py",
                "start_line": 1,
                "content": "regex match",
            }
        ],
    )
    service = TransientSearchService(
        original,
        stream_chunks(chunks),
        "both",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    results, _ = await service.search_hybrid(
        "query", regex_pattern="match", page_size=10
    )

    assert {result["file_path"] for result in results} == {
        "diff.py",
        "db.py",
        "regex.py",
    }


@pytest.mark.asyncio
async def test_hybrid_both_diff_wins_duplicate_db_location() -> None:
    chunks = [
        make_chunk("diff", file_path="shared.py", start_line=7, code="historical diff")
    ]
    original = make_original(
        [
            {
                "chunk_id": "db-duplicate",
                "file_path": "shared.py",
                "start_line": 7,
                "content": "database copy",
                "similarity": 1.0,
                "score": 1.0,
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

    results, _ = await service.search_hybrid("query", page_size=10)

    assert len(results) == 1
    assert results[0]["content"] == "historical diff"


@pytest.mark.asyncio
async def test_both_mode_pruned_diff_location_suppresses_db_copy() -> None:
    chunks = [
        make_chunk(
            f"diff-{index}",
            file_path=f"diff-{index}.py",
            start_line=1,
            code=f"diff-{index}",
        )
        for index in range(50)
    ]
    chunks.append(
        make_chunk(
            "historical-owner",
            file_path="shared.py",
            start_line=7,
            code="low-scoring historical evidence",
        )
    )
    original = make_original(
        [
            {
                "chunk_id": "db-duplicate",
                "file_path": "shared.py",
                "start_line": 7,
                "content": "high-scoring working tree copy",
                "similarity": 1.0,
                "score": 1.0,
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

    results, _ = await service.search_semantic("query", page_size=10)

    assert all(result["file_path"] != "shared.py" for result in results)


@pytest.mark.asyncio
async def test_hybrid_both_uses_stable_bounded_db_snapshot_across_pages() -> None:
    requested_sizes: list[int] = []

    async def db_search(**kwargs):
        requested_sizes.append(kwargs["page_size"])
        return [], {
            "offset": 0,
            "page_size": kwargs["page_size"],
            "has_more": False,
            "next_offset": None,
            "total": 0,
        }

    def service() -> TransientSearchService:
        original = make_original()
        original.search_semantic = AsyncMock(side_effect=db_search)
        return TransientSearchService(
            original,
            stream_chunks([]),
            "both",
            embedding_manager(),
            VectorCache(max_entries=0),
        )

    await service().search_hybrid("query", page_size=10, offset=0)
    await service().search_hybrid("query", page_size=10, offset=40)

    assert requested_sizes == [500, 500]


@pytest.mark.asyncio
async def test_hybrid_both_cancels_db_snapshot_when_diff_stream_fails() -> None:
    cancelled = asyncio.Event()
    original = make_original()

    async def hanging_db(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    original.search_semantic = hanging_db

    async def broken_stream() -> AsyncIterator[list]:
        raise RuntimeError("diff stream failed")
        yield []  # pragma: no cover

    service = TransientSearchService(
        original,
        broken_stream,
        "both",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    with pytest.raises(RuntimeError, match="diff stream failed"):
        await service.search_hybrid("query")

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_hybrid_both_merges_diff_regex_match_into_semantic_result() -> None:
    chunks = [make_chunk("diff", file_path="shared.py", start_line=7, code="target")]
    service = TransientSearchService(
        make_original(),
        stream_chunks(chunks),
        "both",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    results, _ = await service.search_hybrid("query", regex_pattern="target")

    assert len(results) == 1
    assert results[0]["content"] == "target"
    assert results[0]["search_type"] == "hybrid"
    assert "semantic_score" in results[0]
    assert "regex_score" in results[0]


@pytest.mark.asyncio
async def test_hybrid_both_drops_regex_copy_at_diff_owned_location() -> None:
    chunks = [
        make_chunk("diff", file_path="shared.py", start_line=7, code="historical diff")
    ]
    stale = {
        "chunk_id": "db-stale",
        "file_path": "shared.py",
        "start_line": 7,
        "content": "stale working tree copy",
        "similarity": 1.0,
        "score": 1.0,
    }
    service = TransientSearchService(
        make_original([stale], [stale]),
        stream_chunks(chunks),
        "both",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    results, _ = await service.search_hybrid("query", regex_pattern="stale")

    assert len(results) == 1
    assert results[0]["content"] == "historical diff"


@pytest.mark.asyncio
async def test_hybrid_both_exposes_final_partial_snapshot_page() -> None:
    chunks = [
        make_chunk(f"diff-{index}", file_path=f"diff-{index}.py", code="target")
        for index in range(500)
    ]
    service = TransientSearchService(
        make_original(),
        stream_chunks(chunks),
        "both",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    first, first_page = await service.search_hybrid("query", page_size=300)
    final, final_page = await service.search_hybrid("query", page_size=300, offset=300)

    assert len(first) == 300
    assert first_page["total"] == 500
    assert first_page["has_more"] is True
    assert first_page["next_offset"] == 300
    assert len(final) == 200
    assert final_page["has_more"] is False
    assert final_page["next_offset"] is None


def _spying_embedding_manager(
    max_documents: int = 2048,
) -> tuple[MagicMock, list[list[str]]]:
    """Embed every text to a vector derived from its trailing number."""
    requests: list[list[str]] = []

    async def embed(texts: list[str]) -> LocalEmbeddingResult:
        if texts not in (["query"], ["query", "other"]):
            requests.append(list(texts))
        vectors = []
        for text in texts:
            if text in {"query", "other"}:
                vectors.append([1.0, 0.0])
            else:
                # Few distinct scores, so ties exercise source-order breaking.
                angle = (int(text.rsplit("-", 1)[1]) % 7) / 10
                vectors.append([math.cos(angle), math.sin(angle)])
        return LocalEmbeddingResult(
            embeddings=vectors,
            model="test-model",
            provider="test",
            dims=2,
            embedding_space_identity="test-space",
        )

    manager = MagicMock()
    manager.get_provider.return_value.get_max_documents_per_batch.return_value = (
        max_documents
    )
    manager.embed_texts = AsyncMock(side_effect=embed)
    manager.embed_query_texts = manager.embed_texts
    return manager, requests


def _numbered_chunks(count: int) -> list:
    return [
        make_chunk(
            f"c{index}",
            file_path=("src/api/" if index % 20 == 0 else "src/other/")
            + f"f{index}.py",
            start_line=index + 1,
            code=f"chunk-{index}",
        )
        for index in range(count)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario", ["narrow_path_filter", "partial_cache_hits", "bulk_partial_hits"]
)
async def test_chunk_embedding_requests_are_full_batches(scenario: str) -> None:
    """Cache hits and path filters must not fragment embedding requests."""
    chunks = _numbered_chunks(5_000)
    manager, requests = _spying_embedding_manager()
    cache = VectorCache(max_entries=10_000)
    service = TransientSearchService(
        make_original(), stream_chunks(chunks), "diff", manager, cache
    )
    path_filter = "src/api" if scenario == "narrow_path_filter" else None
    if scenario != "narrow_path_filter":
        # Warm every tenth chunk so each 100-chunk stream batch has 90 misses.
        warm = TransientSearchService(
            make_original(), stream_chunks(chunks[::10]), "diff", manager, cache
        )
        await warm.search_semantic("query")
        requests.clear()

    if scenario == "bulk_partial_hits":
        await service.search_semantic_many(["query", "other"])
    else:
        await service.search_semantic("query", path_filter=path_filter)

    assert requests
    assert all(len(request) == 100 for request in requests[:-1])
    expected = 250 if scenario == "narrow_path_filter" else 4_500
    assert sum(len(request) for request in requests) == expected


@pytest.mark.asyncio
async def test_bulk_query_embeddings_respect_provider_document_limit() -> None:
    requests: list[list[str]] = []

    async def embed_queries(texts: list[str]) -> LocalEmbeddingResult:
        requests.append(list(texts))
        return LocalEmbeddingResult(
            embeddings=[[1.0, 0.0] for _ in texts],
            model="test-model",
            provider="test",
            dims=2,
            embedding_space_identity="test-space",
        )

    manager = MagicMock()
    manager.get_provider.return_value.get_max_documents_per_batch.return_value = 1
    manager.embed_query_texts = AsyncMock(side_effect=embed_queries)
    service = TransientSearchService(
        make_original(), stream_chunks([]), "diff", manager, VectorCache(max_entries=0)
    )

    results = await service.search_semantic_many(["first", "second", "third"])

    assert requests == [["first"], ["second"], ["third"]]
    assert len(results) == 3


@pytest.mark.asyncio
async def test_chunk_embedding_requests_respect_provider_document_limit() -> None:
    manager, requests = _spying_embedding_manager(max_documents=64)
    service = TransientSearchService(
        make_original(),
        stream_chunks(_numbered_chunks(500)),
        "diff",
        manager,
        VectorCache(max_entries=0),
    )
    await service.search_semantic("query")
    assert max(len(request) for request in requests) == 64


@pytest.mark.asyncio
async def test_partial_cache_hits_preserve_ranking_and_tie_order() -> None:
    chunks = _numbered_chunks(1_000)

    async def pages(warm_every: int | None) -> list[list[tuple[str, int]]]:
        manager, _ = _spying_embedding_manager()
        cache = VectorCache(max_entries=0 if warm_every is None else 10_000)
        if warm_every is not None:
            await TransientSearchService(
                make_original(),
                stream_chunks(chunks[::warm_every]),
                "diff",
                manager,
                cache,
            ).search_semantic("query")
        service = TransientSearchService(
            make_original(), stream_chunks(chunks), "diff", manager, cache
        )
        return [
            [
                (row["content"], row["start_line"])
                for row in (
                    await service.search_semantic("query", page_size=25, offset=offset)
                )[0]
            ]
            for offset in (0, 25)
        ]

    baseline = await pages(None)
    assert await pages(3) == baseline
    assert await pages(1) == baseline


@pytest.mark.asyncio
async def test_diff_semantic_pagination_continues_past_retained_pool() -> None:
    chunks = [
        make_chunk(f"symbol-{i}", start_line=i + 1, code=f"chunk {i}")
        for i in range(60)
    ]
    service = TransientSearchService(
        make_original(),
        stream_chunks(chunks),
        "diff",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    results, pagination = await service.search_semantic(
        "query", page_size=10, offset=40
    )
    assert len(results) == 10
    assert pagination["has_more"] is True
    assert pagination["next_offset"] == 50

    next_results, next_pagination = await service.search_semantic(
        "query", page_size=10, offset=pagination["next_offset"]
    )
    assert len(next_results) == 10
    assert next_pagination["has_more"] is False

    many = await service.search_semantic_many(
        ["query"], page_size=10, offset=40, result_limit=5
    )
    assert len(many[0][0]) == 10
    assert many[0][1]["has_more"] is True
    assert many[0][1]["next_offset"] == 50

    bulk_next = await service.search_semantic_many(
        ["query"], page_size=10, offset=many[0][1]["next_offset"], result_limit=5
    )
    assert len(bulk_next[0][0]) == 10
    assert bulk_next[0][1]["has_more"] is False


@pytest.mark.asyncio
async def test_transient_regex_times_out_on_catastrophic_backtracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user pattern cannot outlive the transient-search circuit breaker."""
    import chunkhound.services.transient_search_service as tss_module

    monkeypatch.setattr(tss_module, "_TRANSIENT_SEARCH_TIMEOUT_SECONDS", 0.05)
    service = TransientSearchService(
        make_original(),
        stream_chunks([make_chunk("slow", code="a" * 10_000 + "!")]),
        "diff",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(TimeoutError, match="regex search timed out"):
        await service.search_regex_async(r"^(a+)+$")

    assert loop.time() - started < 0.5


@pytest.mark.asyncio
async def test_invalid_transient_regex_raises_value_error() -> None:
    service = TransientSearchService(
        make_original(),
        stream_chunks([make_chunk("diff", code="x")]),
        "diff",
        embedding_manager(),
        VectorCache(max_entries=0),
    )

    with pytest.raises(ValueError, match="Invalid regex pattern"):
        await service.search_regex_async("(unclosed")
