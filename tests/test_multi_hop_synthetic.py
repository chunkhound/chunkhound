"""Synthetic vector-based tests for multi-hop search algorithm mechanics.

Tests use controllable mocks to validate algorithm behavior without corpus dependency.
Each test validates ONE specific mechanic from multi_hop_strategy.py.

Key mechanics tested:
1. Initial retrieval limit (line 78): min(page_size * 3, 100)
2. Rerank index mapping (lines 121-123): rerank_result.index -> documents[]
3. Chunk-based score tracking (lines 256-265): tracks chunk_id not position
4. Score drop calculation (line 267): prev_score - current_score
5. Termination conditions (lines 154-287): time, count, quality, degradation, minimum
"""

import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from chunkhound.interfaces.embedding_provider import RerankResult
from chunkhound.services.search.multi_hop_strategy import MultiHopStrategy


@dataclass
class MockChunk:
    """Minimal chunk for synthetic testing."""

    chunk_id: int
    content: str
    file_path: str = "test.py"
    similarity: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "file_path": self.file_path,
            "similarity": self.similarity,
        }


class ControllableReranker:
    """Reranker with predetermined score outputs per call.

    Used to test reranking mechanics without real embeddings.
    Each call to rerank() returns scores from the next entry in score_schedule.
    """

    def __init__(self, score_schedule: list[list[float]]):
        """Initialize with predetermined scores.

        Args:
            score_schedule: List of score lists, one per rerank() call.
                          Each inner list has scores in document index order.
        """
        self.score_schedule = score_schedule
        self.call_count = 0
        self.calls: list[tuple[str, list[str], int | None]] = []

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        """Return pre-determined scores based on call count."""
        self.calls.append((query, documents, top_k))

        if self.call_count >= len(self.score_schedule):
            # Fallback: return descending scores
            scores = [0.9 - i * 0.05 for i in range(len(documents))]
        else:
            scores = self.score_schedule[self.call_count]
            # Pad if needed
            if len(scores) < len(documents):
                scores = list(scores) + [0.1] * (len(documents) - len(scores))

        self.call_count += 1

        # Create results with correct index mapping
        results = [
            RerankResult(index=i, score=scores[i]) for i in range(len(documents))
        ]
        results.sort(key=lambda x: x.score, reverse=True)

        if top_k is not None:
            results = results[:top_k]
        return results

    def supports_reranking(self) -> bool:
        return True


class ControllableExpander:
    """Database mock with controlled neighbor expansion.

    Used to test expansion mechanics without real database.
    Returns predetermined neighbors for each chunk_id.
    """

    def __init__(self, neighbor_schedule: dict[int, list[MockChunk]]):
        """Initialize with predetermined neighbors.

        Args:
            neighbor_schedule: Maps chunk_id -> list of neighbor MockChunks
        """
        self.neighbor_schedule = neighbor_schedule
        self.calls: list[tuple[int, str, str, int, Any, str | None]] = []

    def find_similar_chunks(
        self,
        chunk_id: int,
        provider: str,
        model: str,
        limit: int = 10,
        threshold: float | None = None,
        path_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return predetermined neighbors for chunk_id."""
        self.calls.append((chunk_id, provider, model, limit, threshold, path_filter))
        neighbors = self.neighbor_schedule.get(chunk_id, [])
        return [n.to_dict() for n in neighbors[:limit]]


@dataclass
class SingleHopCapture:
    """Captures calls to single_hop_search for verification."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    return_results: list[dict[str, Any]] = field(default_factory=list)
    return_pagination: dict[str, Any] = field(default_factory=dict)

    async def __call__(
        self,
        query: str,
        page_size: int,
        offset: int,
        threshold: float | None,
        provider: str,
        model: str,
        path_filter: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Record call and return configured results."""
        self.calls.append(
            {
                "query": query,
                "page_size": page_size,
                "offset": offset,
                "threshold": threshold,
                "provider": provider,
                "model": model,
                "path_filter": path_filter,
            }
        )
        return self.return_results, self.return_pagination


def make_chunks(count: int, base_id: int = 0) -> list[dict[str, Any]]:
    """Create count mock chunks with sequential IDs."""
    return [
        MockChunk(
            chunk_id=base_id + i,
            content=f"content_{base_id + i}",
            similarity=0.8 - i * 0.01,
        ).to_dict()
        for i in range(count)
    ]


# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def high_scores() -> list[float]:
    """Scores that pass all quality thresholds."""
    return [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45]


# =============================================================================
# Tier 1: Unit Mechanics Tests
# =============================================================================


@pytest.mark.asyncio
async def test_initial_limit_capping():
    """Initial retrieval uses min(page_size * 3, 100) cap."""
    single_hop = SingleHopCapture()
    single_hop.return_results = make_chunks(10)

    reranker = ControllableReranker([[0.9] * 10])
    expander = ControllableExpander({})

    strategy = MultiHopStrategy(expander, reranker, single_hop)

    # Test page_size=5 -> limit=15
    await strategy.search(
        "query",
        page_size=5,
        offset=0,
        threshold=None,
        provider="test",
        model="test",
        path_filter=None,
    )
    assert single_hop.calls[0]["page_size"] == 15

    # Reset and test page_size=40 -> limit=100
    single_hop.calls.clear()
    single_hop.return_results = make_chunks(10)
    await strategy.search(
        "query",
        page_size=40,
        offset=0,
        threshold=None,
        provider="test",
        model="test",
        path_filter=None,
    )
    assert single_hop.calls[0]["page_size"] == 100


@pytest.mark.asyncio
async def test_rerank_index_mapping():
    """Rerank indices correctly map to document positions."""
    chunks = make_chunks(6)
    single_hop = SingleHopCapture()
    single_hop.return_results = chunks

    # Scores in document order: [0.3, 0.9, 0.5, 0.7, 0.4, 0.6]
    reranker = ControllableReranker([[0.3, 0.9, 0.5, 0.7, 0.4, 0.6]])
    expander = ControllableExpander({})

    strategy = MultiHopStrategy(expander, reranker, single_hop)
    results, _ = await strategy.search(
        "query",
        page_size=6,
        offset=0,
        threshold=None,
        provider="test",
        model="test",
        path_filter=None,
    )

    # Chunk 1 (score 0.9) should be first
    assert results[0]["chunk_id"] == 1
    assert results[0]["score"] == 0.9


@pytest.mark.asyncio
async def test_expansion_triggers_second_rerank():
    """Expansion with new candidates triggers additional reranking."""
    chunks = make_chunks(6)
    single_hop = SingleHopCapture()
    single_hop.return_results = chunks

    reranker = ControllableReranker(
        [
            [0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
            [0.95, 0.9, 0.75, 0.7, 0.65, 0.6],  # chunk 0 drops from 0.9 to 0.75
        ]
    )

    expander = ControllableExpander(
        {
            0: [MockChunk(100, "neighbor")],
        }
    )

    strategy = MultiHopStrategy(expander, reranker, single_hop)
    await strategy.search(
        "query",
        page_size=6,
        offset=0,
        threshold=None,
        provider="test",
        model="test",
        path_filter=None,
    )

    assert reranker.call_count == 2


@pytest.mark.asyncio
async def test_score_drop_calculation():
    """Score drop >= 0.15 triggers termination."""
    chunks = make_chunks(6)
    single_hop = SingleHopCapture()
    single_hop.return_results = chunks

    reranker = ControllableReranker(
        [
            [0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
            [0.74, 0.85, 0.8, 0.75, 0.7, 0.65, 0.5],  # chunk 0 drops 0.16
        ]
    )

    expander = ControllableExpander(
        {
            0: [MockChunk(100, "neighbor")],
            1: [MockChunk(101, "neighbor")],
            2: [MockChunk(102, "neighbor")],
            3: [MockChunk(103, "neighbor")],
            4: [MockChunk(104, "neighbor")],
        }
    )

    strategy = MultiHopStrategy(expander, reranker, single_hop)
    await strategy.search(
        "query",
        page_size=6,
        offset=0,
        threshold=None,
        provider="test",
        model="test",
        path_filter=None,
    )

    assert reranker.call_count == 2


@pytest.mark.asyncio
async def test_candidate_filtering_by_positive_score():
    """Only candidates with score > 0 are considered for expansion."""
    chunks = make_chunks(6)
    single_hop = SingleHopCapture()
    single_hop.return_results = chunks

    reranker = ControllableReranker(
        [
            [0.5, 0.4, 0.3, 0.0, -0.1, -0.2],
        ]
    )

    expander = ControllableExpander({})

    strategy = MultiHopStrategy(expander, reranker, single_hop)
    await strategy.search(
        "query",
        page_size=6,
        offset=0,
        threshold=None,
        provider="test",
        model="test",
        path_filter=None,
    )

    assert expander.calls == []


# =============================================================================
# Tier 2: Termination Condition Tests
# =============================================================================


@pytest.mark.asyncio
async def test_termination_by_time():
    """Expansion terminates after 5 seconds."""
    from unittest.mock import patch

    chunks = make_chunks(10)
    single_hop = SingleHopCapture()
    single_hop.return_results = chunks

    reranker = ControllableReranker([[0.9] * 10] * 100)
    expander = ControllableExpander({
        i: [MockChunk(100 + i, f"neighbor_{i}")] for i in range(10)
    })

    strategy = MultiHopStrategy(expander, reranker, single_hop)

    call_count = [0]

    def mock_perf_counter():
        call_count[0] += 1
        if call_count[0] <= 2:
            return 0.0
        return 5.1

    with patch("chunkhound.services.search.multi_hop_strategy.time.perf_counter",
               mock_perf_counter):
        await strategy.search("query", page_size=10, offset=0, threshold=None,
                             provider="test", model="test", path_filter=None)

    assert reranker.call_count <= 2


@pytest.mark.asyncio
async def test_termination_by_result_count():
    """Expansion terminates at 500 results."""
    chunks = make_chunks(10)
    single_hop = SingleHopCapture()
    single_hop.return_results = chunks

    reranker = ControllableReranker([[0.9] * 600] * 100)

    expander = ControllableExpander({
        i: [MockChunk(1000 + i * 100 + j, f"n_{i}_{j}") for j in range(100)]
        for i in range(10)
    })

    strategy = MultiHopStrategy(expander, reranker, single_hop)
    results, pagination = await strategy.search("query", page_size=10, offset=0,
                                                threshold=None, provider="test",
                                                model="test", path_filter=None)

    assert pagination["total"] <= 510


@pytest.mark.asyncio
async def test_termination_by_insufficient_candidates():
    """Expansion terminates when < 5 candidates have score > 0."""
    chunks = make_chunks(6)
    single_hop = SingleHopCapture()
    single_hop.return_results = chunks

    reranker = ControllableReranker([
        [0.9, 0.8, 0.7, 0.6, 0.0, -0.1],
    ])

    expander = ControllableExpander({
        i: [MockChunk(100 + i, f"neighbor_{i}")] for i in range(6)
    })

    strategy = MultiHopStrategy(expander, reranker, single_hop)
    await strategy.search("query", page_size=6, offset=0, threshold=None,
                         provider="test", model="test", path_filter=None)

    assert expander.calls == []


@pytest.mark.asyncio
async def test_termination_by_score_degradation():
    """Expansion terminates when score drops >= 0.15."""
    chunks = make_chunks(10)
    single_hop = SingleHopCapture()
    single_hop.return_results = chunks

    reranker = ControllableReranker([
        [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45],
        [0.74, 0.84, 0.79, 0.74, 0.69, 0.64, 0.59, 0.54, 0.49, 0.44, 0.4],
    ])

    expander = ControllableExpander({
        i: [MockChunk(100 + i, f"neighbor_{i}")] for i in range(10)
    })

    strategy = MultiHopStrategy(expander, reranker, single_hop)
    await strategy.search("query", page_size=10, offset=0, threshold=None,
                         provider="test", model="test", path_filter=None)

    assert reranker.call_count == 2


@pytest.mark.asyncio
async def test_termination_by_minimum_relevance():
    """Expansion terminates when top-5 minimum < 0.3."""
    chunks = make_chunks(10)
    single_hop = SingleHopCapture()
    single_hop.return_results = chunks

    reranker = ControllableReranker([
        [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45],
        [0.9, 0.85, 0.8, 0.75, 0.29, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01],
    ])

    expander = ControllableExpander({
        i: [MockChunk(100 + i, f"neighbor_{i}")] for i in range(10)
    })

    strategy = MultiHopStrategy(expander, reranker, single_hop)
    await strategy.search("query", page_size=10, offset=0, threshold=None,
                         provider="test", model="test", path_filter=None)

    assert reranker.call_count == 2


# =============================================================================
# Tier 3: Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_expansion_discovers_distant_chunks():
    """Multi-hop discovers chunks unreachable by single-hop."""
    initial_chunks = make_chunks(6)
    single_hop = SingleHopCapture()
    single_hop.return_results = initial_chunks

    distant_chunk = MockChunk(200, "distant_target", similarity=0.3)

    # Rerank scores: after round 1, chunk 100 gets score 0.88 (2nd highest)
    # so it enters top-5 candidates for expansion in round 2
    reranker = ControllableReranker([
        [0.9, 0.85, 0.8, 0.75, 0.7, 0.65],  # Initial: chunks 0-5
        [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.88],  # Round 1: chunk 100 scores 0.88
        [0.9, 0.88, 0.85, 0.8, 0.75, 0.7, 0.65, 0.55],  # Round 2: chunk 200 discovered
    ])

    expander = ControllableExpander({
        0: [MockChunk(100, "hop1")],
        100: [distant_chunk],
    })

    strategy = MultiHopStrategy(expander, reranker, single_hop)
    results, _ = await strategy.search("query", page_size=20, offset=0, threshold=None,
                                       provider="test", model="test", path_filter=None)

    result_ids = [r["chunk_id"] for r in results]
    assert 200 in result_ids


@pytest.mark.asyncio
async def test_reranking_prevents_drift():
    """Reranking filters out low-relevance expansion candidates."""
    initial_chunks = make_chunks(6)
    single_hop = SingleHopCapture()
    single_hop.return_results = initial_chunks

    relevant = MockChunk(100, "relevant_to_query")
    distractor = MockChunk(101, "completely_unrelated")

    reranker = ControllableReranker([
        [0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
        [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.7, 0.1],
    ])

    expander = ControllableExpander({
        0: [relevant, distractor],
    })

    strategy = MultiHopStrategy(expander, reranker, single_hop)
    results, _ = await strategy.search("query", page_size=8, offset=0, threshold=None,
                                       provider="test", model="test", path_filter=None)

    relevant_pos = None
    distractor_pos = None
    for i, r in enumerate(results):
        if r["chunk_id"] == 100:
            relevant_pos = i
        if r["chunk_id"] == 101:
            distractor_pos = i

    assert relevant_pos is not None and distractor_pos is not None
    assert relevant_pos < distractor_pos


# =============================================================================
# Tier 4: Edge Case and Error Recovery Tests
# =============================================================================


@pytest.mark.asyncio
async def test_few_initial_results_still_reranked():
    """Even with 1-3 initial results, multi-hop reranks them."""
    chunks = make_chunks(3)  # Only 3 results
    single_hop = SingleHopCapture()
    single_hop.return_results = chunks

    reranker = ControllableReranker([
        [0.9, 0.7, 0.5],  # Rerank scores for 3 chunks
    ])
    expander = ControllableExpander({})

    strategy = MultiHopStrategy(expander, reranker, single_hop)
    results, _ = await strategy.search(
        "query",
        page_size=10,
        offset=0,
        threshold=None,
        provider="test",
        model="test",
        path_filter=None,
    )

    # Verify reranker was called (not bypassed)
    assert reranker.call_count == 1, "Reranker should be called even with few results"

    # Verify results have rerank scores applied
    assert results[0]["score"] == 0.9
    assert len(results) == 3


@pytest.mark.asyncio
async def test_rerank_failure_falls_back_to_similarity():
    """When reranking fails, similarity scores are used as fallback."""
    chunks = make_chunks(6)
    single_hop = SingleHopCapture()
    single_hop.return_results = chunks

    class FailingReranker:
        """Reranker that always raises an exception."""

        call_count = 0

        async def rerank(self, query: str, documents: list[str], top_k: int | None = None):
            self.call_count += 1
            raise RuntimeError("Reranking service unavailable")

        def supports_reranking(self) -> bool:
            return True

    reranker = FailingReranker()
    expander = ControllableExpander({})

    strategy = MultiHopStrategy(expander, reranker, single_hop)
    results, _ = await strategy.search(
        "query",
        page_size=6,
        offset=0,
        threshold=None,
        provider="test",
        model="test",
        path_filter=None,
    )

    # Verify reranker was attempted
    assert reranker.call_count == 1

    # Verify results still returned with similarity-based scores
    assert len(results) == 6
    for result in results:
        assert "score" in result
        # Score should be the similarity value (fallback)
        assert result["score"] == result["similarity"]


@pytest.mark.asyncio
async def test_expansion_failure_continues_with_other_candidates():
    """Expansion failure for one chunk doesn't stop expansion of others."""
    chunks = make_chunks(6)
    single_hop = SingleHopCapture()
    single_hop.return_results = chunks

    reranker = ControllableReranker([
        [0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
        [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6],  # After expansion adds chunk 101
    ])

    class PartiallyFailingExpander:
        """Expander that fails for specific chunk_ids."""

        def __init__(self, fail_chunk_ids: set[int], success_neighbors: dict[int, list]):
            self.fail_chunk_ids = fail_chunk_ids
            self.success_neighbors = success_neighbors
            self.calls: list[int] = []

        def find_similar_chunks(
            self,
            chunk_id: int,
            provider: str,
            model: str,
            limit: int = 10,
            threshold: float | None = None,
            path_filter: str | None = None,
        ) -> list[dict]:
            self.calls.append(chunk_id)
            if chunk_id in self.fail_chunk_ids:
                raise RuntimeError(f"Failed to expand chunk {chunk_id}")
            neighbors = self.success_neighbors.get(chunk_id, [])
            return [n.to_dict() for n in neighbors[:limit]]

    # Chunks 0 and 2 fail, but chunk 1 succeeds
    expander = PartiallyFailingExpander(
        fail_chunk_ids={0, 2, 3, 4},
        success_neighbors={1: [MockChunk(101, "neighbor_from_1")]},
    )

    strategy = MultiHopStrategy(expander, reranker, single_hop)
    results, _ = await strategy.search(
        "query",
        page_size=10,
        offset=0,
        threshold=None,
        provider="test",
        model="test",
        path_filter=None,
    )

    # Verify expansion was attempted for multiple candidates
    assert len(expander.calls) >= 2

    # Verify chunk 101 was discovered despite failures
    result_ids = [r["chunk_id"] for r in results]
    assert 101 in result_ids, "Should discover chunk from successful expansion"
