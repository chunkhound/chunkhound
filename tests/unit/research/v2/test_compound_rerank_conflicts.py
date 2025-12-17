"""Edge case tests for compound reranking score conflicts in Phase 3 synthesis.

Tests various scenarios where ROOT query and gap queries produce conflicting
reranking scores, ensuring that averaging is correctly applied per spec lines
465-479 in docs/algorithm-coverage-first-research.md.

Key test scenarios:
- ROOT high, gap low (averaged)
- ROOT low, gap high (averaged)
- Multiple gap queries with varying scores
- Zero scores from gap queries (ensure chunks not unfairly dropped)
"""

import pytest

from chunkhound.interfaces.embedding_provider import RerankResult
from chunkhound.services.research.shared.models import ResearchContext
from chunkhound.services.research.shared.unified_search import UnifiedSearch
from tests.fixtures.fake_providers import FakeEmbeddingProvider


@pytest.fixture
def controlled_rerank_provider():
    """Create fake embedding provider with controllable reranking scores.

    Returns a provider with a custom rerank function that can be overridden
    by tests to return specific scores per query.
    """
    provider = FakeEmbeddingProvider(dims=1536)

    # Store score mapping: {query_substring: {doc_index: score}}
    provider._score_map = {}

    async def controlled_rerank(query, documents, top_k=None):
        """Return controlled scores based on query and document index."""
        results = []

        # Find matching score map based on query substring
        score_dict = None
        for query_substring, scores in provider._score_map.items():
            if query_substring.lower() in query.lower():
                score_dict = scores
                break

        # Generate scores
        for idx in range(len(documents)):
            if score_dict and idx in score_dict:
                score = score_dict[idx]
            else:
                # Fallback to deterministic scoring
                query_vec = provider._generate_deterministic_vector(query)
                doc_vec = provider._generate_deterministic_vector(documents[idx])
                score = sum(a * b for a, b in zip(query_vec, doc_vec))

            results.append(RerankResult(index=idx, score=score))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        # Apply top_k if specified
        if top_k is not None:
            results = results[:top_k]

        return results

    provider.rerank = controlled_rerank
    return provider


@pytest.fixture
def embedding_manager(controlled_rerank_provider):
    """Create embedding manager with controlled rerank provider."""

    class MockEmbeddingManager:
        def get_provider(self):
            return controlled_rerank_provider

    return MockEmbeddingManager()


@pytest.fixture
def db_services(controlled_rerank_provider):
    """Create mock database services with search service."""

    class MockSearchService:
        """Mock search service that returns deterministic chunks."""

        def __init__(self, embedding_provider):
            self.embedding_provider = embedding_provider

        async def search_semantic(
            self,
            query: str,
            page_size: int = 10,
            threshold: float = 0.0,
            force_strategy: str | None = None,
            path_filter: str | None = None,
            time_limit: float | None = None,
            result_limit: int | None = None,
        ):
            """Return mock semantic search results."""
            # Generate embeddings for mock chunks
            embeddings = await self.embedding_provider.embed(
                ["def authenticate():", "def validate_session():", "def cache_data():"]
            )

            chunks = [
                {
                    "chunk_id": "chunk1",
                    "code": "def authenticate():",
                    "embedding": embeddings[0],
                },
                {
                    "chunk_id": "chunk2",
                    "code": "def validate_session():",
                    "embedding": embeddings[1],
                },
                {
                    "chunk_id": "chunk3",
                    "code": "def cache_data():",
                    "embedding": embeddings[2],
                },
            ]
            return chunks, None

        async def search_regex_async(
            self,
            pattern: str,
            page_size: int = 10,
            offset: int = 0,
            path_filter: str | None = None,
        ):
            """Return mock regex search results (empty for simplicity)."""
            return [], None

    class MockProvider:
        def get_base_directory(self):
            from pathlib import Path

            return Path("/fake/base")

    class MockDatabaseServices:
        def __init__(self, embedding_provider):
            self.provider = MockProvider()
            self.search_service = MockSearchService(embedding_provider)

    return MockDatabaseServices(controlled_rerank_provider)


@pytest.fixture
def unified_search(db_services, embedding_manager):
    """Create unified search instance."""
    return UnifiedSearch(db_services, embedding_manager)


@pytest.mark.asyncio
async def test_root_high_gap_low_scores(
    unified_search, controlled_rerank_provider
):
    """Test ROOT query with high scores, gap query with low scores.

    Verifies that compound scores are properly averaged when ROOT query
    gives high relevance but gap query gives low relevance.

    Expected behavior:
    - ROOT: chunk1=0.9, chunk2=0.7, chunk3=0.5
    - gap: chunk1=0.1, chunk2=0.3, chunk3=0.5
    - Average: chunk1=0.5, chunk2=0.5, chunk3=0.5
    """
    context = ResearchContext(root_query="How does authentication work?")

    # Configure controlled scores
    controlled_rerank_provider._score_map = {
        "authentication": {0: 0.9, 1: 0.7, 2: 0.5},  # ROOT query scores
        "session": {0: 0.1, 1: 0.3, 2: 0.5},  # gap query scores
    }

    # Run compound reranking
    chunks = await unified_search.unified_search(
        query="How is session management implemented?",
        context=context,
        rerank_queries=["How does authentication work?", "How is session management implemented?"],
    )

    # Verify we got results
    assert len(chunks) == 3

    # Verify compound scores are averages
    # chunk1: (0.9 + 0.1) / 2 = 0.5
    # chunk2: (0.7 + 0.3) / 2 = 0.5
    # chunk3: (0.5 + 0.5) / 2 = 0.5
    expected_scores = [0.5, 0.5, 0.5]

    for i, chunk in enumerate(chunks):
        assert "rerank_score" in chunk
        # Allow small floating point differences
        assert abs(chunk["rerank_score"] - expected_scores[i]) < 0.01, (
            f"chunk {i}: expected {expected_scores[i]}, got {chunk['rerank_score']}"
        )


@pytest.mark.asyncio
async def test_root_low_gap_high_scores(
    unified_search, controlled_rerank_provider
):
    """Test ROOT query with low scores, gap query with high scores.

    Verifies that gap-relevant chunks aren't under-ranked when ROOT query
    gives them low scores but gap query gives high scores.

    Expected behavior:
    - ROOT: chunk1=0.1, chunk2=0.3, chunk3=0.2
    - gap: chunk1=0.9, chunk2=0.7, chunk3=0.8
    - Average: chunk1=0.5, chunk2=0.5, chunk3=0.5
    """
    context = ResearchContext(root_query="How does authentication work?")

    # Configure controlled scores (inverted from previous test)
    controlled_rerank_provider._score_map = {
        "authentication": {0: 0.1, 1: 0.3, 2: 0.2},  # ROOT query scores (low)
        "session": {0: 0.9, 1: 0.7, 2: 0.8},  # gap query scores (high)
    }

    # Run compound reranking
    chunks = await unified_search.unified_search(
        query="How is session management implemented?",
        context=context,
        rerank_queries=["How does authentication work?", "How is session management implemented?"],
    )

    # Verify we got results
    assert len(chunks) == 3

    # Verify compound scores are averages
    # chunk1: (0.1 + 0.9) / 2 = 0.5
    # chunk2: (0.3 + 0.7) / 2 = 0.5
    # chunk3: (0.2 + 0.8) / 2 = 0.5
    expected_scores = [0.5, 0.5, 0.5]

    for i, chunk in enumerate(chunks):
        assert "rerank_score" in chunk
        assert abs(chunk["rerank_score"] - expected_scores[i]) < 0.01, (
            f"chunk {i}: expected {expected_scores[i]}, got {chunk['rerank_score']}"
        )


@pytest.mark.asyncio
async def test_multiple_gap_queries_varying_scores(
    unified_search, controlled_rerank_provider
):
    """Test ROOT + multiple gap queries with varying scores.

    Verifies correct averaging when multiple gap queries are used with
    different relevance scores for each chunk.

    Expected behavior:
    - ROOT: chunk1=0.5, chunk2=0.5, chunk3=0.5
    - gap1: chunk1=0.9, chunk2=0.1, chunk3=0.5
    - gap2: chunk1=0.1, chunk2=0.9, chunk3=0.5
    - Average: chunk1=0.5, chunk2=0.5, chunk3=0.5
    """
    context = ResearchContext(root_query="How does authentication work?")

    # Configure controlled scores for 3 queries
    controlled_rerank_provider._score_map = {
        "authentication": {0: 0.5, 1: 0.5, 2: 0.5},  # ROOT query scores
        "session": {0: 0.9, 1: 0.1, 2: 0.5},  # gap1 query scores
        "cache": {0: 0.1, 1: 0.9, 2: 0.5},  # gap2 query scores
    }

    # Run compound reranking with 3 queries
    chunks = await unified_search.unified_search(
        query="session and cache",
        context=context,
        rerank_queries=[
            "How does authentication work?",
            "How is session management implemented?",
            "How does cache invalidation work?",
        ],
    )

    # Verify we got results
    assert len(chunks) == 3

    # Verify compound scores are averages across all 3 queries
    # chunk1: (0.5 + 0.9 + 0.1) / 3 = 0.5
    # chunk2: (0.5 + 0.1 + 0.9) / 3 = 0.5
    # chunk3: (0.5 + 0.5 + 0.5) / 3 = 0.5
    expected_scores = [0.5, 0.5, 0.5]

    for i, chunk in enumerate(chunks):
        assert "rerank_score" in chunk
        assert abs(chunk["rerank_score"] - expected_scores[i]) < 0.01, (
            f"chunk {i}: expected {expected_scores[i]}, got {chunk['rerank_score']}"
        )


@pytest.mark.asyncio
async def test_chunk_scores_zero_for_all_gaps(
    unified_search, controlled_rerank_provider
):
    """Test chunk not dropped when gap queries give zero scores.

    Verifies that chunks relevant to ROOT query aren't unfairly penalized
    when gap queries give them zero scores.

    Expected behavior:
    - ROOT: chunk1=0.8, chunk2=0.6, chunk3=0.4
    - gap: chunk1=0.0, chunk2=0.0, chunk3=0.0
    - Average: chunk1=0.4, chunk2=0.3, chunk3=0.2
    - All chunks should be retained (not dropped)
    """
    context = ResearchContext(root_query="How does authentication work?")

    # Configure controlled scores
    controlled_rerank_provider._score_map = {
        "authentication": {0: 0.8, 1: 0.6, 2: 0.4},  # ROOT query scores
        "unrelated": {0: 0.0, 1: 0.0, 2: 0.0},  # gap query scores (all zero)
    }

    # Run compound reranking
    chunks = await unified_search.unified_search(
        query="unrelated topic",
        context=context,
        rerank_queries=["How does authentication work?", "unrelated topic"],
    )

    # Verify we got all results (none dropped due to zero gap scores)
    assert len(chunks) == 3

    # Verify compound scores are averages
    # chunk1: (0.8 + 0.0) / 2 = 0.4
    # chunk2: (0.6 + 0.0) / 2 = 0.3
    # chunk3: (0.4 + 0.0) / 2 = 0.2
    expected_scores = [0.4, 0.3, 0.2]

    for i, chunk in enumerate(chunks):
        assert "rerank_score" in chunk
        assert abs(chunk["rerank_score"] - expected_scores[i]) < 0.01, (
            f"chunk {i}: expected {expected_scores[i]}, got {chunk['rerank_score']}"
        )

    # Verify sorting by compound score descending
    scores = [c["rerank_score"] for c in chunks]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_extreme_score_variance_normalized(
    unified_search, controlled_rerank_provider
):
    """Test extreme score variance across queries is normalized.

    Verifies that averaging handles extreme score differences correctly
    without overflow or underflow issues.

    Expected behavior:
    - ROOT: chunk1=1.0, chunk2=1.0, chunk3=1.0 (max scores)
    - gap: chunk1=0.0, chunk2=0.5, chunk3=1.0 (varied scores)
    - Average: chunk1=0.5, chunk2=0.75, chunk3=1.0
    """
    context = ResearchContext(root_query="How does authentication work?")

    # Configure extreme scores
    controlled_rerank_provider._score_map = {
        "authentication": {0: 1.0, 1: 1.0, 2: 1.0},  # ROOT query (max)
        "session": {0: 0.0, 1: 0.5, 2: 1.0},  # gap query (varied)
    }

    # Run compound reranking
    chunks = await unified_search.unified_search(
        query="session",
        context=context,
        rerank_queries=["How does authentication work?", "session"],
    )

    # Verify we got results
    assert len(chunks) == 3

    # Verify compound scores
    # chunk1: (1.0 + 0.0) / 2 = 0.5
    # chunk2: (1.0 + 0.5) / 2 = 0.75
    # chunk3: (1.0 + 1.0) / 2 = 1.0
    expected_scores = [1.0, 0.75, 0.5]  # Sorted descending

    # Scores should be sorted descending
    scores = [c["rerank_score"] for c in chunks]
    assert scores == sorted(scores, reverse=True)

    # Check values match expected
    for i, expected in enumerate(expected_scores):
        assert abs(chunks[i]["rerank_score"] - expected) < 0.01


@pytest.mark.asyncio
async def test_asymmetric_gap_query_scores(
    unified_search, controlled_rerank_provider
):
    """Test asymmetric gap query scores (one chunk high, others low).

    Verifies that compound averaging correctly handles cases where one
    chunk is highly relevant to gap query but others are not.

    Expected behavior:
    - ROOT: chunk1=0.6, chunk2=0.6, chunk3=0.6 (uniform)
    - gap: chunk1=0.95, chunk2=0.05, chunk3=0.05 (asymmetric)
    - Average: chunk1=0.775, chunk2=0.325, chunk3=0.325
    """
    context = ResearchContext(root_query="How does authentication work?")

    # Configure asymmetric scores
    controlled_rerank_provider._score_map = {
        "authentication": {0: 0.6, 1: 0.6, 2: 0.6},  # ROOT query (uniform)
        "session": {0: 0.95, 1: 0.05, 2: 0.05},  # gap query (asymmetric)
    }

    # Run compound reranking
    chunks = await unified_search.unified_search(
        query="session",
        context=context,
        rerank_queries=["How does authentication work?", "session"],
    )

    # Verify we got results
    assert len(chunks) == 3

    # Verify compound scores
    # chunk1: (0.6 + 0.95) / 2 = 0.775
    # chunk2: (0.6 + 0.05) / 2 = 0.325
    # chunk3: (0.6 + 0.05) / 2 = 0.325
    expected_scores = [0.775, 0.325, 0.325]

    # Verify first chunk (highest compound score) is chunk1
    assert abs(chunks[0]["rerank_score"] - 0.775) < 0.01
    assert chunks[0]["chunk_id"] == "chunk1"

    # Verify other chunks have lower compound scores
    for i in range(1, 3):
        assert abs(chunks[i]["rerank_score"] - 0.325) < 0.01


@pytest.mark.asyncio
async def test_negative_scores_handled_correctly(
    unified_search, controlled_rerank_provider
):
    """Test that negative scores (if returned by reranker) are handled.

    Some rerankers may return negative scores for irrelevant documents.
    Verifies that averaging handles negative scores correctly.

    Expected behavior:
    - ROOT: chunk1=0.5, chunk2=0.0, chunk3=-0.2
    - gap: chunk1=-0.5, chunk2=0.0, chunk3=0.2
    - Average: chunk1=0.0, chunk2=0.0, chunk3=0.0
    """
    context = ResearchContext(root_query="How does authentication work?")

    # Configure scores with negatives
    controlled_rerank_provider._score_map = {
        "authentication": {0: 0.5, 1: 0.0, 2: -0.2},  # ROOT query
        "session": {0: -0.5, 1: 0.0, 2: 0.2},  # gap query
    }

    # Run compound reranking
    chunks = await unified_search.unified_search(
        query="session",
        context=context,
        rerank_queries=["How does authentication work?", "session"],
    )

    # Verify we got results
    assert len(chunks) == 3

    # Verify compound scores handle negatives
    # chunk1: (0.5 + -0.5) / 2 = 0.0
    # chunk2: (0.0 + 0.0) / 2 = 0.0
    # chunk3: (-0.2 + 0.2) / 2 = 0.0
    for chunk in chunks:
        assert "rerank_score" in chunk
        assert abs(chunk["rerank_score"] - 0.0) < 0.01


@pytest.mark.asyncio
async def test_single_high_gap_score_doesnt_dominate(
    unified_search, controlled_rerank_provider
):
    """Test that single high gap score doesn't dominate compound average.

    Verifies that averaging prevents a single gap query from dominating
    the compound score when ROOT and other gaps give low scores.

    Expected behavior:
    - ROOT: chunk1=0.1, chunk2=0.1, chunk3=0.1
    - gap1: chunk1=1.0, chunk2=0.1, chunk3=0.1
    - gap2: chunk1=0.1, chunk2=0.1, chunk3=0.1
    - Average: chunk1=0.4, chunk2=0.1, chunk3=0.1
    """
    context = ResearchContext(root_query="How does authentication work?")

    # Configure scores where gap1 gives chunk1 high score
    controlled_rerank_provider._score_map = {
        "authentication": {0: 0.1, 1: 0.1, 2: 0.1},  # ROOT query
        "session": {0: 1.0, 1: 0.1, 2: 0.1},  # gap1 query (chunk1 high)
        "cache": {0: 0.1, 1: 0.1, 2: 0.1},  # gap2 query
    }

    # Run compound reranking with 3 queries
    chunks = await unified_search.unified_search(
        query="session and cache",
        context=context,
        rerank_queries=[
            "How does authentication work?",
            "session",
            "cache",
        ],
    )

    # Verify we got results
    assert len(chunks) == 3

    # Verify compound scores
    # chunk1: (0.1 + 1.0 + 0.1) / 3 = 0.4
    # chunk2: (0.1 + 0.1 + 0.1) / 3 = 0.1
    # chunk3: (0.1 + 0.1 + 0.1) / 3 = 0.1

    # chunk1 should be first (highest compound score)
    assert abs(chunks[0]["rerank_score"] - 0.4) < 0.01
    assert chunks[0]["chunk_id"] == "chunk1"

    # Other chunks should have lower compound scores
    for i in range(1, 3):
        assert abs(chunks[i]["rerank_score"] - 0.1) < 0.01


@pytest.mark.asyncio
async def test_missing_scores_default_to_zero(
    unified_search, controlled_rerank_provider
):
    """Test that missing scores from reranker default to 0.0.

    Some rerankers may not return scores for all chunks (e.g., filtering
    out very low scores). Verifies that missing scores default to 0.0
    in the averaging calculation per implementation line 346.

    Expected behavior:
    - ROOT: chunk1=0.8, chunk2=0.6 (chunk3 missing, defaults to 0.0)
    - gap: chunk1=missing, chunk2=0.4, chunk3=0.2 (chunk1 defaults to 0.0)
    - Average: chunk1=0.4, chunk2=0.5, chunk3=0.1
    """
    context = ResearchContext(root_query="How does authentication work?")

    # Custom rerank function that omits some chunks
    async def partial_rerank(query, documents, top_k=None):
        """Return partial scores (some chunks missing)."""
        results = []

        if "authentication" in query.lower():
            # ROOT query: return only chunk1 and chunk2
            results = [
                RerankResult(index=0, score=0.8),
                RerankResult(index=1, score=0.6),
                # chunk3 missing (index=2)
            ]
        elif "session" in query.lower():
            # gap query: return only chunk2 and chunk3
            results = [
                # chunk1 missing (index=0)
                RerankResult(index=1, score=0.4),
                RerankResult(index=2, score=0.2),
            ]

        return results

    controlled_rerank_provider.rerank = partial_rerank

    # Run compound reranking
    chunks = await unified_search.unified_search(
        query="session",
        context=context,
        rerank_queries=["How does authentication work?", "session"],
    )

    # Verify we got all results (even those with missing scores)
    assert len(chunks) == 3

    # Verify compound scores with default 0.0 for missing
    # chunk1: (0.8 + 0.0) / 2 = 0.4
    # chunk2: (0.6 + 0.4) / 2 = 0.5
    # chunk3: (0.0 + 0.2) / 2 = 0.1

    # Find each chunk by chunk_id and verify score
    chunk_scores = {c["chunk_id"]: c["rerank_score"] for c in chunks}

    assert abs(chunk_scores["chunk1"] - 0.4) < 0.01, (
        f"chunk1: expected 0.4, got {chunk_scores['chunk1']}"
    )
    assert abs(chunk_scores["chunk2"] - 0.5) < 0.01, (
        f"chunk2: expected 0.5, got {chunk_scores['chunk2']}"
    )
    assert abs(chunk_scores["chunk3"] - 0.1) < 0.01, (
        f"chunk3: expected 0.1, got {chunk_scores['chunk3']}"
    )

    # Verify sorting by compound score (chunk2 should be first)
    assert chunks[0]["chunk_id"] == "chunk2"  # highest: 0.5
    assert chunks[1]["chunk_id"] == "chunk1"  # middle: 0.4
    assert chunks[2]["chunk_id"] == "chunk3"  # lowest: 0.1
