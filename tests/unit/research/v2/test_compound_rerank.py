"""Unit tests for compound reranking in unified search.

Tests that gap filling uses compound reranking (ROOT + gap_query) per spec
lines 327, 367-368 in docs/algorithm-coverage-first-research.md.
"""

import pytest

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.services.research.shared.models import ResearchContext
from chunkhound.services.research.shared.unified_search import UnifiedSearch
from tests.fixtures.fake_providers import FakeEmbeddingProvider


@pytest.fixture
def fake_embedding_provider():
    """Create fake embedding provider with reranking support."""
    return FakeEmbeddingProvider(dims=1536)


@pytest.fixture
def embedding_manager(fake_embedding_provider):
    """Create embedding manager with fake provider."""

    class MockEmbeddingManager:
        def get_provider(self):
            return fake_embedding_provider

    return MockEmbeddingManager()


@pytest.fixture
def db_services(fake_embedding_provider):
    """Create mock database services with search service."""

    class MockChunk:
        """Mock chunk object."""

        def __init__(self, chunk_id: str, code: str, embedding: list[float]):
            self.chunk_id = chunk_id
            self.code = code
            self.embedding = embedding

        def get(self, key: str, default=None):
            return getattr(self, key, default)

        def __getitem__(self, key: str):
            return getattr(self, key)

        def __setitem__(self, key: str, value):
            setattr(self, key, value)

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
                ["def foo():", "def bar():", "def baz():"]
            )

            chunks = [
                {
                    "chunk_id": "chunk1",
                    "code": "def foo():",
                    "embedding": embeddings[0],
                },
                {
                    "chunk_id": "chunk2",
                    "code": "def bar():",
                    "embedding": embeddings[1],
                },
                {
                    "chunk_id": "chunk3",
                    "code": "def baz():",
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

    return MockDatabaseServices(fake_embedding_provider)


@pytest.fixture
def unified_search(db_services, embedding_manager):
    """Create unified search instance."""
    return UnifiedSearch(db_services, embedding_manager)


@pytest.mark.asyncio
async def test_single_query_reranking(unified_search, fake_embedding_provider):
    """Test default single-query reranking (ROOT query only)."""
    context = ResearchContext(root_query="How does authentication work?")

    # Run unified search with single query (default behavior)
    chunks = await unified_search.unified_search(
        query="authentication flow",
        context=context,
    )

    # Verify we got results
    assert len(chunks) > 0

    # Verify all chunks have rerank scores
    for chunk in chunks:
        assert "rerank_score" in chunk
        assert isinstance(chunk["rerank_score"], float)

    # Verify chunks are sorted by rerank score descending
    scores = [c["rerank_score"] for c in chunks]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_compound_query_reranking(unified_search, fake_embedding_provider):
    """Test compound reranking with multiple queries (ROOT + gap_query)."""
    context = ResearchContext(root_query="How does authentication work?")

    # Track reranking calls
    rerank_calls = []
    original_rerank = fake_embedding_provider.rerank

    async def tracked_rerank(query, documents, top_k=None):
        rerank_calls.append(query)
        return await original_rerank(query, documents, top_k)

    fake_embedding_provider.rerank = tracked_rerank

    # Run unified search with compound reranking
    gap_query = "How is session management implemented?"
    chunks = await unified_search.unified_search(
        query=gap_query,
        context=context,
        rerank_queries=[context.root_query, gap_query],
    )

    # Verify we got results
    assert len(chunks) > 0

    # Verify reranking was called with both queries
    assert len(rerank_calls) == 2
    assert context.root_query in rerank_calls
    assert gap_query in rerank_calls

    # Verify all chunks have rerank scores (compound average)
    for chunk in chunks:
        assert "rerank_score" in chunk
        assert isinstance(chunk["rerank_score"], float)

    # Verify chunks are sorted by compound score descending
    scores = [c["rerank_score"] for c in chunks]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_compound_score_averaging(unified_search, fake_embedding_provider):
    """Test that compound scores are properly averaged across queries."""
    context = ResearchContext(root_query="How does caching work?")

    # Mock reranking to return predictable scores
    async def mock_rerank(query, documents, top_k=None):
        from chunkhound.interfaces.embedding_provider import RerankResult

        # Return different scores for each query
        if "caching" in query.lower():
            # First query: high scores
            return [
                RerankResult(index=0, score=0.9),
                RerankResult(index=1, score=0.7),
                RerankResult(index=2, score=0.5),
            ]
        else:
            # Second query: lower scores
            return [
                RerankResult(index=0, score=0.5),
                RerankResult(index=1, score=0.3),
                RerankResult(index=2, score=0.1),
            ]

    fake_embedding_provider.rerank = mock_rerank

    # Run unified search with compound reranking
    chunks = await unified_search.unified_search(
        query="cache invalidation",
        context=context,
        rerank_queries=["How does caching work?", "How is cache invalidated?"],
    )

    # Verify compound scores are averages
    # chunk1: (0.9 + 0.5) / 2 = 0.7
    # chunk2: (0.7 + 0.3) / 2 = 0.5
    # chunk3: (0.5 + 0.1) / 2 = 0.3
    expected_scores = [0.7, 0.5, 0.3]

    for i, chunk in enumerate(chunks):
        assert abs(chunk["rerank_score"] - expected_scores[i]) < 0.01


@pytest.mark.asyncio
async def test_single_rerank_query_in_list(unified_search, fake_embedding_provider):
    """Test that single query in list uses single-query path (not compound)."""
    context = ResearchContext(root_query="How does authentication work?")

    # Track reranking calls
    rerank_calls = []
    original_rerank = fake_embedding_provider.rerank

    async def tracked_rerank(query, documents, top_k=None):
        rerank_calls.append(query)
        return await original_rerank(query, documents, top_k)

    fake_embedding_provider.rerank = tracked_rerank

    # Run unified search with single query in list
    chunks = await unified_search.unified_search(
        query="authentication flow",
        context=context,
        rerank_queries=["How does authentication work?"],
    )

    # Verify we got results
    assert len(chunks) > 0

    # Verify reranking was called only once (single-query optimization)
    assert len(rerank_calls) == 1
    assert rerank_calls[0] == "How does authentication work?"


@pytest.mark.asyncio
async def test_empty_rerank_queries_uses_root(unified_search, fake_embedding_provider):
    """Test that empty rerank_queries list falls back to root query."""
    context = ResearchContext(root_query="How does authentication work?")

    # Track reranking calls
    rerank_calls = []
    original_rerank = fake_embedding_provider.rerank

    async def tracked_rerank(query, documents, top_k=None):
        rerank_calls.append(query)
        return await original_rerank(query, documents, top_k)

    fake_embedding_provider.rerank = tracked_rerank

    # Run unified search with empty rerank_queries
    chunks = await unified_search.unified_search(
        query="authentication flow",
        context=context,
        rerank_queries=[],
    )

    # Verify we got results
    assert len(chunks) > 0

    # Verify reranking used root query as fallback
    assert len(rerank_calls) == 1
    assert rerank_calls[0] == context.root_query


@pytest.mark.asyncio
async def test_no_reranking_support_skips_compound_rerank(
    unified_search, fake_embedding_provider, monkeypatch
):
    """Test that compound reranking is skipped when provider doesn't support it."""
    # Disable reranking support
    monkeypatch.setattr(fake_embedding_provider, "supports_reranking", lambda: False)

    context = ResearchContext(root_query="How does authentication work?")

    # Run unified search with compound reranking
    chunks = await unified_search.unified_search(
        query="authentication flow",
        context=context,
        rerank_queries=["How does authentication work?", "authentication flow"],
    )

    # Verify we got results
    assert len(chunks) > 0

    # Verify chunks don't have rerank scores (reranking was skipped)
    for chunk in chunks:
        assert "rerank_score" not in chunk


@pytest.mark.asyncio
async def test_compound_rerank_with_no_chunks(unified_search, db_services):
    """Test compound reranking gracefully handles empty chunk list."""
    # Mock search service to return no results
    class EmptySearchService:
        async def search_semantic(self, **kwargs):
            return [], None

        async def search_regex_async(self, **kwargs):
            return [], None

    db_services.search_service = EmptySearchService()

    context = ResearchContext(root_query="How does authentication work?")

    # Run unified search with compound reranking on empty results
    chunks = await unified_search.unified_search(
        query="authentication flow",
        context=context,
        rerank_queries=["How does authentication work?", "authentication flow"],
    )

    # Verify we got no results (gracefully handled)
    assert len(chunks) == 0
