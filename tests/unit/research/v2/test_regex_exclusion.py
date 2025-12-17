"""Unit tests for regex exclusion in unified search.

Tests that regex search excludes already-found semantic chunk IDs per spec
lines 186-217 in docs/algorithm-coverage-first-research.md.
"""

import pytest

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
            """Return mock semantic search results with symbols."""
            # Generate embeddings for mock chunks
            embeddings = await self.embedding_provider.embed(
                ["def authenticate():", "def validate_token():", "class AuthManager:"]
            )

            chunks = [
                {
                    "chunk_id": "semantic_chunk1",
                    "code": "def authenticate():",
                    "symbol": "authenticate",
                    "embedding": embeddings[0],
                },
                {
                    "chunk_id": "semantic_chunk2",
                    "code": "def validate_token():",
                    "symbol": "validate_token",
                    "embedding": embeddings[1],
                },
                {
                    "chunk_id": "semantic_chunk3",
                    "code": "class AuthManager:",
                    "symbol": "AuthManager",
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
            """Return mock regex results (simulate overlapping chunk IDs)."""
            # Simulate regex finding both semantic chunks and new chunks
            results = []

            # Simulate different results based on the symbol pattern
            if "authenticate" in pattern:
                results = [
                    {
                        "chunk_id": "semantic_chunk1",  # Overlaps with semantic
                        "code": "def authenticate():",
                    },
                    {
                        "chunk_id": "regex_only_chunk1",  # New chunk
                        "code": "if authenticate():",
                    },
                ]
            elif "validate_token" in pattern:
                results = [
                    {
                        "chunk_id": "semantic_chunk2",  # Overlaps with semantic
                        "code": "def validate_token():",
                    },
                    {
                        "chunk_id": "regex_only_chunk2",  # New chunk
                        "code": "token = validate_token()",
                    },
                ]
            elif "AuthManager" in pattern:
                results = [
                    {
                        "chunk_id": "semantic_chunk3",  # Overlaps with semantic
                        "code": "class AuthManager:",
                    },
                    {
                        "chunk_id": "regex_only_chunk3",  # New chunk
                        "code": "manager = AuthManager()",
                    },
                ]

            return results, None

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
async def test_regex_excludes_semantic_chunks(unified_search, fake_embedding_provider):
    """Test that regex search excludes chunk IDs already found via semantic search."""
    context = ResearchContext(root_query="How does authentication work?")

    # Run unified search
    chunks = await unified_search.unified_search(
        query="authentication implementation",
        context=context,
    )

    # Verify we got results
    assert len(chunks) > 0

    # Extract chunk IDs
    chunk_ids = [c.get("chunk_id") or c.get("id") for c in chunks]

    # Verify semantic chunks are present (from semantic search)
    assert "semantic_chunk1" in chunk_ids
    assert "semantic_chunk2" in chunk_ids
    assert "semantic_chunk3" in chunk_ids

    # Verify regex-only chunks are present (not filtered out)
    assert "regex_only_chunk1" in chunk_ids
    assert "regex_only_chunk2" in chunk_ids
    assert "regex_only_chunk3" in chunk_ids

    # Verify no duplicates (each chunk appears exactly once)
    assert len(chunk_ids) == len(set(chunk_ids))

    # Count expected: 3 semantic + 3 regex-only = 6 unique chunks
    assert len(chunks) == 6


@pytest.mark.asyncio
async def test_regex_with_empty_semantic_results(unified_search, db_services):
    """Test regex search when semantic search returns no results."""
    # Mock search service to return no semantic results
    class EmptySemanticSearchService:
        async def search_semantic(self, **kwargs):
            return [], None

        async def search_regex_async(self, **kwargs):
            # Should still return regex results
            return [
                {"chunk_id": "regex_chunk1", "code": "def foo():"},
                {"chunk_id": "regex_chunk2", "code": "def bar():"},
            ], None

    db_services.search_service = EmptySemanticSearchService()

    context = ResearchContext(root_query="How does authentication work?")

    # Run unified search
    chunks = await unified_search.unified_search(
        query="authentication implementation",
        context=context,
    )

    # Should return empty because no symbols extracted (no semantic results)
    # Regex search depends on symbol extraction from semantic results
    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_search_by_symbols_with_exclusion(unified_search):
    """Test search_by_symbols method with explicit exclude_ids parameter."""
    symbols = ["authenticate", "validate_token"]

    # Call search_by_symbols with exclusion set
    exclude_ids = {"semantic_chunk1", "semantic_chunk2"}
    results = await unified_search.search_by_symbols(
        symbols,
        path_filter=None,
        exclude_ids=exclude_ids,
    )

    # Extract chunk IDs
    chunk_ids = [c.get("chunk_id") or c.get("id") for c in results]

    # Verify excluded chunks are not present
    assert "semantic_chunk1" not in chunk_ids
    assert "semantic_chunk2" not in chunk_ids

    # Verify non-excluded chunks are present
    assert "regex_only_chunk1" in chunk_ids
    assert "regex_only_chunk2" in chunk_ids

    # Verify no duplicates
    assert len(chunk_ids) == len(set(chunk_ids))


@pytest.mark.asyncio
async def test_search_by_symbols_without_exclusion(unified_search):
    """Test search_by_symbols method without exclusion (default behavior)."""
    symbols = ["authenticate", "validate_token"]

    # Call search_by_symbols without exclusion set
    results = await unified_search.search_by_symbols(
        symbols,
        path_filter=None,
        exclude_ids=None,
    )

    # Extract chunk IDs
    chunk_ids = [c.get("chunk_id") or c.get("id") for c in results]

    # Verify all chunks are present (no exclusion)
    assert "semantic_chunk1" in chunk_ids
    assert "semantic_chunk2" in chunk_ids
    assert "regex_only_chunk1" in chunk_ids
    assert "regex_only_chunk2" in chunk_ids

    # Expected: 2 semantic + 2 regex-only = 4 chunks
    assert len(results) == 4


@pytest.mark.asyncio
async def test_search_by_symbols_with_empty_exclusion_set(unified_search):
    """Test search_by_symbols with empty exclusion set (same as no exclusion)."""
    symbols = ["authenticate"]

    # Call search_by_symbols with empty exclusion set
    results = await unified_search.search_by_symbols(
        symbols,
        path_filter=None,
        exclude_ids=set(),
    )

    # Extract chunk IDs
    chunk_ids = [c.get("chunk_id") or c.get("id") for c in results]

    # Verify all chunks are present
    assert "semantic_chunk1" in chunk_ids
    assert "regex_only_chunk1" in chunk_ids

    # Expected: 2 chunks (1 semantic overlap + 1 regex-only)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_exclusion_with_missing_chunk_ids(unified_search, db_services):
    """Test that chunks without chunk_id field are gracefully handled."""

    class MalformedChunkSearchService:
        async def search_semantic(self, **kwargs):
            # Return chunk without chunk_id
            return [
                {
                    "code": "def foo():",
                    "symbol": "foo",
                }
            ], None

        async def search_regex_async(self, **kwargs):
            # Return mix of chunks with/without IDs
            return [
                {"chunk_id": "regex_chunk1", "code": "def bar():"},
                {"code": "def baz():"},  # Missing chunk_id
            ], None

    db_services.search_service = MalformedChunkSearchService()

    context = ResearchContext(root_query="Test query")

    # Should not crash with missing chunk IDs
    chunks = await unified_search.unified_search(
        query="test",
        context=context,
    )

    # Verify execution completed without errors
    assert isinstance(chunks, list)


@pytest.mark.asyncio
async def test_exclusion_performance_with_large_sets(unified_search, db_services):
    """Test that exclusion works efficiently with large exclude sets."""

    class LargeResultSearchService:
        async def search_semantic(self, **kwargs):
            # Generate 100 semantic chunks
            return [
                {
                    "chunk_id": f"semantic_{i}",
                    "code": f"def func{i}():",
                    "symbol": f"func{i}",
                }
                for i in range(100)
            ], None

        async def search_regex_async(self, **kwargs):
            # Generate overlapping results (50 duplicates + 50 new)
            # Properly implement pagination by respecting offset parameter
            offset = kwargs.get("offset", 0)
            page_size = kwargs.get("page_size", 100)

            # Full result set: 50 semantic overlaps + 50 new regex chunks
            all_results = []
            # First 50 overlap with semantic
            for i in range(50):
                all_results.append({"chunk_id": f"semantic_{i}", "code": f"def func{i}():"})
            # Last 50 are new
            for i in range(50):
                all_results.append({"chunk_id": f"regex_{i}", "code": f"call func{i}()"})

            # Return paginated slice
            return all_results[offset:offset + page_size], None

    db_services.search_service = LargeResultSearchService()

    context = ResearchContext(root_query="Test query")

    # Run unified search with large result sets
    chunks = await unified_search.unified_search(
        query="test",
        context=context,
    )

    # Extract chunk IDs
    chunk_ids = [c.get("chunk_id") or c.get("id") for c in chunks]

    # Expected behavior with pagination fix:
    # - 100 semantic chunks
    # - target_count = max(20, 100 * 0.3) = 30 regex chunks
    # - MAX_SYMBOLS_TO_SEARCH = 5 symbols
    # - target_per_symbol = max(1, 30 // 5) = 6 chunks per symbol
    # - But mock returns same chunks for all symbols, so after dedup: min(50, 6) = 6 unique regex chunks
    # - Total: 100 semantic + 6 regex = 106 chunks
    assert len(chunks) == 106
    assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates

    # Verify all semantic chunks present
    for i in range(100):
        assert f"semantic_{i}" in chunk_ids

    # Verify we got 6 regex chunks (target_per_symbol)
    regex_chunks = [cid for cid in chunk_ids if cid.startswith("regex_")]
    assert len(regex_chunks) == 6


@pytest.mark.asyncio
async def test_exclusion_with_none_exclude_ids(unified_search):
    """Test that None exclude_ids is handled as empty set."""
    symbols = ["authenticate"]

    # Call with explicit None (should work same as empty set)
    results = await unified_search.search_by_symbols(
        symbols,
        path_filter=None,
        exclude_ids=None,
    )

    # Should return results without errors
    assert len(results) > 0
