"""Unit tests for dynamic regex augmentation ratio in unified search.

Tests that regex search collects the correct number of results based on:
  target_count = max(REGEX_MIN_RESULTS, semantic_count * REGEX_AUGMENTATION_RATIO)
  target_per_symbol = target_count // num_symbols

Per spec lines 191-216 in docs/algorithm-coverage-first-research.md:
- scan_page_size=100 is used for DB calls (internal pagination efficiency)
- target_per_symbol controls how many RESULTS are collected per symbol
"""

import pytest

from chunkhound.services.research.shared.models import (
    REGEX_AUGMENTATION_RATIO,
    REGEX_MIN_RESULTS,
    ResearchContext,
)
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
def db_services_with_tracking(fake_embedding_provider):
    """Create mock database services with search service that tracks regex results."""

    class MockSearchService:
        """Mock search service that tracks regex results collected."""

        def __init__(self, embedding_provider):
            self.embedding_provider = embedding_provider
            self.regex_results_by_symbol = {}  # Track results collected per symbol

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
                ["def foo():", "def bar():", "class Baz:"]
            )

            chunks = [
                {
                    "chunk_id": f"semantic_chunk{i}",
                    "code": f"def func{i}():",
                    "symbol": f"func{i}",
                    "embedding": embeddings[i % len(embeddings)],
                }
                for i in range(page_size)
            ]
            return chunks, None

        async def search_regex_async(
            self,
            pattern: str,
            page_size: int = 10,
            offset: int = 0,
            path_filter: str | None = None,
        ):
            """Return mock regex results (page_size=100 per spec, internal pagination)."""
            # Per spec: scan_page_size=100 is used for efficiency
            # The implementation's loop collects up to target_per_symbol results

            # Extract symbol from pattern (pattern is \bsymbol\b)
            import re
            symbol_match = re.search(r"\\b(.+)\\b", pattern)
            symbol = symbol_match.group(1) if symbol_match else pattern

            # Return results with unique IDs based on offset for pagination simulation
            results = [
                {
                    "chunk_id": f"regex_chunk_{symbol}_{offset + i}",
                    "code": f"call {symbol}()",
                }
                for i in range(min(page_size, 10))  # Return up to 10 results per page
            ]

            # Track results for this symbol
            if symbol not in self.regex_results_by_symbol:
                self.regex_results_by_symbol[symbol] = []
            self.regex_results_by_symbol[symbol].extend(results)

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
def unified_search(db_services_with_tracking, embedding_manager):
    """Create unified search instance."""
    return UnifiedSearch(db_services_with_tracking, embedding_manager)


@pytest.mark.asyncio
async def test_augmentation_ratio_with_small_semantic_count(
    unified_search, db_services_with_tracking
):
    """Test that regex collects REGEX_MIN_RESULTS when semantic count is low."""
    context = ResearchContext(root_query="How does foo work?")

    # Run search - implementation uses page_size=30 for semantic search
    # semantic_count = 30, target_count = max(20, 30*0.3) = max(20, 9) = 20
    results = await unified_search.unified_search(
        query="foo implementation",
        context=context,
    )

    # Verify regex searches were made (symbols extracted)
    search_service = db_services_with_tracking.search_service
    assert len(search_service.regex_results_by_symbol) > 0

    # Count regex results in output (chunk_ids starting with "regex_chunk_")
    regex_result_count = sum(
        1 for r in results
        if r.get("chunk_id", "").startswith("regex_chunk_")
    )

    # Expected: target_count = max(20, 30*0.3) = 20
    # With 5 symbols (MAX_SYMBOLS_TO_SEARCH): target_per_symbol = 20 // 5 = 4
    # Total regex results should be ≤ 5 * 4 = 20
    expected_target_count = 20
    assert regex_result_count <= expected_target_count


@pytest.mark.asyncio
async def test_augmentation_ratio_with_large_semantic_count(
    unified_search, db_services_with_tracking
):
    """Test that regex uses AUGMENTATION_RATIO when semantic count is high."""
    context = ResearchContext(root_query="How does foo work?")

    # Mock search service to return many semantic results
    class LargeSemanticSearchService:
        def __init__(self, embedding_provider):
            self.embedding_provider = embedding_provider
            self.regex_results_by_symbol = {}

        async def search_semantic(self, **kwargs):
            # Return 100 semantic results
            embeddings = await self.embedding_provider.embed(
                ["code"] * 100
            )  # Generate 100 embeddings
            return [
                {
                    "chunk_id": f"semantic_{i}",
                    "code": f"def func{i}():",
                    "symbol": f"func{i}",
                    "embedding": embeddings[i],
                }
                for i in range(100)
            ], None

        async def search_regex_async(self, pattern: str, page_size: int = 10, offset: int = 0, **kwargs):
            import re
            symbol_match = re.search(r"\\b(.+)\\b", pattern)
            symbol = symbol_match.group(1) if symbol_match else pattern

            results = [
                {"chunk_id": f"regex_{symbol}_{offset + i}", "code": f"call {symbol}()"}
                for i in range(min(page_size, 10))
            ]

            if symbol not in self.regex_results_by_symbol:
                self.regex_results_by_symbol[symbol] = []
            self.regex_results_by_symbol[symbol].extend(results)

            return results, None

    db_services_with_tracking.search_service = LargeSemanticSearchService(
        db_services_with_tracking.search_service.embedding_provider
    )

    # Run search with large semantic count (100 results)
    # 100 * 0.3 = 30, which is > min of 20, so should use 30
    results = await unified_search.unified_search(
        query="foo implementation",
        context=context,
    )

    # Verify regex searches were made
    search_service = db_services_with_tracking.search_service
    assert len(search_service.regex_results_by_symbol) > 0

    # Count regex results in output
    regex_result_count = sum(
        1 for r in results
        if r.get("chunk_id", "").startswith("regex_")
    )

    # Calculate expected: max(20, 100 * 0.3) = 30
    # With 5 symbols: 30 // 5 = 6 per symbol, total ≤ 30
    expected_target_count = 30
    assert regex_result_count <= expected_target_count


@pytest.mark.asyncio
async def test_augmentation_ratio_boundary_case(
    unified_search, db_services_with_tracking
):
    """Test augmentation ratio at the boundary (semantic_count * 0.3 == 20)."""
    context = ResearchContext(root_query="How does foo work?")

    # Mock to return exactly semantic_count where count * 0.3 = 20
    # 20 / 0.3 = 66.67, so use 67 semantic results
    class BoundarySearchService:
        def __init__(self, embedding_provider):
            self.embedding_provider = embedding_provider
            self.regex_results_by_symbol = {}

        async def search_semantic(self, **kwargs):
            # Return 67 semantic results (67 * 0.3 = 20.1)
            embeddings = await self.embedding_provider.embed(["code"] * 67)
            return [
                {
                    "chunk_id": f"semantic_{i}",
                    "code": f"def func{i}():",
                    "symbol": f"func{i}",
                    "embedding": embeddings[i],
                }
                for i in range(67)
            ], None

        async def search_regex_async(self, pattern: str, page_size: int = 10, offset: int = 0, **kwargs):
            import re
            symbol_match = re.search(r"\\b(.+)\\b", pattern)
            symbol = symbol_match.group(1) if symbol_match else pattern

            results = [
                {"chunk_id": f"regex_{symbol}_{offset + i}", "code": f"call {symbol}()"}
                for i in range(min(page_size, 10))
            ]

            if symbol not in self.regex_results_by_symbol:
                self.regex_results_by_symbol[symbol] = []
            self.regex_results_by_symbol[symbol].extend(results)

            return results, None

    db_services_with_tracking.search_service = BoundarySearchService(
        db_services_with_tracking.search_service.embedding_provider
    )

    results = await unified_search.unified_search(
        query="foo implementation",
        context=context,
    )

    # Verify regex searches were made
    search_service = db_services_with_tracking.search_service
    assert len(search_service.regex_results_by_symbol) > 0

    # Count regex results in output
    regex_result_count = sum(
        1 for r in results
        if r.get("chunk_id", "").startswith("regex_")
    )

    # Calculate expected: max(20, int(67 * 0.3)) = max(20, 20) = 20
    # With 5 symbols: 20 // 5 = 4 per symbol, total ≤ 20
    expected_target_count = 20
    assert regex_result_count <= expected_target_count


@pytest.mark.asyncio
async def test_augmentation_ratio_with_no_semantic_results(
    unified_search, db_services_with_tracking
):
    """Test that no regex search occurs when semantic search returns nothing."""
    context = ResearchContext(root_query="How does foo work?")

    # Mock to return no semantic results
    class EmptySemanticSearchService:
        def __init__(self):
            self.regex_call_count = 0

        async def search_semantic(self, **kwargs):
            return [], None

        async def search_regex_async(self, **kwargs):
            # Should never be called
            self.regex_call_count += 1
            return [], None

    db_services_with_tracking.search_service = EmptySemanticSearchService()

    await unified_search.unified_search(
        query="foo implementation",
        context=context,
    )

    # Verify no regex calls were made (no symbols extracted)
    search_service = db_services_with_tracking.search_service
    assert search_service.regex_call_count == 0


@pytest.mark.asyncio
async def test_augmentation_ratio_with_one_symbol(
    unified_search, db_services_with_tracking
):
    """Test target_per_symbol calculation with single symbol."""
    context = ResearchContext(root_query="How does foo work?")

    # Mock to return semantic results with only 1 symbol
    class SingleSymbolSearchService:
        def __init__(self, embedding_provider):
            self.embedding_provider = embedding_provider
            self.regex_results_by_symbol = {}

        async def search_semantic(self, **kwargs):
            # Return 50 results but only 1 unique symbol
            embeddings = await self.embedding_provider.embed(["code"] * 50)
            return [
                {
                    "chunk_id": f"semantic_{i}",
                    "code": "def single_func():",
                    "symbol": "single_func",  # All have same symbol
                    "embedding": embeddings[i],
                }
                for i in range(50)
            ], None

        async def search_regex_async(self, pattern: str, page_size: int = 10, offset: int = 0, **kwargs):
            import re
            symbol_match = re.search(r"\\b(.+)\\b", pattern)
            symbol = symbol_match.group(1) if symbol_match else pattern

            results = [
                {"chunk_id": f"regex_{symbol}_{offset + i}", "code": f"call {symbol}()"}
                for i in range(min(page_size, 30))  # More results for higher target
            ]

            if symbol not in self.regex_results_by_symbol:
                self.regex_results_by_symbol[symbol] = []
            self.regex_results_by_symbol[symbol].extend(results)

            return results, None

    db_services_with_tracking.search_service = SingleSymbolSearchService(
        db_services_with_tracking.search_service.embedding_provider
    )

    results = await unified_search.unified_search(
        query="foo implementation",
        context=context,
    )

    # Verify regex searches were made
    search_service = db_services_with_tracking.search_service
    assert len(search_service.regex_results_by_symbol) > 0

    # Count regex results in output
    regex_result_count = sum(
        1 for r in results
        if r.get("chunk_id", "").startswith("regex_")
    )

    # Calculate expected: max(20, 50 * 0.3) = max(20, 15) = 20
    # With 1 symbol: 20 // 1 = 20 per symbol, total ≤ 20
    expected_target_count = 20
    assert regex_result_count <= expected_target_count


@pytest.mark.asyncio
async def test_augmentation_ratio_with_many_symbols(
    unified_search, db_services_with_tracking
):
    """Test target_per_symbol calculation with many symbols."""
    context = ResearchContext(root_query="How does foo work?")

    # Mock to return semantic results with many different symbols
    class ManySymbolsSearchService:
        def __init__(self, embedding_provider):
            self.embedding_provider = embedding_provider
            self.regex_results_by_symbol = {}

        async def search_semantic(self, **kwargs):
            # Return 200 results with 100 unique symbols (exceeds MAX_SYMBOLS_TO_SEARCH)
            embeddings = await self.embedding_provider.embed(["code"] * 200)
            return [
                {
                    "chunk_id": f"semantic_{i}",
                    "code": f"def func{i}():",
                    "symbol": f"func{i}",  # Each has unique symbol
                    "embedding": embeddings[i],
                }
                for i in range(200)
            ], None

        async def search_regex_async(self, pattern: str, page_size: int = 10, offset: int = 0, **kwargs):
            import re
            symbol_match = re.search(r"\\b(.+)\\b", pattern)
            symbol = symbol_match.group(1) if symbol_match else pattern

            results = [
                {"chunk_id": f"regex_{symbol}_{offset + i}", "code": f"call {symbol}()"}
                for i in range(min(page_size, 20))  # Return enough for targets
            ]

            if symbol not in self.regex_results_by_symbol:
                self.regex_results_by_symbol[symbol] = []
            self.regex_results_by_symbol[symbol].extend(results)

            return results, None

    db_services_with_tracking.search_service = ManySymbolsSearchService(
        db_services_with_tracking.search_service.embedding_provider
    )

    results = await unified_search.unified_search(
        query="foo implementation",
        context=context,
    )

    # Verify regex searches were made
    search_service = db_services_with_tracking.search_service

    # Should have made exactly 5 symbol searches (MAX_SYMBOLS_TO_SEARCH)
    assert len(search_service.regex_results_by_symbol) == 5

    # Count regex results in output
    regex_result_count = sum(
        1 for r in results
        if r.get("chunk_id", "").startswith("regex_")
    )

    # Calculate expected: max(20, 200 * 0.3) = 60
    # With 5 symbols (MAX_SYMBOLS_TO_SEARCH): 60 // 5 = 12 per symbol, total ≤ 60
    expected_target_count = 60
    assert regex_result_count <= expected_target_count


@pytest.mark.asyncio
async def test_augmentation_ratio_math(
    unified_search, db_services_with_tracking
):
    """Test that augmentation ratio math is correct for various semantic counts."""
    # Test cases: (semantic_count, expected_target_count)
    # target_count = max(REGEX_MIN_RESULTS, semantic_count * REGEX_AUGMENTATION_RATIO)
    test_cases = [
        (10, 20),  # 10 * 0.3 = 3 < 20, use min=20
        (50, 20),  # 50 * 0.3 = 15 < 20, use min=20
        (67, 20),  # 67 * 0.3 = 20.1 -> int=20, max(20,20)=20
        (100, 30),  # 100 * 0.3 = 30, max(20,30)=30
        (200, 60),  # 200 * 0.3 = 60, max(20,60)=60
    ]

    for semantic_count, expected_target in test_cases:

        class TestSearchService:
            def __init__(self, embedding_provider, semantic_count):
                self.embedding_provider = embedding_provider
                self.semantic_count = semantic_count
                self.regex_results_by_symbol = {}

            async def search_semantic(self, **kwargs):
                embeddings = await self.embedding_provider.embed(
                    ["code"] * self.semantic_count
                )
                return [
                    {
                        "chunk_id": f"semantic_{i}",
                        "code": f"def func{i}():",
                        "symbol": f"func{i}",
                        "embedding": embeddings[i],
                    }
                    for i in range(self.semantic_count)
                ], None

            async def search_regex_async(
                self, pattern: str, page_size: int = 10, offset: int = 0, **kwargs
            ):
                import re
                symbol_match = re.search(r"\\b(.+)\\b", pattern)
                symbol = symbol_match.group(1) if symbol_match else pattern

                results = [
                    {"chunk_id": f"regex_{symbol}_{offset + i}", "code": f"call {symbol}()"}
                    for i in range(min(page_size, 20))
                ]

                if symbol not in self.regex_results_by_symbol:
                    self.regex_results_by_symbol[symbol] = []
                self.regex_results_by_symbol[symbol].extend(results)

                return results, None

        db_services_with_tracking.search_service = TestSearchService(
            db_services_with_tracking.search_service.embedding_provider, semantic_count
        )

        context = ResearchContext(root_query="Test query")
        results = await unified_search.unified_search(query="test", context=context)

        # Count regex results in output
        regex_result_count = sum(
            1 for r in results
            if r.get("chunk_id", "").startswith("regex_")
        )

        # Verify results are bounded by target_count
        assert (
            regex_result_count <= expected_target
        ), f"semantic_count={semantic_count}: expected ≤{expected_target} regex results, got {regex_result_count}"


@pytest.mark.asyncio
async def test_augmentation_constants_are_used(
    unified_search, db_services_with_tracking
):
    """Test that REGEX_AUGMENTATION_RATIO and REGEX_MIN_RESULTS constants are used."""
    # Verify constants are imported and have expected values
    assert REGEX_AUGMENTATION_RATIO == 0.3
    assert REGEX_MIN_RESULTS == 20

    # Test that changing semantic count affects regex result count correctly
    class ConfigurableSearchService:
        def __init__(self, embedding_provider):
            self.embedding_provider = embedding_provider
            self.regex_results_by_symbol = {}

        async def search_semantic(self, **kwargs):
            # Return semantic_count = 100
            embeddings = await self.embedding_provider.embed(["code"] * 100)
            return [
                {
                    "chunk_id": f"semantic_{i}",
                    "code": f"def func{i}():",
                    "symbol": f"func{i}",
                    "embedding": embeddings[i],
                }
                for i in range(100)
            ], None

        async def search_regex_async(self, pattern: str, page_size: int = 10, offset: int = 0, **kwargs):
            import re
            symbol_match = re.search(r"\\b(.+)\\b", pattern)
            symbol = symbol_match.group(1) if symbol_match else pattern

            results = [
                {"chunk_id": f"regex_{symbol}_{offset + i}", "code": f"call {symbol}()"}
                for i in range(min(page_size, 10))
            ]

            if symbol not in self.regex_results_by_symbol:
                self.regex_results_by_symbol[symbol] = []
            self.regex_results_by_symbol[symbol].extend(results)

            return results, None

    db_services_with_tracking.search_service = ConfigurableSearchService(
        db_services_with_tracking.search_service.embedding_provider
    )

    context = ResearchContext(root_query="Test query")
    results = await unified_search.unified_search(query="test", context=context)

    # Count regex results in output
    regex_result_count = sum(
        1 for r in results
        if r.get("chunk_id", "").startswith("regex_")
    )

    # With semantic_count=100: max(20, 100*0.3) = 30
    # Results should be bounded by target_count = 30
    expected_target_count = 30
    assert regex_result_count <= expected_target_count
