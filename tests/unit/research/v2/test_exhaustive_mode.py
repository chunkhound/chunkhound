"""Tests for exhaustive mode configuration in coverage-first research.

Exhaustive mode enables unlimited result collection with extended time limits:
- time_limit: 5s → 600s
- result_limit: 500 → None (unlimited)

These tests verify that the configuration properly flows through:
1. ResearchConfig.get_effective_time_limit() / get_effective_result_limit()
2. UnifiedSearch (accepts config in __init__)
3. SearchService.search_semantic() (accepts time_limit/result_limit params)
4. MultiHopStrategy.search() (respects the limits)
"""

import pytest

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.services.research.shared.unified_search import UnifiedSearch


class TestResearchConfigExhaustiveMode:
    """Test ResearchConfig exhaustive mode settings."""

    def test_default_mode_uses_standard_limits(self):
        """Test that default config uses standard limits."""
        config = ResearchConfig()

        assert config.exhaustive_mode is False
        assert config.get_effective_time_limit() == 5.0  # multi_hop_time_limit
        assert config.get_effective_result_limit() == 500  # multi_hop_result_limit

    def test_exhaustive_mode_uses_extended_limits(self):
        """Test that exhaustive mode uses extended limits."""
        config = ResearchConfig(exhaustive_mode=True)

        assert config.exhaustive_mode is True
        assert config.get_effective_time_limit() == 600.0  # exhaustive_time_limit
        assert config.get_effective_result_limit() is None  # No limit

    def test_exhaustive_mode_custom_timeout(self):
        """Test that exhaustive mode respects custom timeout."""
        config = ResearchConfig(
            exhaustive_mode=True,
            exhaustive_time_limit=900.0,
        )

        assert config.get_effective_time_limit() == 900.0

    def test_standard_mode_custom_limits(self):
        """Test that standard mode respects custom limits."""
        config = ResearchConfig(
            exhaustive_mode=False,
            multi_hop_time_limit=10.0,
            multi_hop_result_limit=1000,
        )

        assert config.get_effective_time_limit() == 10.0
        assert config.get_effective_result_limit() == 1000


class TestUnifiedSearchConfigIntegration:
    """Test that UnifiedSearch properly integrates with config."""

    def test_unified_search_accepts_none_config(self):
        """Test that UnifiedSearch works without config (backward compatibility)."""
        from unittest.mock import MagicMock

        db_services = MagicMock()
        embedding_manager = MagicMock()

        # Should not raise
        unified_search = UnifiedSearch(db_services, embedding_manager, config=None)

        assert unified_search._config is None

    def test_unified_search_accepts_config(self):
        """Test that UnifiedSearch accepts and stores config."""
        from unittest.mock import MagicMock

        db_services = MagicMock()
        embedding_manager = MagicMock()
        config = ResearchConfig(exhaustive_mode=True)

        unified_search = UnifiedSearch(db_services, embedding_manager, config=config)

        assert unified_search._config is config
        assert unified_search._config.exhaustive_mode is True


@pytest.mark.asyncio
class TestMultiHopStrategyExhaustiveMode:
    """Test that multi-hop strategy respects exhaustive mode limits."""

    async def test_multi_hop_default_limits(self):
        """Test multi-hop with default limits (5s, 500 results)."""
        from unittest.mock import AsyncMock, MagicMock

        from chunkhound.services.search.multi_hop_strategy import MultiHopStrategy

        # Mock dependencies
        db_provider = MagicMock()
        embedding_provider = MagicMock()
        embedding_provider.supports_reranking.return_value = True
        embedding_provider.rerank = AsyncMock(return_value=[])

        # Mock single-hop search to return minimal results (fall back to single-hop)
        single_hop_search = AsyncMock(return_value=([], {}))

        strategy = MultiHopStrategy(
            db_provider, embedding_provider, single_hop_search
        )

        # Call with default parameters (no time_limit/result_limit specified)
        results, _ = await strategy.search(
            query="test query",
            page_size=10,
            offset=0,
            threshold=None,
            provider="test",
            model="test-model",
            path_filter=None,
        )

        # Should fall back to single-hop due to insufficient results
        single_hop_search.assert_called()

    async def test_multi_hop_exhaustive_time_limit(self):
        """Test multi-hop with exhaustive time limit (600s)."""
        from unittest.mock import AsyncMock, MagicMock

        from chunkhound.services.search.multi_hop_strategy import MultiHopStrategy

        # Mock dependencies
        db_provider = MagicMock()
        embedding_provider = MagicMock()
        embedding_provider.supports_reranking.return_value = True
        embedding_provider.rerank = AsyncMock(return_value=[])

        # Mock single-hop search to return enough results for expansion
        mock_results = [
            {
                "chunk_id": f"chunk_{i}",
                "content": f"content {i}",
                "similarity": 0.9 - (i * 0.01),
            }
            for i in range(20)
        ]
        single_hop_search = AsyncMock(return_value=(mock_results, {}))

        # Mock find_similar_chunks to return no neighbors (terminate expansion)
        db_provider.find_similar_chunks.return_value = []

        strategy = MultiHopStrategy(
            db_provider, embedding_provider, single_hop_search
        )

        # Call with exhaustive time limit
        results, _ = await strategy.search(
            query="test query",
            page_size=10,
            offset=0,
            threshold=None,
            provider="test",
            model="test-model",
            path_filter=None,
            time_limit=600.0,  # Exhaustive mode
            result_limit=None,  # No limit
        )

        # Should complete without hitting time limit
        assert len(results) <= 10  # Pagination applied

    async def test_multi_hop_unlimited_results(self):
        """Test multi-hop with unlimited results (result_limit=None)."""
        from unittest.mock import AsyncMock, MagicMock

        from chunkhound.services.search.multi_hop_strategy import MultiHopStrategy

        # Mock dependencies
        db_provider = MagicMock()
        embedding_provider = MagicMock()
        embedding_provider.supports_reranking.return_value = True

        # Mock rerank to return scored results
        def mock_rerank(query, documents, top_k=None):
            from chunkhound.interfaces.embedding_provider import RerankResult

            return [
                RerankResult(index=i, score=0.9 - (i * 0.01))
                for i in range(len(documents))
            ]

        embedding_provider.rerank = AsyncMock(side_effect=mock_rerank)

        # Mock single-hop search to return enough results
        mock_results = [
            {
                "chunk_id": f"chunk_{i}",
                "content": f"content {i}",
                "similarity": 0.9 - (i * 0.01),
            }
            for i in range(20)
        ]
        single_hop_search = AsyncMock(return_value=(mock_results, {}))

        # Mock find_similar_chunks to return no neighbors (terminate expansion)
        db_provider.find_similar_chunks.return_value = []

        strategy = MultiHopStrategy(
            db_provider, embedding_provider, single_hop_search
        )

        # Call with unlimited result limit
        results, pagination = await strategy.search(
            query="test query",
            page_size=50,  # Request more than default 500 limit
            offset=0,
            threshold=None,
            provider="test",
            model="test-model",
            path_filter=None,
            time_limit=10.0,
            result_limit=None,  # Unlimited
        )

        # Should not be capped by 500 limit
        # (In this test, we get 20 results, but the point is no limit is enforced)
        assert pagination["total"] == len(mock_results)


@pytest.mark.asyncio
class TestSearchServiceExhaustiveMode:
    """Test that search_service passes through exhaustive mode parameters."""

    async def test_search_semantic_passes_time_and_result_limits(self):
        """Test that search_semantic passes time_limit and result_limit to strategy."""
        from unittest.mock import AsyncMock, MagicMock

        from chunkhound.services.search_service import SearchService

        # Mock dependencies
        db_provider = MagicMock()
        embedding_provider = MagicMock()
        embedding_provider.name = "test"
        embedding_provider.model = "test-model"
        embedding_provider.supports_reranking.return_value = True

        # Mock multi-hop strategy
        multi_hop_strategy = MagicMock()
        multi_hop_strategy.search = AsyncMock(return_value=([], {}))

        search_service = SearchService(db_provider, embedding_provider)
        search_service._multi_hop_strategy = multi_hop_strategy

        # Call with exhaustive parameters
        await search_service.search_semantic(
            query="test query",
            page_size=10,
            time_limit=600.0,
            result_limit=None,
            force_strategy="multi_hop",
        )

        # Verify parameters were passed through
        multi_hop_strategy.search.assert_called_once()
        call_kwargs = multi_hop_strategy.search.call_args.kwargs
        assert call_kwargs["time_limit"] == 600.0
        assert call_kwargs["result_limit"] is None


@pytest.mark.asyncio
class TestUnifiedSearchExhaustiveMode:
    """Test that UnifiedSearch uses config to set exhaustive parameters."""

    async def test_unified_search_uses_config_for_limits(self):
        """Test that unified_search passes config limits to search_service."""
        from unittest.mock import AsyncMock, MagicMock

        from chunkhound.services.research.shared.models import ResearchContext

        # Mock dependencies
        db_services = MagicMock()
        embedding_manager = MagicMock()

        # Mock search_service to capture parameters
        search_service = MagicMock()
        search_service.search_semantic = AsyncMock(return_value=([], {}))
        db_services.search_service = search_service

        # Create config with exhaustive mode
        config = ResearchConfig(exhaustive_mode=True)

        # Create unified search with config
        unified_search = UnifiedSearch(db_services, embedding_manager, config)

        # Mock embedding provider for reranking
        embedding_provider = MagicMock()
        embedding_provider.supports_reranking.return_value = False
        embedding_manager.get_provider.return_value = embedding_provider

        # Call unified_search
        context = ResearchContext(root_query="test query")
        await unified_search.unified_search(
            query="test query",
            context=context,
            expanded_queries=["test query"],  # Disable query expansion
        )

        # Verify search_service was called with exhaustive limits
        search_service.search_semantic.assert_called()
        call_kwargs = search_service.search_semantic.call_args.kwargs
        assert call_kwargs["time_limit"] == 600.0
        assert call_kwargs["result_limit"] is None

    async def test_unified_search_without_config_uses_defaults(self):
        """Test that unified_search without config uses default limits."""
        from unittest.mock import AsyncMock, MagicMock

        from chunkhound.services.research.shared.models import ResearchContext

        # Mock dependencies
        db_services = MagicMock()
        embedding_manager = MagicMock()

        # Mock search_service
        search_service = MagicMock()
        search_service.search_semantic = AsyncMock(return_value=([], {}))
        db_services.search_service = search_service

        # Create unified search WITHOUT config
        unified_search = UnifiedSearch(db_services, embedding_manager, config=None)

        # Mock embedding provider
        embedding_provider = MagicMock()
        embedding_provider.supports_reranking.return_value = False
        embedding_manager.get_provider.return_value = embedding_provider

        # Call unified_search
        context = ResearchContext(root_query="test query")
        await unified_search.unified_search(
            query="test query",
            context=context,
            expanded_queries=["test query"],
        )

        # Verify search_service was called with None (uses multi-hop defaults)
        search_service.search_semantic.assert_called()
        call_kwargs = search_service.search_semantic.call_args.kwargs
        assert call_kwargs["time_limit"] is None
        assert call_kwargs["result_limit"] is None
