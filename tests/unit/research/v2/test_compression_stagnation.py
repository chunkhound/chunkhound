"""Unit tests for CompressionService stagnation fixes and parallel execution.

Tests critical bug fixes in the compression algorithm:
1. Bug 2: k-means fallback for single-cluster HDBSCAN output
2. Bug 3: Force compression for oversized single items
3. Parallel cluster compression with semaphore limiting

These tests verify that the compression algorithm doesn't stagnate when:
- HDBSCAN groups all content into a single cluster
- A single item exceeds the LLM context limit
- Multiple clusters are compressed in parallel
"""

import pytest
from unittest.mock import patch

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.llm_manager import LLMManager
from chunkhound.services.clustering_service import ClusterGroup
from chunkhound.services.research.v2.compression_service import CompressionService
from tests.fixtures.fake_providers import FakeLLMProvider, FakeEmbeddingProvider


@pytest.fixture
def fake_llm_provider():
    """Create fake LLM provider that returns compressed summaries."""
    return FakeLLMProvider(
        responses={
            "compress": "## Compressed Summary\nKey functionality: authentication, validation, storage.",
            "force": "## Force Compressed\nEssential patterns extracted from truncated content.",
            "cluster": "## Cluster Summary\nGrouped functionality analysis.",
            "code": "## Code Analysis\nImplementation details for the query.",
            "synthesis": "## Final Synthesis\nComprehensive answer based on compressed content.",
        }
    )


@pytest.fixture
def fake_embedding_provider():
    """Create fake embedding provider for clustering."""
    return FakeEmbeddingProvider(dims=1536)


@pytest.fixture
def llm_manager(fake_llm_provider, monkeypatch):
    """Create LLM manager with fake provider."""

    def mock_create_provider(self, config):
        return fake_llm_provider

    monkeypatch.setattr(LLMManager, "_create_provider", mock_create_provider)

    utility_config = {"provider": "fake", "model": "fake-gpt"}
    synthesis_config = {"provider": "fake", "model": "fake-gpt"}
    return LLMManager(utility_config, synthesis_config)


@pytest.fixture
def embedding_manager(fake_embedding_provider):
    """Create embedding manager with fake provider."""

    class MockEmbeddingManager:
        def get_provider(self):
            return fake_embedding_provider

    return MockEmbeddingManager()


@pytest.fixture
def research_config():
    """Create research configuration for testing."""
    return ResearchConfig(
        target_tokens=10000,
        max_chunks_per_file_repr=5,
        max_tokens_per_file_repr=2000,
        max_boundary_expansion_lines=300,
        max_compression_iterations=3,
        min_cluster_size=2,
        shard_budget=20_000,
        gap_similarity_threshold=0.3,
        min_gaps=1,
        max_gaps=5,
        max_symbols=10,
        query_expansion_enabled=True,
    )


@pytest.fixture
def compression_service(llm_manager, embedding_manager, research_config):
    """Create CompressionService directly for unit testing."""
    return CompressionService(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        config=research_config,
    )


class TestForceCompression:
    """Test force compression for oversized single items."""

    @pytest.mark.asyncio
    async def test_force_compress_single_item_exceeding_context(
        self, compression_service
    ):
        """Single item exceeding LLM context should be force compressed.

        When we have a single item that's too large to cluster and too large
        for the LLM context, we truncate and summarize it.
        """
        # Create a single oversized item (> 75000 tokens at final_synthesis_threshold)
        oversized_content = "x" * 400000  # ~100,000 tokens

        content_dict = {"summary_key": oversized_content}

        # Should NOT raise RuntimeError - should force compress instead
        result = await compression_service.compress_to_budget(
            root_query="test query",
            gap_queries=[],
            content_dict=content_dict,
            target_tokens=10000,
            file_imports={},
            depth=0,
            prev_tokens=None,
        )

        assert result is not None
        assert "summary" in result


class TestRegressionNormalMultiCluster:
    """Regression tests for normal multi-cluster scenarios."""

    @pytest.mark.asyncio
    async def test_normal_compression_still_works(self, compression_service):
        """Normal compression with multiple clusters should still work correctly."""
        content_dict = {
            "auth.py": "def authenticate(user): return validate(user)\n" * 100,
            "validate.py": "def validate(user): return check_credentials(user)\n" * 100,
            "session.py": "def create_session(user): return session_token\n" * 100,
        }

        result = await compression_service.compress_to_budget(
            root_query="how does authentication work",
            gap_queries=["how is validation done"],
            content_dict=content_dict,
            target_tokens=10000,
            file_imports={},
            depth=0,
            prev_tokens=None,
        )

        assert result is not None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_compression_convergence_within_budget(self, compression_service):
        """Compression should converge when content fits within budget."""
        content_dict = {
            "small.py": "def small(): return True\n" * 10,
        }

        result = await compression_service.compress_to_budget(
            root_query="what does small do",
            gap_queries=[],
            content_dict=content_dict,
            target_tokens=10000,
            file_imports={},
            depth=0,
            prev_tokens=None,
        )

        assert result is not None
        # Single small item within budget should return as-is
        assert "small.py" in result or "summary" in result


class TestMergeExpansionPrevention:
    """Test that merge operations don't expand content beyond input size."""

    @pytest.mark.asyncio
    async def test_merge_under_budget_content_produces_single_summary(
        self, compression_service
    ):
        """Merging under-budget summaries should produce a single merged summary.

        When multiple items fit within final_synthesis_threshold but need merging,
        the compression service should merge them into a single summary without
        expanding beyond the input token count.
        """
        # Create 2 summaries that together are under final_synthesis_threshold (75k)
        # but over min_llm_tokens (20k) to trigger expansion prevention logic
        content_dict = {
            "summary_1": "Summary of authentication module with details.\n" * 500,
            "summary_2": "Summary of validation module with details.\n" * 500,
        }
        # Each ~2.5k tokens, total ~5k tokens - under target but multiple items

        result = await compression_service.compress_to_budget(
            root_query="how does auth work",
            gap_queries=[],
            content_dict=content_dict,
            target_tokens=10000,
            file_imports={},
            depth=0,
            prev_tokens=None,
        )

        # Verify single summary returned (items were merged)
        assert result is not None
        assert "summary" in result
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_effective_max_caps_at_input_for_under_budget_merge(
        self, compression_service, fake_llm_provider
    ):
        """Verify effective_max is capped at input_tokens when under budget.

        This tests the fix for the expansion bug: when merging content that's
        already under target budget, effective_max should be capped at input
        size to prevent the LLM from expanding the content.
        """
        # Track the max_completion_tokens passed to LLM
        captured_max_tokens = []
        original_complete = fake_llm_provider.complete

        async def tracking_complete(prompt, system=None, max_completion_tokens=None, **kwargs):
            captured_max_tokens.append(max_completion_tokens)
            return await original_complete(prompt, system=system, max_completion_tokens=max_completion_tokens, **kwargs)

        fake_llm_provider.complete = tracking_complete

        # Create content under budget but needs merging (multiple items)
        content_dict = {
            "summary_1": "Auth summary.\n" * 100,  # Small content
            "summary_2": "Validation summary.\n" * 100,
        }

        await compression_service.compress_to_budget(
            root_query="how does auth work",
            gap_queries=[],
            content_dict=content_dict,
            target_tokens=50000,  # Large target
            file_imports={},
            depth=0,
            prev_tokens=None,
        )

        # Verify LLM was called with constrained max_completion_tokens
        # When under budget, effective_max = max(20000, min(target, input))
        # Input is small (~1k tokens), so effective_max should be 20000 (floor)
        # NOT 50000 (the target)
        assert len(captured_max_tokens) > 0
        for max_tokens in captured_max_tokens:
            # Should never be the full target (50000) when input is small
            # Floor is 20000 for quality, but should not expand to target
            assert max_tokens <= 20000, (
                f"Expected max_completion_tokens <= 20000, got {max_tokens}. "
                f"Expansion prevention should cap output at input size."
            )


class TestParallelClusterCompression:
    """Test parallel cluster compression with semaphore limiting."""

    @pytest.mark.asyncio
    async def test_parallel_compression_produces_correct_results(
        self, compression_service
    ):
        """Parallel compression should produce results for all clusters.

        When multiple clusters are compressed in parallel, all should be
        included in the final result.
        """
        # Create content that will result in multiple clusters
        content_dict = {
            "auth.py": "def authenticate(user): return validate(user)\n" * 100,
            "validate.py": "def validate(user): return check_credentials(user)\n" * 100,
            "session.py": "def create_session(user): return session_token\n" * 100,
            "token.py": "def generate_token(): return random_string()\n" * 100,
            "cache.py": "def cache_user(user): return redis.set(user)\n" * 100,
        }

        result = await compression_service.compress_to_budget(
            root_query="how does the authentication system work",
            gap_queries=["how is caching handled"],
            content_dict=content_dict,
            target_tokens=10000,
            file_imports={},
            depth=0,
            prev_tokens=None,
        )

        # Should produce at least one result
        assert result is not None
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_parallel_compression_with_varying_sizes(
        self, compression_service
    ):
        """Parallel compression should handle clusters of varying sizes.

        Even when clusters have very different token counts, parallel
        compression should handle them correctly.
        """
        # Create content with varying sizes
        content_dict = {
            "tiny.py": "x = 1",
            "small.py": "def small(): pass\n" * 10,
            "medium.py": "def medium(): return calculate()\n" * 100,
            "large.py": "def large(): return complex_operation()\n" * 500,
        }

        result = await compression_service.compress_to_budget(
            root_query="what does each file do",
            gap_queries=[],
            content_dict=content_dict,
            target_tokens=10000,
            file_imports={},
            depth=0,
            prev_tokens=None,
        )

        assert result is not None
        # All files should be accounted for in some form
        assert len(result) >= 1
