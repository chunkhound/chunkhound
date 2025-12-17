"""Unit tests for CompressionService stagnation fixes.

Tests critical bug fixes in the compression algorithm:
1. Bug 2: k-means fallback for single-cluster HDBSCAN output
2. Bug 3: Force compression for oversized single items

These tests verify that the compression algorithm doesn't stagnate when:
- HDBSCAN groups all content into a single cluster
- A single item exceeds the LLM context limit
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


class TestKMeansFallback:
    """Test k-means fallback for single-cluster HDBSCAN output."""

    @pytest.mark.asyncio
    async def test_kmeans_fallback_triggers_for_single_oversized_cluster(
        self, compression_service
    ):
        """When HDBSCAN returns 1 cluster that exceeds context, fall back to k-means.

        The condition `output_count >= input_count` didn't trigger for
        1 cluster from N inputs. The fix adds a check for single oversized clusters.
        """
        from chunkhound.services.clustering_service import ClusteringService

        content_dict = {
            f"file{i}.py": f"def function_{i}(): pass\n" * 1000
            for i in range(5)
        }

        kmeans_called = False

        async def mock_cluster_files_kmeans(files, n_clusters):
            nonlocal kmeans_called
            kmeans_called = True
            file_paths = list(files.keys())
            mid = len(file_paths) // 2
            return [
                ClusterGroup(
                    cluster_id=0,
                    file_paths=file_paths[:mid],
                    files_content={fp: files[fp] for fp in file_paths[:mid]},
                    total_tokens=5000,
                ),
                ClusterGroup(
                    cluster_id=1,
                    file_paths=file_paths[mid:],
                    files_content={fp: files[fp] for fp in file_paths[mid:]},
                    total_tokens=5000,
                ),
            ], {"num_clusters": 2, "avg_tokens_per_cluster": 5000}

        with patch.object(
            ClusteringService,
            "cluster_files",
            return_value=(
                [
                    ClusterGroup(
                        cluster_id=0,
                        file_paths=list(content_dict.keys()),
                        files_content=content_dict,
                        total_tokens=80000,  # Exceeds 75000 final_synthesis_threshold
                    )
                ],
                {"num_clusters": 1, "num_native_clusters": 1, "avg_tokens_per_cluster": 80000},
            ),
        ):
            with patch.object(
                ClusteringService,
                "cluster_files_kmeans",
                side_effect=mock_cluster_files_kmeans,
            ):
                result = await compression_service._cluster_content(
                    content_dict, cluster_budget=10000
                )

                assert kmeans_called, "K-means fallback should be triggered for single oversized cluster"
                assert len(result) == 2, "K-means should produce 2 clusters"


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
