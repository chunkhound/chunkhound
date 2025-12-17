"""Edge case tests for HDBSCAN noise point handling in gap detection.

HDBSCAN assigns label=-1 to outlier/noise points. These tests verify that
noise points are properly included in gap detection shards and not silently
dropped during clustering.

Test Coverage:
- Noise points are included in shards (not dropped)
- All chunks are noise (fallback behavior)
- Mixed clusters and noise (both processed)
- Chunk count preservation through clustering
"""

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.v2.gap_detection import GapDetectionService
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider


@pytest.fixture
def fake_llm_provider():
    """Create fake LLM provider for gap detection."""
    return FakeLLMProvider(
        responses={
            "gap": '{"gaps": [{"query": "Test gap", "rationale": "Testing", "confidence": 0.8}]}',
        }
    )


@pytest.fixture
def fake_embedding_provider():
    """Create fake embedding provider for chunk clustering."""
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
def db_services():
    """Create mock database services."""

    class MockProvider:
        def get_base_directory(self):
            from pathlib import Path

            return Path("/fake/base")

    class MockDatabaseServices:
        provider = MockProvider()

    return MockDatabaseServices()


@pytest.fixture
def research_config():
    """Create research configuration for testing."""
    return ResearchConfig(
        shard_budget=20_000,
        min_cluster_size=5,  # High enough to force noise points
        gap_similarity_threshold=0.3,
        min_gaps=1,
        max_gaps=5,
        max_symbols=10,
        query_expansion_enabled=True,
        target_tokens=10000,
        max_chunks_per_file_repr=5,
        max_tokens_per_file_repr=2000,
        max_boundary_expansion_lines=300,
        max_compression_iterations=3,
    )


@pytest.fixture
def gap_detection_service(llm_manager, embedding_manager, db_services, research_config):
    """Create gap detection service with mocked dependencies."""
    return GapDetectionService(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=research_config,
    )


class TestNoisePointsIncludedInShards:
    """Test that noise points (label=-1) are included in shards."""

    @pytest.mark.asyncio
    async def test_noise_points_included_in_shards(self, gap_detection_service):
        """Verify chunks with HDBSCAN cluster label=-1 are included in gap detection shards.

        HDBSCAN assigns label=-1 to outlier/noise points. This test ensures noise points
        are not silently dropped but are included in the final shards for gap detection.
        """
        # Create test chunks with embeddings
        chunks = [
            {"chunk_id": f"c{i}", "code": f"code {i}", "file_path": f"file{i}.py"}
            for i in range(10)
        ]

        # Mock HDBSCAN to return mix of clusters and noise points
        # Clusters: 0, 0, 1, 1, -1, -1, 2, 2, -1, -1
        mock_labels = np.array([0, 0, 1, 1, -1, -1, 2, 2, -1, -1])

        with patch("hdbscan.HDBSCAN") as mock_hdbscan_class:
            # Mock HDBSCAN fit_predict to return our controlled labels
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = mock_labels
            mock_hdbscan_class.return_value = mock_clusterer

            # Run clustering
            cluster_groups = await gap_detection_service._cluster_chunks_hdbscan(chunks)

            # Verify: Should have 4 groups (cluster 0, 1, 2, and noise=-1)
            assert len(cluster_groups) == 4, (
                f"Expected 4 cluster groups (3 clusters + 1 noise group), got {len(cluster_groups)}"
            )

            # Verify: Total chunks preserved
            total_chunks = sum(len(group) for group in cluster_groups)
            assert total_chunks == 10, (
                f"Expected all 10 chunks to be included, got {total_chunks}"
            )

            # Verify: Noise group has 4 chunks
            cluster_sizes = sorted([len(group) for group in cluster_groups])
            assert cluster_sizes == [2, 2, 2, 4], (
                f"Expected cluster sizes [2, 2, 2, 4], got {cluster_sizes}"
            )

            # Verify: Noise chunks are in a single group
            noise_chunk_ids = {f"c{i}" for i in [4, 5, 8, 9]}
            for group in cluster_groups:
                group_ids = {chunk["chunk_id"] for chunk in group}
                if len(group) == 4:
                    # This should be the noise group
                    assert group_ids == noise_chunk_ids, (
                        f"Noise group should contain {noise_chunk_ids}, got {group_ids}"
                    )

    @pytest.mark.asyncio
    async def test_noise_points_flow_through_sharding(
        self, gap_detection_service
    ):
        """Verify noise points are preserved through clustering AND sharding.

        Tests the complete flow: chunks -> clustering -> sharding.
        Ensures noise points aren't dropped at the sharding step.
        """
        # Create test chunks
        chunks = [
            {"chunk_id": f"c{i}", "code": f"x" * 50, "file_path": f"file{i}.py"}
            for i in range(8)
        ]

        # Mock HDBSCAN to return noise points: [0, 0, -1, -1, 1, 1, -1, -1]
        mock_labels = np.array([0, 0, -1, -1, 1, 1, -1, -1])

        with patch("hdbscan.HDBSCAN") as mock_hdbscan_class:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = mock_labels
            mock_hdbscan_class.return_value = mock_clusterer

            # Run clustering
            cluster_groups = await gap_detection_service._cluster_chunks_hdbscan(chunks)

            # Run sharding
            shards = gap_detection_service._shard_by_tokens(cluster_groups)

            # Verify: All chunks still present after sharding
            total_chunks_in_shards = sum(len(shard) for shard in shards)
            assert total_chunks_in_shards == 8, (
                f"Expected all 8 chunks in shards, got {total_chunks_in_shards}"
            )

            # Verify: Noise chunks are present (c2, c3, c6, c7)
            all_chunk_ids = {
                chunk["chunk_id"] for shard in shards for chunk in shard
            }
            noise_chunk_ids = {"c2", "c3", "c6", "c7"}
            assert noise_chunk_ids.issubset(all_chunk_ids), (
                f"Noise chunks {noise_chunk_ids} not found in shards {all_chunk_ids}"
            )


class TestAllChunksNoiseFallback:
    """Test behavior when all chunks are classified as noise."""

    @pytest.mark.asyncio
    async def test_all_chunks_noise_fallback(self, gap_detection_service):
        """When all chunks are noise (min_cluster_size too high), verify fallback behavior.

        HDBSCAN may classify all chunks as noise when min_cluster_size is too large
        relative to dataset size. Verify graceful fallback (single cluster).
        """
        # Create test chunks
        chunks = [
            {"chunk_id": f"c{i}", "code": f"unique code {i}", "file_path": f"file{i}.py"}
            for i in range(3)  # Small dataset
        ]

        # Mock HDBSCAN to return all noise: [-1, -1, -1]
        mock_labels = np.array([-1, -1, -1])

        with patch("hdbscan.HDBSCAN") as mock_hdbscan_class:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = mock_labels
            mock_hdbscan_class.return_value = mock_clusterer

            # Run clustering
            cluster_groups = await gap_detection_service._cluster_chunks_hdbscan(chunks)

            # Verify: Should have exactly 1 group (all noise)
            assert len(cluster_groups) == 1, (
                f"Expected 1 cluster group (all noise), got {len(cluster_groups)}"
            )

            # Verify: All chunks are in the noise group
            assert len(cluster_groups[0]) == 3, (
                f"Expected 3 chunks in noise group, got {len(cluster_groups[0])}"
            )

            # Verify: All chunk IDs present
            chunk_ids = {chunk["chunk_id"] for chunk in cluster_groups[0]}
            expected_ids = {"c0", "c1", "c2"}
            assert chunk_ids == expected_ids, (
                f"Expected {expected_ids}, got {chunk_ids}"
            )

    @pytest.mark.asyncio
    async def test_all_noise_chunks_can_be_sharded(self, gap_detection_service):
        """Verify that all-noise cluster can be sharded without errors.

        Tests edge case where HDBSCAN returns only noise points and verifies
        sharding doesn't crash or drop chunks.
        """
        chunks = [
            {"chunk_id": f"c{i}", "code": "x" * 1000, "file_path": f"file{i}.py"}
            for i in range(5)
        ]

        # Mock HDBSCAN to return all noise
        mock_labels = np.array([-1, -1, -1, -1, -1])

        with patch("hdbscan.HDBSCAN") as mock_hdbscan_class:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = mock_labels
            mock_hdbscan_class.return_value = mock_clusterer

            # Run clustering
            cluster_groups = await gap_detection_service._cluster_chunks_hdbscan(chunks)

            # Run sharding
            shards = gap_detection_service._shard_by_tokens(cluster_groups)

            # Verify: Sharding succeeded
            assert len(shards) >= 1, "Expected at least 1 shard"

            # Verify: All chunks present in shards
            total_chunks = sum(len(shard) for shard in shards)
            assert total_chunks == 5, f"Expected 5 chunks in shards, got {total_chunks}"


class TestMixedClustersAndNoise:
    """Test handling of mixed valid clusters and noise points."""

    @pytest.mark.asyncio
    async def test_mixed_clusters_and_noise_both_processed(
        self, gap_detection_service
    ):
        """Verify both clustered and noise chunks are processed in gap detection.

        Tests realistic scenario where HDBSCAN finds both valid clusters and
        noise points. Ensures both categories are preserved and processed.
        """
        # Create test chunks
        chunks = [
            {"chunk_id": f"c{i}", "code": f"code {i}", "file_path": f"file{i}.py"}
            for i in range(12)
        ]

        # Mock HDBSCAN: 3 valid clusters + noise points
        # Cluster 0: c0, c1, c2, c3 (4 chunks)
        # Cluster 1: c4, c5, c6 (3 chunks)
        # Cluster 2: c7, c8 (2 chunks)
        # Noise: c9, c10, c11 (3 chunks)
        mock_labels = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, -1, -1, -1])

        with patch("hdbscan.HDBSCAN") as mock_hdbscan_class:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = mock_labels
            mock_hdbscan_class.return_value = mock_clusterer

            # Run clustering
            cluster_groups = await gap_detection_service._cluster_chunks_hdbscan(chunks)

            # Verify: 4 groups (3 clusters + 1 noise group)
            assert len(cluster_groups) == 4, (
                f"Expected 4 cluster groups, got {len(cluster_groups)}"
            )

            # Verify: Total chunks preserved
            total_chunks = sum(len(group) for group in cluster_groups)
            assert total_chunks == 12, (
                f"Expected all 12 chunks, got {total_chunks}"
            )

            # Verify: Cluster sizes match expected
            cluster_sizes = sorted([len(group) for group in cluster_groups])
            assert cluster_sizes == [2, 3, 3, 4], (
                f"Expected cluster sizes [2, 3, 3, 4], got {cluster_sizes}"
            )

            # Verify: Each cluster has expected chunks
            all_chunk_ids = set()
            for group in cluster_groups:
                group_ids = {chunk["chunk_id"] for chunk in group}
                all_chunk_ids.update(group_ids)

            expected_ids = {f"c{i}" for i in range(12)}
            assert all_chunk_ids == expected_ids, (
                f"Expected {expected_ids}, got {all_chunk_ids}"
            )

    @pytest.mark.asyncio
    async def test_noise_and_clusters_maintain_separation(
        self, gap_detection_service
    ):
        """Verify noise chunks remain separate from valid clusters.

        Tests that noise points (label=-1) are grouped separately from
        valid clusters (label>=0), maintaining HDBSCAN's semantic grouping.
        """
        chunks = [
            {"chunk_id": f"c{i}", "code": f"code {i}", "file_path": f"file{i}.py"}
            for i in range(6)
        ]

        # Mock HDBSCAN: 1 cluster + noise
        # Cluster 0: c0, c1, c2
        # Noise: c3, c4, c5
        mock_labels = np.array([0, 0, 0, -1, -1, -1])

        with patch("hdbscan.HDBSCAN") as mock_hdbscan_class:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = mock_labels
            mock_hdbscan_class.return_value = mock_clusterer

            # Run clustering
            cluster_groups = await gap_detection_service._cluster_chunks_hdbscan(chunks)

            # Verify: 2 groups (1 cluster + 1 noise)
            assert len(cluster_groups) == 2, (
                f"Expected 2 groups, got {len(cluster_groups)}"
            )

            # Verify: Groups have expected sizes
            group_sizes = sorted([len(group) for group in cluster_groups])
            assert group_sizes == [3, 3], (
                f"Expected group sizes [3, 3], got {group_sizes}"
            )

            # Verify: Groups contain expected chunks
            cluster_chunks = {"c0", "c1", "c2"}
            noise_chunks = {"c3", "c4", "c5"}

            for group in cluster_groups:
                group_ids = {chunk["chunk_id"] for chunk in group}
                # Group should match either cluster or noise, not mix
                assert group_ids == cluster_chunks or group_ids == noise_chunks, (
                    f"Group {group_ids} doesn't match cluster {cluster_chunks} "
                    f"or noise {noise_chunks}"
                )


class TestChunkCountPreservation:
    """Test that chunk count is preserved through clustering and sharding."""

    @pytest.mark.asyncio
    async def test_noise_points_not_silently_dropped(self, gap_detection_service):
        """Count chunks before/after clustering, verify none lost.

        Critical test: Ensures noise points are not silently dropped anywhere
        in the clustering pipeline. Total chunk count must be preserved.
        """
        # Create test chunks
        input_chunks = [
            {"chunk_id": f"c{i}", "code": f"code {i}", "file_path": f"file{i}.py"}
            for i in range(15)
        ]

        # Mock HDBSCAN with mix of clusters and noise
        # Random distribution: some clusters, some noise
        mock_labels = np.array([0, 0, 0, 1, 1, -1, -1, 2, 2, 2, -1, 3, 3, -1, -1])

        with patch("hdbscan.HDBSCAN") as mock_hdbscan_class:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = mock_labels
            mock_hdbscan_class.return_value = mock_clusterer

            # Count chunks BEFORE clustering
            chunks_before = len(input_chunks)

            # Run clustering
            cluster_groups = await gap_detection_service._cluster_chunks_hdbscan(
                input_chunks
            )

            # Count chunks AFTER clustering
            chunks_after = sum(len(group) for group in cluster_groups)

            # CRITICAL ASSERTION: No chunks dropped
            assert chunks_before == chunks_after, (
                f"Chunk count mismatch: before={chunks_before}, after={chunks_after}. "
                f"Noise points may have been silently dropped!"
            )

            # Verify: All original chunk IDs present
            input_ids = {chunk["chunk_id"] for chunk in input_chunks}
            output_ids = {
                chunk["chunk_id"] for group in cluster_groups for chunk in group
            }
            assert input_ids == output_ids, (
                f"Chunk IDs changed. Missing: {input_ids - output_ids}, "
                f"Extra: {output_ids - input_ids}"
            )

    @pytest.mark.asyncio
    async def test_chunk_preservation_through_full_pipeline(
        self, gap_detection_service
    ):
        """Verify chunk count preservation through clustering AND sharding.

        Tests complete pipeline: chunks -> clustering -> sharding.
        Ensures no chunks are lost at any stage.
        """
        # Create test chunks
        input_chunks = [
            {"chunk_id": f"c{i}", "code": f"x" * 100, "file_path": f"file{i}.py"}
            for i in range(20)
        ]

        # Mock HDBSCAN with realistic distribution
        # 4 clusters + noise points
        mock_labels = np.array([
            0, 0, 0, 0, 0,          # Cluster 0 (5 chunks)
            1, 1, 1, 1,             # Cluster 1 (4 chunks)
            -1, -1, -1,             # Noise (3 chunks)
            2, 2, 2,                # Cluster 2 (3 chunks)
            -1, -1,                 # Noise (2 chunks)
            3, 3, 3,                # Cluster 3 (3 chunks)
        ])

        with patch("hdbscan.HDBSCAN") as mock_hdbscan_class:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = mock_labels
            mock_hdbscan_class.return_value = mock_clusterer

            # Count BEFORE pipeline
            chunks_before = len(input_chunks)

            # Run clustering
            cluster_groups = await gap_detection_service._cluster_chunks_hdbscan(
                input_chunks
            )

            # Count AFTER clustering
            chunks_after_clustering = sum(len(group) for group in cluster_groups)
            assert chunks_before == chunks_after_clustering, (
                f"Clustering lost chunks: {chunks_before} -> {chunks_after_clustering}"
            )

            # Run sharding
            shards = gap_detection_service._shard_by_tokens(cluster_groups)

            # Count AFTER sharding
            chunks_after_sharding = sum(len(shard) for shard in shards)
            assert chunks_before == chunks_after_sharding, (
                f"Sharding lost chunks: {chunks_before} -> {chunks_after_sharding}"
            )

            # Verify: All chunk IDs preserved
            input_ids = {chunk["chunk_id"] for chunk in input_chunks}
            output_ids = {
                chunk["chunk_id"] for shard in shards for chunk in shard
            }
            assert input_ids == output_ids, (
                f"Chunk IDs not preserved through pipeline. "
                f"Missing: {input_ids - output_ids}, Extra: {output_ids - input_ids}"
            )

    @pytest.mark.asyncio
    async def test_no_duplicate_chunks_after_clustering(
        self, gap_detection_service
    ):
        """Verify clustering doesn't duplicate chunks (each chunk in exactly one group).

        Tests that HDBSCAN clustering properly partitions chunks - each chunk
        should appear in exactly one cluster group.
        """
        chunks = [
            {"chunk_id": f"c{i}", "code": f"code {i}", "file_path": f"file{i}.py"}
            for i in range(10)
        ]

        # Mock HDBSCAN
        mock_labels = np.array([0, 0, 1, 1, -1, -1, 2, 2, -1, 0])

        with patch("hdbscan.HDBSCAN") as mock_hdbscan_class:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = mock_labels
            mock_hdbscan_class.return_value = mock_clusterer

            # Run clustering
            cluster_groups = await gap_detection_service._cluster_chunks_hdbscan(chunks)

            # Collect all chunk IDs from all groups
            all_chunk_ids = []
            for group in cluster_groups:
                all_chunk_ids.extend([chunk["chunk_id"] for chunk in group])

            # Verify: No duplicates (each chunk in exactly one group)
            assert len(all_chunk_ids) == len(set(all_chunk_ids)), (
                f"Found duplicate chunks after clustering: "
                f"{len(all_chunk_ids)} total, {len(set(all_chunk_ids))} unique"
            )

            # Verify: Count matches input
            assert len(all_chunk_ids) == len(chunks), (
                f"Chunk count mismatch: input={len(chunks)}, output={len(all_chunk_ids)}"
            )
