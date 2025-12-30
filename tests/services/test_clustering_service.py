"""Tests for clustering service.

Tests k-means clustering with specified number of clusters and
HDBSCAN clustering for natural semantic grouping.
"""

import pytest

from chunkhound.services.clustering_service import ClusteringService, ClusterGroup
from tests.fixtures.fake_providers import FakeLLMProvider, FakeEmbeddingProvider


class TestKMeansClustering:
    """Test k-means clustering."""

    @pytest.fixture
    def fake_llm_provider(self) -> FakeLLMProvider:
        """Create fake LLM provider for token estimation."""
        return FakeLLMProvider(model="fake-gpt")

    @pytest.fixture
    def fake_embedding_provider(self) -> FakeEmbeddingProvider:
        """Create fake embedding provider with predictable embeddings."""
        return FakeEmbeddingProvider(model="fake-embed")

    @pytest.fixture
    def clustering_service(
        self, fake_llm_provider: FakeLLMProvider, fake_embedding_provider: FakeEmbeddingProvider
    ) -> ClusteringService:
        """Create clustering service with fake providers."""
        return ClusteringService(
            embedding_provider=fake_embedding_provider,
            llm_provider=fake_llm_provider,
        )

    @pytest.mark.asyncio
    async def test_single_file_creates_single_cluster(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that single file with n_clusters=1 creates one cluster."""
        files = {"file1.py": "def test(): pass"}

        clusters, metadata = await clustering_service.cluster_files(files, n_clusters=1)

        assert len(clusters) == 1
        assert clusters[0].cluster_id == 0
        assert clusters[0].file_paths == ["file1.py"]
        assert metadata["num_clusters"] == 1
        assert metadata["total_files"] == 1

    @pytest.mark.asyncio
    async def test_small_files_single_cluster(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that files under token budget create single cluster when n_clusters=1."""
        # Files with total ~200 tokens (well under 30k limit)
        files = {f"file{i}.py": "def test(): pass\n" * 10 for i in range(5)}

        clusters, metadata = await clustering_service.cluster_files(files, n_clusters=1)

        assert len(clusters) == 1
        assert metadata["num_clusters"] == 1
        assert metadata["total_files"] == 5
        assert sum(len(c.file_paths) for c in clusters) == 5

    @pytest.mark.asyncio
    async def test_kmeans_creates_requested_clusters(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that k-means creates the requested number of clusters."""
        # Create 9 files
        files = {f"file{i}.py": f"content {i}" * 100 for i in range(9)}

        # Request 3 clusters
        clusters, metadata = await clustering_service.cluster_files(files, n_clusters=3)

        # Should create exactly 3 clusters
        assert len(clusters) == 3
        assert metadata["num_clusters"] == 3
        assert metadata["total_files"] == 9

        # All files accounted for
        assert sum(len(c.file_paths) for c in clusters) == 9

    @pytest.mark.asyncio
    async def test_empty_files_raises_error(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that empty files dict raises ValueError."""
        with pytest.raises(ValueError, match="Cannot cluster empty files"):
            await clustering_service.cluster_files({}, n_clusters=1)

    @pytest.mark.asyncio
    async def test_all_files_accounted_for(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that all input files appear in exactly one cluster."""
        files = {f"file{i}.py": f"content {i}" * 100 for i in range(10)}

        clusters, metadata = await clustering_service.cluster_files(files, n_clusters=3)

        # Collect all files from all clusters
        clustered_files = set()
        for cluster in clusters:
            clustered_files.update(cluster.file_paths)

        # Verify all input files are present
        assert clustered_files == set(files.keys())
        assert metadata["total_files"] == len(files)

    @pytest.mark.asyncio
    async def test_metadata_structure(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that metadata has expected structure."""
        files = {f"file{i}.py": f"content {i}" * 100 for i in range(5)}

        clusters, metadata = await clustering_service.cluster_files(files, n_clusters=2)

        # Verify expected metadata fields present
        assert "num_clusters" in metadata
        assert "total_files" in metadata
        assert "total_tokens" in metadata
        assert "avg_tokens_per_cluster" in metadata

        # Verify no HDBSCAN-specific fields
        assert "num_native_clusters" not in metadata
        assert "num_outliers" not in metadata

        # Verify values are reasonable
        assert metadata["num_clusters"] == 2
        assert metadata["total_files"] == 5
        assert metadata["total_tokens"] > 0
        assert metadata["avg_tokens_per_cluster"] > 0

    @pytest.mark.asyncio
    async def test_cluster_groups_have_content(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that ClusterGroup objects contain file content."""
        files = {"file1.py": "content1", "file2.py": "content2"}

        clusters, _ = await clustering_service.cluster_files(files, n_clusters=1)

        assert len(clusters) == 1
        cluster = clusters[0]

        # Verify cluster structure
        assert isinstance(cluster, ClusterGroup)
        assert isinstance(cluster.cluster_id, int)
        assert isinstance(cluster.file_paths, list)
        assert isinstance(cluster.files_content, dict)
        assert isinstance(cluster.total_tokens, int)

        # Verify content matches input
        for file_path in cluster.file_paths:
            assert file_path in cluster.files_content
            assert cluster.files_content[file_path] == files[file_path]

    @pytest.mark.asyncio
    async def test_token_counting_accuracy(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that token counts in metadata are accurate."""
        files = {f"file{i}.py": "test content" * 10 for i in range(5)}

        clusters, metadata = await clustering_service.cluster_files(files, n_clusters=2)

        # Calculate total from clusters
        cluster_total = sum(c.total_tokens for c in clusters)

        # Should match metadata total
        assert cluster_total == metadata["total_tokens"]

        # Average should be reasonable
        expected_avg = metadata["total_tokens"] / len(clusters)
        assert abs(metadata["avg_tokens_per_cluster"] - expected_avg) <= 1  # Allow rounding error

    @pytest.mark.asyncio
    async def test_budget_tracking_with_multiple_clusters(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that clusters track tokens correctly across multiple clusters."""
        # Create files with varying sizes
        files = {
            "small1.py": "x" * 100,
            "small2.py": "x" * 150,
            "medium1.py": "x" * 500,
            "medium2.py": "x" * 600,
            "large1.py": "x" * 1000,
            "large2.py": "x" * 1200,
        }

        clusters, metadata = await clustering_service.cluster_files(files, n_clusters=2)

        # All files should be in clusters
        total_files_in_clusters = sum(len(c.file_paths) for c in clusters)
        assert total_files_in_clusters == 6

        # Each cluster should have at least one file
        for cluster in clusters:
            assert len(cluster.file_paths) > 0
            assert cluster.total_tokens > 0


class TestClusterGroup:
    """Test ClusterGroup dataclass."""

    def test_cluster_group_creation(self) -> None:
        """Test creating ClusterGroup instance."""
        cluster = ClusterGroup(
            cluster_id=0,
            file_paths=["file1.py", "file2.py"],
            files_content={"file1.py": "content1", "file2.py": "content2"},
            total_tokens=100,
        )

        assert cluster.cluster_id == 0
        assert len(cluster.file_paths) == 2
        assert len(cluster.files_content) == 2
        assert cluster.total_tokens == 100

    def test_cluster_group_equality(self) -> None:
        """Test ClusterGroup equality comparison (dataclass feature)."""
        cluster1 = ClusterGroup(
            cluster_id=0,
            file_paths=["file1.py"],
            files_content={"file1.py": "content"},
            total_tokens=50,
        )
        cluster2 = ClusterGroup(
            cluster_id=0,
            file_paths=["file1.py"],
            files_content={"file1.py": "content"},
            total_tokens=50,
        )

        assert cluster1 == cluster2


class TestHDBSCANClustering:
    """Test HDBSCAN clustering with outlier reassignment."""

    @pytest.fixture
    def fake_llm_provider(self) -> FakeLLMProvider:
        """Create fake LLM provider for token estimation."""
        return FakeLLMProvider(model="fake-gpt")

    @pytest.fixture
    def fake_embedding_provider(self) -> FakeEmbeddingProvider:
        """Create fake embedding provider with predictable embeddings."""
        return FakeEmbeddingProvider(model="fake-embed")

    @pytest.fixture
    def clustering_service(
        self, fake_llm_provider: FakeLLMProvider, fake_embedding_provider: FakeEmbeddingProvider
    ) -> ClusteringService:
        """Create clustering service with fake providers."""
        return ClusteringService(
            embedding_provider=fake_embedding_provider,
            llm_provider=fake_llm_provider,
        )

    @pytest.mark.asyncio
    async def test_hdbscan_single_file_creates_single_cluster(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that single file creates one cluster."""
        files = {"file1.py": "def test(): pass"}

        clusters, metadata = await clustering_service.cluster_files_hdbscan(files)

        assert len(clusters) == 1
        assert clusters[0].cluster_id == 0
        assert clusters[0].file_paths == ["file1.py"]
        assert metadata["num_clusters"] == 1
        assert metadata["num_native_clusters"] == 1
        assert metadata["num_outliers"] == 0
        assert metadata["total_files"] == 1

    @pytest.mark.asyncio
    async def test_hdbscan_discovers_clusters(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that HDBSCAN discovers natural clusters."""
        # Create files - HDBSCAN will find clusters based on embeddings
        files = {f"file{i}.py": f"content {i}" * 100 for i in range(10)}

        clusters, metadata = await clustering_service.cluster_files_hdbscan(files)

        # Should create at least 1 cluster
        assert len(clusters) >= 1
        assert metadata["num_clusters"] >= 1

        # All files accounted for
        assert sum(len(c.file_paths) for c in clusters) == 10
        assert metadata["total_files"] == 10

    @pytest.mark.asyncio
    async def test_hdbscan_metadata_includes_outlier_info(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that HDBSCAN metadata includes outlier information."""
        files = {f"file{i}.py": f"content {i}" * 100 for i in range(5)}

        clusters, metadata = await clustering_service.cluster_files_hdbscan(files)

        # Verify HDBSCAN-specific metadata fields present
        assert "num_clusters" in metadata
        assert "num_native_clusters" in metadata
        assert "num_outliers" in metadata
        assert "total_files" in metadata
        assert "total_tokens" in metadata
        assert "avg_tokens_per_cluster" in metadata

        # num_outliers should be non-negative
        assert metadata["num_outliers"] >= 0

        # num_native_clusters should be at least 0
        assert metadata["num_native_clusters"] >= 0

    @pytest.mark.asyncio
    async def test_hdbscan_all_files_accounted_for(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that all input files appear in exactly one cluster after outlier reassignment."""
        files = {f"file{i}.py": f"unique content {i}" * 50 for i in range(15)}

        clusters, metadata = await clustering_service.cluster_files_hdbscan(files)

        # Collect all files from all clusters
        clustered_files = set()
        for cluster in clusters:
            clustered_files.update(cluster.file_paths)

        # Verify all input files are present (no files dropped as outliers)
        assert clustered_files == set(files.keys())
        assert metadata["total_files"] == len(files)

    @pytest.mark.asyncio
    async def test_hdbscan_empty_files_raises_error(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that empty files dict raises ValueError."""
        with pytest.raises(ValueError, match="Cannot cluster empty files"):
            await clustering_service.cluster_files_hdbscan({})

    @pytest.mark.asyncio
    async def test_hdbscan_cluster_groups_have_content(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that ClusterGroup objects contain file content."""
        files = {"file1.py": "content1", "file2.py": "content2", "file3.py": "content3"}

        clusters, _ = await clustering_service.cluster_files_hdbscan(files)

        # All clusters should have proper structure
        for cluster in clusters:
            assert isinstance(cluster, ClusterGroup)
            assert isinstance(cluster.cluster_id, int)
            assert isinstance(cluster.file_paths, list)
            assert isinstance(cluster.files_content, dict)
            assert isinstance(cluster.total_tokens, int)

            # Verify content matches input
            for file_path in cluster.file_paths:
                assert file_path in cluster.files_content
                assert cluster.files_content[file_path] == files[file_path]

    @pytest.mark.asyncio
    async def test_hdbscan_no_negative_labels_in_output(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that no clusters have negative IDs (outliers reassigned)."""
        files = {f"file{i}.py": f"content {i}" * 100 for i in range(10)}

        clusters, _ = await clustering_service.cluster_files_hdbscan(files)

        # All cluster IDs should be non-negative (outliers reassigned)
        for cluster in clusters:
            assert cluster.cluster_id >= 0

    @pytest.mark.asyncio
    async def test_hdbscan_token_counting_accuracy(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that token counts in metadata are accurate."""
        files = {f"file{i}.py": "test content" * 10 for i in range(5)}

        clusters, metadata = await clustering_service.cluster_files_hdbscan(files)

        # Calculate total from clusters
        cluster_total = sum(c.total_tokens for c in clusters)

        # Should match metadata total
        assert cluster_total == metadata["total_tokens"]

        # Average should be reasonable
        expected_avg = metadata["total_tokens"] / len(clusters)
        assert abs(metadata["avg_tokens_per_cluster"] - expected_avg) <= 1


class TestClusterFilesHDBSCANBounded:
    """Tests for HDBSCAN clustering with token bounds enforcement."""

    @pytest.fixture
    def fake_llm_provider(self) -> FakeLLMProvider:
        """Create fake LLM provider for token estimation."""
        return FakeLLMProvider(model="fake-gpt")

    @pytest.fixture
    def fake_embedding_provider(self) -> FakeEmbeddingProvider:
        """Create fake embedding provider with predictable embeddings."""
        return FakeEmbeddingProvider(model="fake-embed")

    @pytest.fixture
    def clustering_service(
        self, fake_llm_provider: FakeLLMProvider, fake_embedding_provider: FakeEmbeddingProvider
    ) -> ClusteringService:
        """Create clustering service with fake providers."""
        return ClusteringService(
            embedding_provider=fake_embedding_provider,
            llm_provider=fake_llm_provider,
        )

    def _make_content_with_tokens(self, target_tokens: int) -> str:
        """Create content that estimates to approximately target_tokens.

        FakeLLMProvider uses len(text) // 4 for token estimation.
        """
        # 4 chars per token
        return "x" * (target_tokens * 4)

    def _make_unique_content_with_tokens(self, target_tokens: int, seed: str) -> str:
        """Create unique content that estimates to approximately target_tokens.

        Uses seed to ensure different files produce different embeddings.
        FakeEmbeddingProvider uses hash-based vectors, so different content
        produces different embeddings.
        """
        # 4 chars per token, but use varied characters based on seed
        base_char = seed[0] if seed else "x"
        prefix = f"# {seed}\n"
        remaining_chars = (target_tokens * 4) - len(prefix)
        return prefix + base_char * remaining_chars

    @pytest.mark.asyncio
    async def test_splitting_oversized_cluster(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that clusters exceeding max_tokens get split into subclusters."""
        # Create files with distinct embeddings but similar enough for HDBSCAN clustering
        # Use 6 files of ~20k tokens each = 120k total
        # With 50k max, we need at least 3 clusters, so split should produce 4+ subclusters
        files = {}
        for i in range(6):
            # Each file has distinctly different content for k-means to separate
            unique_content = f"unique_module_{i}_" + ("a" * i * 1000) + self._make_content_with_tokens(19_900)
            files[f"large{i}.py"] = unique_content

        clusters, metadata = await clustering_service.cluster_files_hdbscan_bounded(
            files,
            min_tokens_per_cluster=10_000,
            max_tokens_per_cluster=50_000,
        )

        # With ~120k tokens total and 50k max, splitting should occur
        # At least one split should have been attempted
        assert metadata["num_splits"] >= 1, (
            f"Expected at least 1 split, got {metadata['num_splits']}. "
            f"Native clusters: {metadata['num_native_clusters']}"
        )

        # With 6 files of ~20k each, we should get at least 3 clusters
        # (k-means splits should distribute ~40k per cluster target)
        assert len(clusters) >= 2, (
            f"Expected at least 2 clusters after split, got {len(clusters)}. "
            f"Splits: {metadata['num_splits']}"
        )

        # All files should be accounted for
        all_files = set()
        for cluster in clusters:
            all_files.update(cluster.file_paths)
        assert all_files == set(files.keys())

        # Most clusters should be close to bounds
        # (k-means doesn't guarantee exact bounds, but should reduce average)
        avg_tokens = sum(c.total_tokens for c in clusters) / len(clusters)
        assert avg_tokens < 60_000, (
            f"Average cluster tokens {avg_tokens:.0f} should be reduced by splitting"
        )

    @pytest.mark.asyncio
    async def test_merging_undersized_clusters(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that clusters below min_tokens get merged into nearest neighbor."""
        # Create files with small token counts (~5k tokens each, below 15k minimum)
        # Use very different content to create distinct embeddings for multiple clusters
        # Total: 6 files * 5k = 30k tokens
        files = {}
        for i in range(6):
            # Each file has unique content to create distinct embeddings
            unique_content = f"unique_module_{i}_" + self._make_content_with_tokens(4_990)
            files[f"small_{i}.py"] = unique_content

        clusters, metadata = await clustering_service.cluster_files_hdbscan_bounded(
            files,
            min_tokens_per_cluster=15_000,
            max_tokens_per_cluster=100_000,
        )

        # All resulting clusters should meet minimum threshold
        # Note: If all files are in one cluster from HDBSCAN, no merges needed
        # If multiple clusters form, they must meet threshold after merging
        for cluster in clusters:
            assert cluster.total_tokens >= 15_000 or len(clusters) == 1, (
                f"Cluster {cluster.cluster_id} has {cluster.total_tokens} tokens, "
                f"below min of 15,000 and there are {len(clusters)} clusters"
            )

        # If HDBSCAN produced multiple native clusters, merges should occur
        if metadata["num_native_clusters"] > 1:
            # Either merges occurred or the single cluster exception applies
            assert metadata["num_merges"] >= 1 or len(clusters) == 1

        # All files should be accounted for
        all_files = set()
        for cluster in clusters:
            all_files.update(cluster.file_paths)
        assert all_files == set(files.keys())

    @pytest.mark.asyncio
    async def test_no_changes_when_within_bounds(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that clusters already within bounds are not modified."""
        # Create files with ~25k tokens each (within 15k-50k bounds)
        medium_content = self._make_content_with_tokens(25_000)
        files = {
            f"medium{i}.py": medium_content for i in range(2)
        }

        clusters, metadata = await clustering_service.cluster_files_hdbscan_bounded(
            files,
            min_tokens_per_cluster=15_000,
            max_tokens_per_cluster=50_000,
        )

        # No splits or merges should occur
        assert metadata["num_splits"] == 0
        assert metadata["num_merges"] == 0

        # All clusters within bounds
        for cluster in clusters:
            assert 15_000 <= cluster.total_tokens <= 50_000

    @pytest.mark.asyncio
    async def test_single_undersized_cluster_passes_through(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that a single undersized cluster cannot be merged (no target)."""
        # Single small file with ~5k tokens
        small_content = self._make_content_with_tokens(5_000)
        files = {"single.py": small_content}

        clusters, metadata = await clustering_service.cluster_files_hdbscan_bounded(
            files,
            min_tokens_per_cluster=15_000,
            max_tokens_per_cluster=50_000,
        )

        # Should still produce one cluster
        assert len(clusters) == 1
        assert metadata["num_clusters"] == 1

        # No merges possible (only one cluster)
        assert metadata["num_merges"] == 0

        # The single cluster passes through unchanged
        assert clusters[0].file_paths == ["single.py"]

    @pytest.mark.asyncio
    async def test_all_files_merge_into_single_cluster(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that when total tokens < min_threshold, all end up in one cluster."""
        # Create files with unique content to potentially form different clusters
        # Total: ~10k tokens, min_threshold: 15k (none can meet threshold alone)
        files = {}
        for i in range(5):
            unique_content = f"tiny_unique_{i}_" + self._make_content_with_tokens(1_990)
            files[f"tiny{i}.py"] = unique_content

        clusters, metadata = await clustering_service.cluster_files_hdbscan_bounded(
            files,
            min_tokens_per_cluster=15_000,
            max_tokens_per_cluster=50_000,
        )

        # Should end up with a single cluster (total tokens < min_threshold)
        assert len(clusters) == 1, f"Expected 1 cluster, got {len(clusters)}"
        assert metadata["num_clusters"] == 1

        # All files in that cluster
        assert len(clusters[0].file_paths) == 5

        # If HDBSCAN created multiple native clusters, merges should have occurred
        # If HDBSCAN created one cluster, no merges needed
        if metadata["num_native_clusters"] > 1:
            assert metadata["num_merges"] >= 1

    @pytest.mark.asyncio
    async def test_metadata_accuracy(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that returned metadata accurately reflects operations performed."""
        # Create mix of file sizes to trigger both splits and merges
        large_content = self._make_content_with_tokens(60_000)  # Will need split
        small_content = self._make_content_with_tokens(5_000)   # Will need merge

        files = {
            "large.py": large_content,
            "small1.py": small_content,
            "small2.py": small_content,
        }
        # Total: 70k tokens

        clusters, metadata = await clustering_service.cluster_files_hdbscan_bounded(
            files,
            min_tokens_per_cluster=15_000,
            max_tokens_per_cluster=50_000,
        )

        # Verify metadata fields exist
        assert "num_clusters" in metadata
        assert "num_native_clusters" in metadata
        assert "num_outliers" in metadata
        assert "num_splits" in metadata
        assert "num_merges" in metadata
        assert "total_files" in metadata
        assert "total_tokens" in metadata
        assert "avg_tokens_per_cluster" in metadata

        # Verify total_tokens matches sum of all file tokens
        expected_total = sum(
            clustering_service._llm_provider.estimate_tokens(content)
            for content in files.values()
        )
        assert metadata["total_tokens"] == expected_total

        # Verify total_files
        assert metadata["total_files"] == len(files)

        # Verify num_clusters matches actual cluster count
        assert metadata["num_clusters"] == len(clusters)

        # Verify avg_tokens_per_cluster
        expected_avg = metadata["total_tokens"] / len(clusters)
        assert abs(metadata["avg_tokens_per_cluster"] - int(expected_avg)) <= 1

        # Verify cluster tokens sum matches total
        cluster_tokens_sum = sum(c.total_tokens for c in clusters)
        assert cluster_tokens_sum == metadata["total_tokens"]

    @pytest.mark.asyncio
    async def test_empty_files_raises_error(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that empty files dict raises ValueError."""
        with pytest.raises(ValueError, match="Cannot cluster empty files"):
            await clustering_service.cluster_files_hdbscan_bounded({})

    @pytest.mark.asyncio
    async def test_all_files_accounted_for_after_bounds_enforcement(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that all input files appear in output after splits and merges."""
        # Mix of sizes to trigger various operations
        files = {
            "huge.py": self._make_content_with_tokens(80_000),   # Split needed
            "medium.py": self._make_content_with_tokens(30_000), # OK
            "small.py": self._make_content_with_tokens(5_000),   # Merge needed
        }

        clusters, metadata = await clustering_service.cluster_files_hdbscan_bounded(
            files,
            min_tokens_per_cluster=15_000,
            max_tokens_per_cluster=50_000,
        )

        # Collect all files from all clusters
        clustered_files = set()
        for cluster in clusters:
            clustered_files.update(cluster.file_paths)

        # Verify all input files are present
        assert clustered_files == set(files.keys())
        assert metadata["total_files"] == len(files)

    @pytest.mark.asyncio
    async def test_cluster_ids_are_sequential(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that final cluster IDs are renumbered sequentially from 0."""
        # Create scenario that produces multiple clusters
        medium_content = self._make_content_with_tokens(25_000)
        files = {f"file{i}.py": medium_content for i in range(4)}

        clusters, _ = await clustering_service.cluster_files_hdbscan_bounded(
            files,
            min_tokens_per_cluster=15_000,
            max_tokens_per_cluster=50_000,
        )

        # Cluster IDs should be sequential starting from 0
        cluster_ids = sorted(c.cluster_id for c in clusters)
        expected_ids = list(range(len(clusters)))
        assert cluster_ids == expected_ids

    @pytest.mark.asyncio
    async def test_cluster_groups_have_correct_content(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that ClusterGroup objects contain correct file content."""
        content1 = self._make_content_with_tokens(20_000)
        content2 = self._make_content_with_tokens(20_000)
        files = {
            "file1.py": content1,
            "file2.py": content2,
        }

        clusters, _ = await clustering_service.cluster_files_hdbscan_bounded(
            files,
            min_tokens_per_cluster=15_000,
            max_tokens_per_cluster=50_000,
        )

        # Verify content in clusters matches input
        for cluster in clusters:
            assert isinstance(cluster, ClusterGroup)
            for file_path in cluster.file_paths:
                assert file_path in cluster.files_content
                assert cluster.files_content[file_path] == files[file_path]

    @pytest.mark.asyncio
    async def test_merge_respects_max_tokens(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that merge phase doesn't exceed max_tokens_per_cluster."""
        # Create 4 files: 2 small (5k each) and 2 large (45k each)
        # Use unique content so k-means can distinguish them
        # With max=50k, small files cannot merge into large ones
        files = {
            "small1.py": self._make_unique_content_with_tokens(5_000, seed="small1"),
            "small2.py": self._make_unique_content_with_tokens(5_000, seed="small2"),
            "large1.py": self._make_unique_content_with_tokens(45_000, seed="large1"),
            "large2.py": self._make_unique_content_with_tokens(45_000, seed="large2"),
        }

        clusters, metadata = await clustering_service.cluster_files_hdbscan_bounded(
            files,
            min_tokens_per_cluster=10_000,  # Small files are undersized
            max_tokens_per_cluster=50_000,
        )

        # Verify no cluster exceeds max_tokens
        for cluster in clusters:
            assert cluster.total_tokens <= 50_000, (
                f"Cluster {cluster.cluster_id} has {cluster.total_tokens:,} tokens, "
                f"exceeding max of 50,000"
            )

        # All files should still be present
        all_files = set()
        for cluster in clusters:
            all_files.update(cluster.file_paths)
        assert all_files == set(files.keys())

    @pytest.mark.asyncio
    async def test_alternative_merge_target_found(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that alternative merge target is used when nearest would exceed bounds."""
        # Create files such that smallest must find alternative target
        # 3 small files (3k each), 1 medium (30k), 1 near-max (47k)
        # Small files can merge with medium but not near-max
        # Use unique content so k-means can distinguish them
        files = {
            "tiny1.py": self._make_unique_content_with_tokens(3_000, seed="tiny1"),
            "tiny2.py": self._make_unique_content_with_tokens(3_000, seed="tiny2"),
            "tiny3.py": self._make_unique_content_with_tokens(3_000, seed="tiny3"),
            "medium.py": self._make_unique_content_with_tokens(30_000, seed="medium"),
            "nearmax.py": self._make_unique_content_with_tokens(47_000, seed="nearmax"),
        }

        clusters, metadata = await clustering_service.cluster_files_hdbscan_bounded(
            files,
            min_tokens_per_cluster=10_000,
            max_tokens_per_cluster=50_000,
        )

        # Verify no cluster exceeds max
        for cluster in clusters:
            assert cluster.total_tokens <= 50_000

        # All files present
        all_files = set()
        for cluster in clusters:
            all_files.update(cluster.file_paths)
        assert all_files == set(files.keys())

    @pytest.mark.asyncio
    async def test_unmergeable_clusters_preserved(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that unmergeable clusters are kept as-is, not discarded."""
        # Create a scenario where a small cluster cannot be merged anywhere
        # All other clusters are near max_tokens
        # Use unique content so k-means can distinguish them
        files = {
            "tiny.py": self._make_unique_content_with_tokens(2_000, seed="tiny"),
            "nearmax1.py": self._make_unique_content_with_tokens(49_000, seed="nearmax1"),
            "nearmax2.py": self._make_unique_content_with_tokens(49_000, seed="nearmax2"),
        }

        clusters, metadata = await clustering_service.cluster_files_hdbscan_bounded(
            files,
            min_tokens_per_cluster=10_000,  # tiny is undersized
            max_tokens_per_cluster=50_000,
        )

        # Tiny file should still be in output (not discarded)
        all_files = set()
        for cluster in clusters:
            all_files.update(cluster.file_paths)
        assert "tiny.py" in all_files, "Unmergeable file was discarded"

        # Should have num_unmergeable in metadata
        assert "num_unmergeable" in metadata
        # The tiny cluster cannot merge anywhere (would exceed 50k)
        assert metadata["num_unmergeable"] >= 1
