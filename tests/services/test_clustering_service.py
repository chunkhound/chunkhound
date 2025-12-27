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
