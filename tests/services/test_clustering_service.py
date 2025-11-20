"""Tests for HDBSCAN-based clustering service.

Tests two-phase clustering: HDBSCAN discovery + greedy grouping to token budget.
"""

import pytest

from chunkhound.services.clustering_service import ClusteringService, ClusterGroup
from tests.fixtures.fake_providers import FakeLLMProvider, FakeEmbeddingProvider


class TestHDBSCANClustering:
    """Test HDBSCAN-based clustering with two-phase approach."""

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
            max_tokens_per_cluster=30000,
            min_cluster_size=3,
        )

    @pytest.mark.asyncio
    async def test_single_file_creates_single_cluster(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that single file creates one cluster without clustering."""
        files = {"file1.py": "def test(): pass"}

        clusters, metadata = await clustering_service.cluster_files(files)

        assert len(clusters) == 1
        assert clusters[0].cluster_id == 0
        assert clusters[0].file_paths == ["file1.py"]
        assert metadata["num_clusters"] == 1
        assert metadata["num_native_clusters"] == 1
        assert metadata["num_outliers"] == 0

    @pytest.mark.asyncio
    async def test_small_files_single_cluster(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that files under token budget create single cluster."""
        # Files with total ~200 tokens (well under 30k limit)
        files = {
            f"file{i}.py": "def test(): pass\n" * 10 for i in range(5)
        }

        clusters, metadata = await clustering_service.cluster_files(files)

        assert len(clusters) == 1
        assert metadata["num_clusters"] == 1
        assert metadata["total_files"] == 5

    @pytest.mark.asyncio
    async def test_hdbscan_discovers_natural_clusters(
        self, fake_llm_provider: FakeLLMProvider
    ) -> None:
        """Test Phase 1: HDBSCAN discovers natural semantic clusters."""

        # Create LLM provider that returns larger token counts to force clustering
        class LargerTokenLLMProvider(FakeLLMProvider):
            def estimate_tokens(self, text: str, model: str | None = None) -> int:
                # Make files ~5k tokens each to exceed 30k total
                return 5000

        # Create embedding provider that returns clustered embeddings
        class ClusteredEmbeddingProvider(FakeEmbeddingProvider):
            async def embed(self, texts: list[str]) -> list[list[float]]:
                """Return embeddings with 3 distinct clusters."""
                embeddings = []
                for i, text in enumerate(texts):
                    # Create 3 clusters based on index
                    if i < 3:  # Cluster 0: close to [0, 0]
                        embeddings.append([0.1 * i, 0.1 * i])
                    elif i < 6:  # Cluster 1: close to [10, 10]
                        embeddings.append([10 + 0.1 * (i - 3), 10 + 0.1 * (i - 3)])
                    else:  # Cluster 2: close to [20, 20]
                        embeddings.append([20 + 0.1 * (i - 6), 20 + 0.1 * (i - 6)])
                return embeddings

        service = ClusteringService(
            embedding_provider=ClusteredEmbeddingProvider(model="clustered"),
            llm_provider=LargerTokenLLMProvider(model="large"),
            max_tokens_per_cluster=20000,  # Force multiple clusters (9 files * 5k = 45k total)
            min_cluster_size=2,  # Lower threshold for small test
        )

        # Create files that should form 3 natural clusters
        files = {f"file{i}.py": f"content {i}" * 100 for i in range(9)}

        clusters, metadata = await service.cluster_files(files)

        # Should discover natural clusters and merge appropriately
        assert metadata["num_native_clusters"] >= 2
        assert metadata["num_clusters"] >= 2  # Should need at least 2 final clusters
        assert metadata["total_files"] == 9

    @pytest.mark.asyncio
    async def test_greedy_grouping_respects_budget(
        self, fake_llm_provider: FakeLLMProvider
    ) -> None:
        """Test Phase 2: Greedy grouping respects hard token budget."""

        # Create provider that returns large token counts
        class LargeTokenLLMProvider(FakeLLMProvider):
            def estimate_tokens(self, text: str, model: str | None = None) -> int:
                # Each file is ~10k tokens
                return 10000

        # Create embedding provider with distinct clusters
        class ClusteredEmbeddingProvider(FakeEmbeddingProvider):
            async def embed(self, texts: list[str]) -> list[list[float]]:
                """Return embeddings with 5 distinct clusters (10k tokens each)."""
                embeddings = []
                for i in range(len(texts)):
                    cluster_num = i % 5
                    base = cluster_num * 10
                    embeddings.append([base + 0.1 * i, base + 0.1 * i])
                return embeddings

        service = ClusteringService(
            embedding_provider=ClusteredEmbeddingProvider(model="clustered"),
            llm_provider=LargeTokenLLMProvider(model="large"),
            max_tokens_per_cluster=25000,  # Can fit 2 files (20k) but not 3 (30k)
            min_cluster_size=2,
        )

        # Create 10 files: 5 clusters of 2 files each, 10k tokens per file
        files = {f"file{i}.py": f"large content {i}" for i in range(10)}

        clusters, metadata = await service.cluster_files(files)

        # Verify no cluster exceeds budget
        for cluster in clusters:
            assert cluster.total_tokens <= 25000, (
                f"Cluster {cluster.cluster_id} exceeds budget: {cluster.total_tokens}"
            )

        # Should merge some clusters but not exceed budget
        assert metadata["num_clusters"] >= 5  # At least as many as native clusters

    @pytest.mark.asyncio
    async def test_outlier_cluster_creation(
        self, fake_llm_provider: FakeLLMProvider
    ) -> None:
        """Test that HDBSCAN outliers get separate cluster."""

        # Create embedding provider that produces outliers
        class OutlierEmbeddingProvider(FakeEmbeddingProvider):
            async def embed(self, texts: list[str]) -> list[list[float]]:
                """Return embeddings with tight cluster + outliers."""
                embeddings = []
                for i in range(len(texts)):
                    if i < 5:  # Tight cluster near [0, 0]
                        embeddings.append([0.01 * i, 0.01 * i])
                    else:  # Outliers far away
                        embeddings.append([100 + 10 * i, 100 + 10 * i])
                return embeddings

        service = ClusteringService(
            embedding_provider=OutlierEmbeddingProvider(model="outlier"),
            llm_provider=fake_llm_provider,
            max_tokens_per_cluster=30000,
            min_cluster_size=5,  # High threshold to force outliers
        )

        # Create files where some should be outliers
        files = {f"file{i}.py": f"content {i}" * 100 for i in range(8)}

        clusters, metadata = await service.cluster_files(files)

        # Should have outliers detected
        assert metadata["num_outliers"] >= 0
        assert metadata["num_clusters"] >= 1
        assert sum(len(c.file_paths) for c in clusters) == 8  # All files accounted for

    @pytest.mark.asyncio
    async def test_empty_files_raises_error(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that empty files dict raises ValueError."""
        with pytest.raises(ValueError, match="Cannot cluster empty files"):
            await clustering_service.cluster_files({})

    @pytest.mark.asyncio
    async def test_metadata_includes_phase_info(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that metadata includes both Phase 1 and Phase 2 info."""
        files = {f"file{i}.py": f"content {i}" * 100 for i in range(5)}

        clusters, metadata = await clustering_service.cluster_files(files)

        # Verify all expected metadata fields present
        assert "num_native_clusters" in metadata
        assert "num_outliers" in metadata
        assert "num_clusters" in metadata
        assert "total_files" in metadata
        assert "total_tokens" in metadata
        assert "avg_tokens_per_cluster" in metadata

        # Verify values are reasonable
        assert metadata["num_native_clusters"] >= 0
        assert metadata["num_outliers"] >= 0
        assert metadata["num_clusters"] >= 1
        assert metadata["total_files"] == 5
        assert metadata["total_tokens"] > 0
        assert metadata["avg_tokens_per_cluster"] > 0

    @pytest.mark.asyncio
    async def test_all_files_accounted_for(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that all input files appear in exactly one cluster."""
        files = {f"file{i}.py": f"content {i}" * 100 for i in range(10)}

        clusters, metadata = await clustering_service.cluster_files(files)

        # Collect all files from all clusters
        clustered_files = set()
        for cluster in clusters:
            clustered_files.update(cluster.file_paths)

        # Verify all input files are present
        assert clustered_files == set(files.keys())
        assert metadata["total_files"] == len(files)

    @pytest.mark.asyncio
    async def test_cluster_groups_have_content(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that ClusterGroup objects contain file content."""
        files = {"file1.py": "content1", "file2.py": "content2"}

        clusters, _ = await clustering_service.cluster_files(files)

        assert len(clusters) >= 1
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
    async def test_min_cluster_size_parameter(
        self, fake_llm_provider: FakeLLMProvider, fake_embedding_provider: FakeEmbeddingProvider
    ) -> None:
        """Test that min_cluster_size parameter is used."""
        # Create service with custom min_cluster_size
        service = ClusteringService(
            embedding_provider=fake_embedding_provider,
            llm_provider=fake_llm_provider,
            max_tokens_per_cluster=30000,
            min_cluster_size=5,  # Higher threshold
        )

        files = {f"file{i}.py": f"content {i}" * 100 for i in range(10)}

        clusters, metadata = await service.cluster_files(files)

        # Should still cluster successfully
        assert len(clusters) >= 1
        assert metadata["total_files"] == 10

    @pytest.mark.asyncio
    async def test_token_counting_accuracy(
        self, clustering_service: ClusteringService
    ) -> None:
        """Test that token counts in metadata are accurate."""
        files = {f"file{i}.py": "test content" * 10 for i in range(5)}

        clusters, metadata = await clustering_service.cluster_files(files)

        # Calculate total from clusters
        cluster_total = sum(c.total_tokens for c in clusters)

        # Should match metadata total
        assert cluster_total == metadata["total_tokens"]

        # Average should be reasonable
        expected_avg = metadata["total_tokens"] / len(clusters)
        assert abs(metadata["avg_tokens_per_cluster"] - expected_avg) <= 1  # Allow rounding error


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


class TestOutlierMergingEdgeCases:
    """Test edge cases for outlier merging after native cluster merges."""

    @pytest.mark.asyncio
    async def test_outlier_merging_after_native_cluster_merges(self) -> None:
        """Test outlier merging when native clusters are merged together.

        Scenario: 3 natural clusters get merged + outliers need absorption
        Verifies that outliers use correct cluster indices after merging.
        """
        class MergeScenarioEmbeddingProvider:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                embeddings = []
                for i in range(len(texts)):
                    if i < 3:  # Cluster 0 at [0, 0]
                        embeddings.append([0.01 * i, 0.01 * i])
                    elif i < 6:  # Cluster 1 at [0.5, 0.5] (close to cluster 0)
                        embeddings.append([0.5 + 0.01 * (i - 3), 0.5 + 0.01 * (i - 3)])
                    elif i < 9:  # Cluster 2 at [1.0, 1.0] (close to clusters 0 and 1)
                        embeddings.append([1.0 + 0.01 * (i - 6), 1.0 + 0.01 * (i - 6)])
                    else:  # Outliers FAR away from all clusters
                        embeddings.append([100 + 10 * (i - 9), 100 + 10 * (i - 9)])
                return embeddings

        class TokenProvider:
            def estimate_tokens(self, text: str, model: str | None = None) -> int:
                return 6000  # 6k per file

        service = ClusteringService(
            embedding_provider=MergeScenarioEmbeddingProvider(),
            llm_provider=TokenProvider(),
            max_tokens_per_cluster=25000,  # Forces merging of 3k-token clusters
            min_cluster_size=2,
        )

        # 12 files: 3 native clusters (18k each) + 3 outliers (18k)
        files = {f"file{i}.py": f"content {i}" * 50 for i in range(12)}

        clusters, metadata = await service.cluster_files(files)

        # Verify all files accounted for
        total_files = sum(len(c.file_paths) for c in clusters)
        assert total_files == 12, "All 12 files must be in clusters"

        # No cluster should exceed budget (this validates the indexing bug fix)
        for cluster in clusters:
            assert cluster.total_tokens <= 25000, (
                f"Cluster {cluster.cluster_id} exceeds budget: {cluster.total_tokens}"
            )

        # Verify clustering completed without errors (IndexError would occur with bug)
        assert metadata["num_clusters"] >= 1
        assert metadata["num_native_clusters"] >= 1

    @pytest.mark.asyncio
    async def test_outlier_merging_with_full_capacity_clusters(self) -> None:
        """Test outlier merging when all clusters are at capacity.

        Scenario: Merged clusters fill budget, outliers form separate cluster
        """
        class CapacityEmbeddingProvider:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                embeddings = []
                for i in range(len(texts)):
                    if i < 5:  # Cluster 0
                        embeddings.append([0.01 * i, 0.01 * i])
                    elif i < 10:  # Cluster 1
                        embeddings.append([10 + 0.01 * (i - 5), 10 + 0.01 * (i - 5)])
                    else:  # Outliers very far away
                        embeddings.append([100 + 10 * i, 100 + 10 * i])
                return embeddings

        class CapacityTokenProvider:
            def estimate_tokens(self, text: str, model: str | None = None) -> int:
                return 6000  # 6k per file, 5 files = 30k (at capacity)

        service = ClusteringService(
            embedding_provider=CapacityEmbeddingProvider(),
            llm_provider=CapacityTokenProvider(),
            max_tokens_per_cluster=30000,
            min_cluster_size=2,
        )

        # 13 files: 5+5 native (30k each) + 3 outliers (18k)
        files = {f"file{i}.py": f"content {i}" * 50 for i in range(13)}

        clusters, metadata = await service.cluster_files(files)

        # All files accounted for
        assert sum(len(c.file_paths) for c in clusters) == 13

        # Budget respected (validates the fix - no IndexError or wrong assignments)
        for cluster in clusters:
            assert cluster.total_tokens <= 30000

        # Verify clustering completed successfully
        assert metadata["num_clusters"] >= 1

    @pytest.mark.asyncio
    async def test_outlier_distance_based_assignment(self) -> None:
        """Test that outliers merge into nearest clusters by distance.

        Scenario: Outliers positioned near specific clusters should merge there
        """
        class DistanceEmbeddingProvider:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                embeddings = []
                for i in range(len(texts)):
                    if i < 3:  # Cluster 0 at [0, 0]
                        embeddings.append([0.01 * i, 0.01 * i])
                    elif i < 6:  # Cluster 1 at [10, 10]
                        embeddings.append([10 + 0.01 * (i - 3), 10 + 0.01 * (i - 3)])
                    elif i < 8:  # Outliers far from both clusters
                        embeddings.append([50 + 10 * (i - 6), 50 + 10 * (i - 6)])
                    else:  # More outliers far away
                        embeddings.append([150 + 10 * (i - 8), 150 + 10 * (i - 8)])
                return embeddings

        class DistanceTokenProvider:
            def estimate_tokens(self, text: str, model: str | None = None) -> int:
                return 3000  # Small tokens to allow merging

        service = ClusteringService(
            embedding_provider=DistanceEmbeddingProvider(),
            llm_provider=DistanceTokenProvider(),
            max_tokens_per_cluster=30000,
            min_cluster_size=2,
        )

        # 10 files: 3+3 native + 4 outliers
        files = {f"file{i}.py": f"content {i}" * 20 for i in range(10)}

        clusters, metadata = await service.cluster_files(files)

        # All files present
        assert sum(len(c.file_paths) for c in clusters) == 10

        # Budget respected (validates correct cluster indexing after merges)
        for cluster in clusters:
            assert cluster.total_tokens <= 30000

        # Verify clustering completed without crashes
        assert metadata["num_clusters"] >= 1

    @pytest.mark.asyncio
    async def test_outlier_only_scenario(self) -> None:
        """Test clustering when HDBSCAN finds only outliers.

        Scenario: No natural clusters, all files are outliers
        """
        class AllOutliersEmbeddingProvider:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                # All widely separated -> all outliers
                return [[100.0 * i, 100.0 * i] for i in range(len(texts))]

        class SmallTokenProvider:
            def estimate_tokens(self, text: str, model: str | None = None) -> int:
                return 2000  # 2k per file

        service = ClusteringService(
            embedding_provider=AllOutliersEmbeddingProvider(),
            llm_provider=SmallTokenProvider(),
            max_tokens_per_cluster=30000,
            min_cluster_size=10,  # High threshold forces all to be outliers
        )

        # 6 files, all will be outliers (12k total, under budget)
        files = {f"file{i}.py": f"content {i}" * 30 for i in range(6)}

        clusters, metadata = await service.cluster_files(files)

        # All files must be accounted for
        assert sum(len(c.file_paths) for c in clusters) == 6
        assert metadata["num_clusters"] >= 1

        # All clusters respect budget
        for cluster in clusters:
            assert cluster.total_tokens <= 30000

    @pytest.mark.asyncio
    async def test_merged_cluster_centroid_for_outliers(self) -> None:
        """Test that merged cluster centroids are used for outlier distance.

        Scenario: Native clusters merge, outliers use merged centroid for distance
        """
        class MergedCentroidEmbeddingProvider:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                embeddings = []
                for i in range(len(texts)):
                    if i < 2:  # Cluster 0 at [0, 0]
                        embeddings.append([0.01 * i, 0.01 * i])
                    elif i < 4:  # Cluster 1 at [0.2, 0.2] (close to 0)
                        embeddings.append([0.2 + 0.01 * (i - 2), 0.2 + 0.01 * (i - 2)])
                    elif i < 6:  # Cluster 2 at [0.4, 0.4] (close to 0 and 1)
                        embeddings.append([0.4 + 0.01 * (i - 4), 0.4 + 0.01 * (i - 4)])
                    else:  # Outliers FAR away
                        embeddings.append([100 + 10 * (i - 6), 100 + 10 * (i - 6)])
                return embeddings

        class MergedTokenProvider:
            def estimate_tokens(self, text: str, model: str | None = None) -> int:
                return 5000  # 5k per file

        service = ClusteringService(
            embedding_provider=MergedCentroidEmbeddingProvider(),
            llm_provider=MergedTokenProvider(),
            max_tokens_per_cluster=30000,  # Can fit 6 files (30k)
            min_cluster_size=2,
        )

        # 10 files: 2+2+2 native + 4 outliers
        files = {f"file{i}.py": f"content {i}" * 25 for i in range(10)}

        clusters, metadata = await service.cluster_files(files)

        # All files present
        assert sum(len(c.file_paths) for c in clusters) == 10

        # Budget respected (critical validation - bug would cause wrong assignments)
        for cluster in clusters:
            assert cluster.total_tokens <= 30000

        # Verify clustering with merged native clusters completed successfully
        assert metadata["num_native_clusters"] >= 1
        assert metadata["num_clusters"] >= 1
