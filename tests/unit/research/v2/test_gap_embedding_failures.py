"""Edge case tests for gap query embedding failures.

Tests verify behavior when the embedding provider fails during Phase 2 Step 2.4a,
which embeds gap queries for clustering. Tests cover complete failures, partial
failures, and dimension mismatches.
"""

import numpy as np
import pytest

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.v2.gap_detection import GapDetectionService
from chunkhound.services.research.v2.models import GapCandidate
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider


class FailureEmbeddingProvider(FakeEmbeddingProvider):
    """Embedding provider that can simulate various failure modes."""

    def __init__(
        self,
        model: str = "fake-embeddings",
        dims: int = 1536,
        fail_mode: str = "none",
        partial_fail_index: int = -1,
        wrong_dims: int | None = None,
    ):
        """Initialize failure embedding provider.

        Args:
            model: Model name
            dims: Expected embedding dimensions
            fail_mode: Failure mode: 'none', 'empty', 'timeout', 'error'
            partial_fail_index: For partial failures, index to fail after
            wrong_dims: For dimension mismatch, return this many dimensions
        """
        super().__init__(model=model, dims=dims)
        self._fail_mode = fail_mode
        self._partial_fail_index = partial_fail_index
        self._wrong_dims = wrong_dims
        self._embed_call_count = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings with configurable failures."""
        self._embed_call_count += 1

        # Mode: Complete failure - raise exception
        if self._fail_mode == "error":
            raise RuntimeError("Embedding provider unavailable")

        # Mode: Timeout - raise timeout exception
        if self._fail_mode == "timeout":
            import asyncio

            raise asyncio.TimeoutError("Embedding request timed out")

        # Mode: Return empty list
        if self._fail_mode == "empty":
            return []

        # Mode: Wrong dimensions
        if self._wrong_dims is not None:
            return [
                self._generate_vector_with_dims(text, self._wrong_dims)
                for text in texts
            ]

        # Mode: Partial failure - succeed for some, fail after index
        if self._partial_fail_index >= 0:
            if len(texts) > self._partial_fail_index:
                # Only return embeddings up to partial_fail_index
                partial_texts = texts[: self._partial_fail_index]
                return await super().embed(partial_texts)

        # Default: success
        return await super().embed(texts)

    def _generate_vector_with_dims(self, text: str, dims: int) -> list[float]:
        """Generate vector with specified dimensions."""
        vector = self._generate_deterministic_vector(text)
        if dims > len(vector):
            # Pad with zeros
            return vector + [0.0] * (dims - len(vector))
        else:
            # Truncate
            return vector[:dims]


@pytest.fixture
def fake_llm_provider():
    """Create fake LLM provider for gap detection."""
    return FakeLLMProvider(
        responses={
            "gap": (
                '{"gaps": [{"query": "How is caching implemented?", '
                '"rationale": "Missing cache layer", "confidence": 0.85}]}'
            ),
        }
    )


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
        min_cluster_size=2,
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


def create_gap_detection_service(
    llm_manager, db_services, research_config, embedding_provider
):
    """Create gap detection service with custom embedding provider."""

    class MockEmbeddingManager:
        def __init__(self, provider):
            self._provider = provider

        def get_provider(self):
            return self._provider

    embedding_manager = MockEmbeddingManager(embedding_provider)
    return GapDetectionService(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=research_config,
    )


class TestEmbeddingProviderReturnsEmpty:
    """Test behavior when embedding provider returns empty list."""

    @pytest.mark.asyncio
    async def test_empty_embeddings_from_provider(
        self, llm_manager, db_services, research_config
    ):
        """Should handle empty embedding list gracefully.

        When the embedding provider returns an empty list instead of raising
        an exception, the service should handle this gracefully and return
        an empty numpy array.
        """
        # Create provider that returns empty list
        embedding_provider = FailureEmbeddingProvider(fail_mode="empty")
        service = create_gap_detection_service(
            llm_manager, db_services, research_config, embedding_provider
        )

        gaps = [
            GapCandidate("query1", "rationale1", 0.9, 0),
            GapCandidate("query2", "rationale2", 0.8, 1),
        ]

        embeddings = await service._embed_gap_queries(gaps)

        # Should return empty numpy array (not crash)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 0

    @pytest.mark.asyncio
    async def test_empty_embeddings_affects_clustering(
        self, llm_manager, db_services, research_config
    ):
        """Should handle empty embeddings during clustering step.

        When embeddings are empty, clustering should handle gracefully
        without crashing.
        """
        embedding_provider = FailureEmbeddingProvider(fail_mode="empty")
        service = create_gap_detection_service(
            llm_manager, db_services, research_config, embedding_provider
        )

        # Empty embeddings array
        embeddings = np.array([])

        labels = service._cluster_gap_queries(embeddings)

        # Should return empty labels (not crash)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == 0


class TestEmbeddingWrongDimensions:
    """Test behavior when embeddings have unexpected dimensions."""

    @pytest.mark.asyncio
    async def test_embeddings_with_fewer_dimensions(
        self, llm_manager, db_services, research_config
    ):
        """Should detect when embeddings have fewer dimensions than expected.

        When the provider returns embeddings with wrong dimensions (e.g., 768
        instead of 1536), we verify that the service receives these vectors.
        Downstream clustering may fail or produce unexpected results.
        """
        # Create provider that returns 768-dim vectors instead of 1536
        embedding_provider = FailureEmbeddingProvider(dims=1536, wrong_dims=768)
        service = create_gap_detection_service(
            llm_manager, db_services, research_config, embedding_provider
        )

        gaps = [
            GapCandidate("query1", "rationale1", 0.9, 0),
            GapCandidate("query2", "rationale2", 0.8, 1),
        ]

        embeddings = await service._embed_gap_queries(gaps)

        # Should return embeddings with wrong dimensions (768 instead of 1536)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 768  # Wrong dimensions

    @pytest.mark.asyncio
    async def test_embeddings_with_more_dimensions(
        self, llm_manager, db_services, research_config
    ):
        """Should detect when embeddings have more dimensions than expected.

        Tests the case where provider returns larger vectors than configured,
        which may indicate a model mismatch or configuration error.
        """
        # Create provider that returns 3072-dim vectors instead of 1536
        embedding_provider = FailureEmbeddingProvider(dims=1536, wrong_dims=3072)
        service = create_gap_detection_service(
            llm_manager, db_services, research_config, embedding_provider
        )

        gaps = [GapCandidate("query1", "rationale1", 0.9, 0)]

        embeddings = await service._embed_gap_queries(gaps)

        # Should return embeddings with wrong dimensions (3072 instead of 1536)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == 3072  # Wrong dimensions

    @pytest.mark.asyncio
    async def test_clustering_with_wrong_dimensions(
        self, llm_manager, db_services, research_config
    ):
        """Should verify clustering still works with mismatched dimensions.

        Clustering algorithms typically work with any consistent dimension,
        but performance and quality may degrade with dimension mismatches.
        """
        embedding_provider = FailureEmbeddingProvider(dims=1536, wrong_dims=768)
        service = create_gap_detection_service(
            llm_manager, db_services, research_config, embedding_provider
        )

        # Embeddings with wrong dimensions
        embeddings = np.random.rand(5, 768)  # 768 instead of 1536

        # Clustering should still work (doesn't validate dimensions)
        labels = service._cluster_gap_queries(embeddings)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == 5


class TestPartialBatchEmbeddingFailure:
    """Test behavior when some embeddings succeed and others fail."""

    @pytest.mark.asyncio
    async def test_partial_batch_returns_fewer_embeddings(
        self, llm_manager, db_services, research_config
    ):
        """Should handle when provider returns fewer embeddings than requested.

        When embedding provider partially fails (e.g., due to rate limiting
        or batch size constraints), it may return fewer embeddings than
        the number of input texts.
        """
        # Fail after 2 embeddings
        embedding_provider = FailureEmbeddingProvider(partial_fail_index=2)
        service = create_gap_detection_service(
            llm_manager, db_services, research_config, embedding_provider
        )

        gaps = [
            GapCandidate("query1", "rationale1", 0.9, 0),
            GapCandidate("query2", "rationale2", 0.8, 1),
            GapCandidate("query3", "rationale3", 0.7, 2),
            GapCandidate("query4", "rationale4", 0.6, 3),
        ]

        embeddings = await service._embed_gap_queries(gaps)

        # Should only return 2 embeddings (not 4)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2  # Only 2 succeeded
        assert embeddings.shape[1] == 1536

    @pytest.mark.asyncio
    async def test_clustering_with_partial_embeddings(
        self, llm_manager, db_services, research_config
    ):
        """Should handle clustering when embeddings don't match gap count.

        When the number of embeddings doesn't match the number of gaps,
        clustering will work on the available embeddings but gap-to-cluster
        mapping will be incorrect. This test documents the current behavior.
        """
        embedding_provider = FailureEmbeddingProvider(partial_fail_index=3)
        service = create_gap_detection_service(
            llm_manager, db_services, research_config, embedding_provider
        )

        # 5 gaps, but only 3 embeddings will be generated
        gaps = [GapCandidate(f"query{i}", "rationale", 0.9, i) for i in range(5)]

        embeddings = await service._embed_gap_queries(gaps)

        # Partial embeddings (only 3)
        assert embeddings.shape[0] == 3

        # Clustering will work on the 3 embeddings
        labels = service._cluster_gap_queries(embeddings)
        assert len(labels) == 3  # Matches embeddings, not gaps


class TestEmbeddingTimeoutHandling:
    """Test behavior when embedding call times out."""

    @pytest.mark.asyncio
    async def test_timeout_raises_exception(
        self, llm_manager, db_services, research_config
    ):
        """Should propagate timeout exception to caller.

        When the embedding provider times out, the service should allow
        the timeout exception to propagate so the caller can handle it
        appropriately (e.g., retry, fallback, or fail gracefully).
        """
        import asyncio

        embedding_provider = FailureEmbeddingProvider(fail_mode="timeout")
        service = create_gap_detection_service(
            llm_manager, db_services, research_config, embedding_provider
        )

        gaps = [GapCandidate("query1", "rationale1", 0.9, 0)]

        # Should raise TimeoutError
        with pytest.raises(asyncio.TimeoutError, match="timed out"):
            await service._embed_gap_queries(gaps)

    @pytest.mark.asyncio
    async def test_provider_error_raises_exception(
        self, llm_manager, db_services, research_config
    ):
        """Should propagate provider errors to caller.

        When the embedding provider encounters an error (e.g., API failure,
        network error), the service should propagate the exception rather
        than silently failing or returning invalid data.
        """
        embedding_provider = FailureEmbeddingProvider(fail_mode="error")
        service = create_gap_detection_service(
            llm_manager, db_services, research_config, embedding_provider
        )

        gaps = [GapCandidate("query1", "rationale1", 0.9, 0)]

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="unavailable"):
            await service._embed_gap_queries(gaps)


class TestIntegrationWithClusteringPipeline:
    """Integration tests for embedding failures in full clustering pipeline."""

    @pytest.mark.asyncio
    async def test_empty_embeddings_skips_clustering(
        self, llm_manager, db_services, research_config
    ):
        """Should handle empty embeddings in full clustering pipeline.

        Verifies that when embeddings are empty, the clustering step
        handles it gracefully without crashing the entire gap detection
        pipeline.
        """
        embedding_provider = FailureEmbeddingProvider(fail_mode="empty")
        service = create_gap_detection_service(
            llm_manager, db_services, research_config, embedding_provider
        )

        gaps = [GapCandidate("query1", "rationale1", 0.9, 0)]

        # Embed gaps (returns empty)
        embeddings = await service._embed_gap_queries(gaps)
        assert len(embeddings) == 0

        # Cluster gaps (should handle empty gracefully)
        labels = service._cluster_gap_queries(embeddings)
        assert len(labels) == 0

    @pytest.mark.asyncio
    async def test_zero_length_embeddings_from_empty_queries(
        self, llm_manager, db_services, research_config
    ):
        """Should handle edge case of empty gap list.

        Tests the documented behavior where empty gap list returns
        empty embeddings array.
        """
        embedding_provider = FakeEmbeddingProvider()
        service = create_gap_detection_service(
            llm_manager, db_services, research_config, embedding_provider
        )

        # Empty gaps list
        gaps = []

        embeddings = await service._embed_gap_queries(gaps)

        # Should return empty array (documented in line 390-391)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 0


class TestEmbeddingProviderCallPatterns:
    """Test embedding provider call patterns and batch behavior."""

    @pytest.mark.asyncio
    async def test_single_call_for_all_gaps(
        self, llm_manager, db_services, research_config
    ):
        """Should make single embedding call for all gap queries.

        Verifies that _embed_gap_queries batches all queries into a single
        provider call for efficiency (lines 393-397).
        """
        embedding_provider = FailureEmbeddingProvider()
        service = create_gap_detection_service(
            llm_manager, db_services, research_config, embedding_provider
        )

        gaps = [GapCandidate(f"query{i}", "rationale", 0.9, i) for i in range(5)]

        await service._embed_gap_queries(gaps)

        # Should make exactly 1 embedding call for all gaps
        assert embedding_provider._embed_call_count == 1

    @pytest.mark.asyncio
    async def test_preserves_query_order_in_embeddings(
        self, llm_manager, db_services, research_config
    ):
        """Should return embeddings in same order as gap queries.

        Verifies that embedding order matches gap query order, which is
        critical for correct gap-to-cluster mapping.
        """
        embedding_provider = FakeEmbeddingProvider()
        service = create_gap_detection_service(
            llm_manager, db_services, research_config, embedding_provider
        )

        gaps = [
            GapCandidate("alpha query", "rationale", 0.9, 0),
            GapCandidate("beta query", "rationale", 0.8, 1),
            GapCandidate("gamma query", "rationale", 0.7, 2),
        ]

        embeddings1 = await service._embed_gap_queries(gaps)

        # Reverse order
        gaps_reversed = list(reversed(gaps))
        embeddings2 = await service._embed_gap_queries(gaps_reversed)

        # Embeddings should be different (order matters)
        # First embedding for "alpha" should differ from first embedding for "gamma"
        assert not np.array_equal(embeddings1[0], embeddings2[0])
        # But first embedding of original should match last of reversed
        np.testing.assert_array_almost_equal(embeddings1[0], embeddings2[2])
