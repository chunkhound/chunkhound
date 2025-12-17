"""Edge case tests for shard budget boundary conditions in gap detection.

Tests exact boundary conditions for token-based sharding in
GapDetectionService._shard_by_tokens.
Default shard_budget is 40,000 tokens, and token estimation uses 4 chars
per token.
"""

import pytest

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.v2.gap_detection import GapDetectionService
from tests.fixtures.fake_providers import FakeLLMProvider


@pytest.fixture
def fake_llm_provider():
    """Create fake LLM provider with deterministic token estimation."""
    return FakeLLMProvider(model="fake-gpt")


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
def fake_embedding_provider():
    """Create fake embedding provider."""
    from tests.fixtures.fake_providers import FakeEmbeddingProvider

    return FakeEmbeddingProvider(dims=1536)


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
    """Create research configuration with default shard_budget (40,000 tokens)."""
    return ResearchConfig(
        shard_budget=40_000,  # Default budget for testing
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


@pytest.fixture
def gap_detection_service(llm_manager, embedding_manager, db_services, research_config):
    """Create gap detection service with mocked dependencies."""
    return GapDetectionService(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=research_config,
    )


class TestShardBudgetBoundaries:
    """Test exact boundary conditions for shard_budget (default: 40,000 tokens).

    Token estimation: 4 chars per token (from FakeLLMProvider.estimate_tokens)
    Budget: 40,000 tokens = 160,000 chars

    Boundary conditions to test:
    1. Cluster exactly at budget (40,000 tokens)
    2. Cluster one token over budget (40,001 tokens)
    3. Single chunk exceeds budget (can't be split)
    """

    def test_cluster_exactly_at_budget(self, gap_detection_service):
        """Cluster with exactly 40,000 tokens should fit in single shard.

        This verifies that the boundary condition check
        (current_tokens + chunk_tokens > budget)
        correctly includes clusters that exactly match the budget.

        Expected behavior: 40,000 tokens <= 40,000 budget → single shard
        """
        # Create chunks totaling exactly 40,000 tokens (160,000 chars)
        # Split into multiple chunks to verify aggregation works
        chunk1 = {"chunk_id": "c1", "content": "x" * 80_000}  # 20,000 tokens
        chunk2 = {"chunk_id": "c2", "content": "x" * 80_000}  # 20,000 tokens
        cluster_groups = [[chunk1, chunk2]]

        shards = gap_detection_service._shard_by_tokens(cluster_groups)

        # Should create single shard since total equals budget
        assert len(shards) == 1, "Expected 1 shard for cluster exactly at budget"
        assert len(shards[0]) == 2, "Expected both chunks in single shard"

        # Verify chunk preservation
        chunk_ids = {c["chunk_id"] for c in shards[0]}
        assert chunk_ids == {"c1", "c2"}

    def test_cluster_one_token_over_budget(self, gap_detection_service):
        """Cluster with 40,001 tokens should split into two shards.

        This verifies the boundary check correctly splits when budget is
        exceeded by even 1 token.

        Expected behavior: First chunk triggers split when second chunk
        would exceed budget.
        - Chunk 1: 20,000 tokens → fits in shard 1
        - Chunk 2: 20,001 tokens → exceeds budget (40,001 > 40,000)
          → new shard
        """
        # Create chunks totaling 40,001 tokens (160,004 chars)
        chunk1 = {"chunk_id": "c1", "content": "x" * 80_000}  # 20,000 tokens
        chunk2 = {"chunk_id": "c2", "content": "x" * 80_004}  # 20,001 tokens
        cluster_groups = [[chunk1, chunk2]]

        shards = gap_detection_service._shard_by_tokens(cluster_groups)

        # Should split into 2 shards since total exceeds budget
        assert len(shards) == 2, (
            "Expected 2 shards for cluster exceeding budget by 1 token"
        )
        assert len(shards[0]) == 1, "Expected first shard to have 1 chunk"
        assert len(shards[1]) == 1, "Expected second shard to have 1 chunk"

        # Verify chunk distribution
        assert shards[0][0]["chunk_id"] == "c1"
        assert shards[1][0]["chunk_id"] == "c2"

    def test_single_chunk_exceeds_budget(self, gap_detection_service):
        """Single chunk > 40,000 tokens should get its own shard.

        This verifies oversized chunks aren't dropped and get their own
        shard. The algorithm can't split individual chunks, so oversized
        ones must be accommodated.

        Expected behavior: Oversized chunk gets its own shard regardless
        of budget.
        """
        # Create single chunk exceeding budget: 50,000 tokens (200,000 chars)
        oversized_chunk = {
            "chunk_id": "oversized",
            "content": "x" * 200_000,
            "file_path": "large_file.py",
        }
        cluster_groups = [[oversized_chunk]]

        shards = gap_detection_service._shard_by_tokens(cluster_groups)

        # Should create single shard with oversized chunk
        assert len(shards) == 1, "Expected 1 shard for oversized chunk"
        assert len(shards[0]) == 1, "Expected oversized chunk in its own shard"
        assert shards[0][0]["chunk_id"] == "oversized"

        # Verify chunk data preserved
        assert shards[0][0]["file_path"] == "large_file.py"

    def test_multiple_oversized_chunks_separate_shards(self, gap_detection_service):
        """Multiple oversized chunks should each get their own shard.

        Verifies that when multiple chunks exceed the budget, each is placed
        in a separate shard since they can't be combined.

        Expected behavior: 3 oversized chunks → 3 separate shards
        """
        # Create 3 oversized chunks (each 50,000 tokens = 200,000 chars)
        oversized1 = {"chunk_id": "over1", "content": "x" * 200_000}
        oversized2 = {"chunk_id": "over2", "content": "y" * 200_000}
        oversized3 = {"chunk_id": "over3", "content": "z" * 200_000}
        cluster_groups = [[oversized1, oversized2, oversized3]]

        shards = gap_detection_service._shard_by_tokens(cluster_groups)

        # Each oversized chunk should get its own shard
        assert len(shards) == 3, "Expected 3 shards for 3 oversized chunks"
        assert all(len(shard) == 1 for shard in shards), (
            "Each shard should have 1 chunk"
        )

        # Verify chunk IDs preserved in order
        chunk_ids = [shard[0]["chunk_id"] for shard in shards]
        assert chunk_ids == ["over1", "over2", "over3"]

    def test_boundary_with_small_chunks_accumulate(self, gap_detection_service):
        """Many small chunks should accumulate until budget exceeded.

        Verifies that the algorithm correctly accumulates small chunks
        until the budget is exceeded, then starts a new shard.

        Expected behavior: Small chunks accumulate until total would
        exceed 40,000 tokens.
        """
        # Create 11 chunks of 4,000 tokens each (16,000 chars)
        # First 10 chunks = 40,000 tokens (fits exactly)
        # 11th chunk would exceed budget
        chunks = [{"chunk_id": f"c{i}", "content": "x" * 16_000} for i in range(11)]
        cluster_groups = [chunks]

        shards = gap_detection_service._shard_by_tokens(cluster_groups)

        # Should create 2 shards: 10 chunks in first, 1 in second
        assert len(shards) == 2, "Expected 2 shards for 11 x 4,000-token chunks"
        assert len(shards[0]) == 10, "Expected first shard to have 10 chunks"
        assert len(shards[1]) == 1, "Expected second shard to have 1 chunk"

        # Verify all chunks preserved
        all_chunk_ids = [c["chunk_id"] for shard in shards for c in shard]
        expected_ids = [f"c{i}" for i in range(11)]
        assert all_chunk_ids == expected_ids, "All chunks should be preserved in order"

    def test_all_chunks_preserved_after_sharding(self, gap_detection_service):
        """Verify all chunks are preserved through sharding regardless of splits.

        This is a critical invariant: sharding must never drop chunks.

        Expected behavior: All input chunks appear in output shards.
        """
        # Create varied chunk sizes to trigger multiple shards
        chunks = [
            {"chunk_id": "c1", "content": "x" * 80_000},  # 20,000 tokens
            {"chunk_id": "c2", "content": "x" * 80_000},  # 20,000 tokens
            # 1,000 tokens (would exceed first shard)
            {"chunk_id": "c3", "content": "x" * 4_000},
            {"chunk_id": "c4", "content": "x" * 40_000},  # 10,000 tokens
            {"chunk_id": "c5", "content": "x" * 200_000},  # 50,000 tokens
        ]
        cluster_groups = [chunks]

        shards = gap_detection_service._shard_by_tokens(cluster_groups)

        # Collect all chunks from all shards
        all_chunks = [chunk for shard in shards for chunk in shard]
        all_chunk_ids = [c["chunk_id"] for c in all_chunks]

        # Verify no chunks lost
        expected_ids = ["c1", "c2", "c3", "c4", "c5"]
        assert all_chunk_ids == expected_ids, "All chunks must be preserved"
        assert len(all_chunks) == 5, "Chunk count must match input"

    def test_exact_budget_multiple_clusters(self, gap_detection_service):
        """Multiple clusters each exactly at budget should create separate shards.

        Verifies that clusters are sharded independently - each cluster that fits
        the budget gets its own shard.

        Expected behavior: 3 clusters x 40,000 tokens each → 3 shards
        """
        # Create 3 clusters, each exactly 40,000 tokens (160,000 chars)
        cluster1 = [{"chunk_id": "c1", "content": "x" * 160_000}]  # 40,000 tokens
        cluster2 = [{"chunk_id": "c2", "content": "y" * 160_000}]  # 40,000 tokens
        cluster3 = [{"chunk_id": "c3", "content": "z" * 160_000}]  # 40,000 tokens
        cluster_groups = [cluster1, cluster2, cluster3]

        shards = gap_detection_service._shard_by_tokens(cluster_groups)

        # Each cluster should be its own shard
        assert len(shards) == 3, "Expected 3 shards for 3 budget-sized clusters"
        assert all(len(shard) == 1 for shard in shards), (
            "Each shard should have 1 chunk"
        )

        # Verify chunk IDs
        chunk_ids = [shard[0]["chunk_id"] for shard in shards]
        assert chunk_ids == ["c1", "c2", "c3"]

    def test_boundary_off_by_one_accumulation(self, gap_detection_service):
        """Test accumulation boundary with off-by-one scenarios.

        Verifies the algorithm correctly handles the boundary when
        current_tokens + chunk_tokens exactly equals or exceeds the
        budget.

        Expected behavior:
        - 39,999 + 1 = 40,000 → fits in shard (equals budget)
        - 39,999 + 2 = 40,001 → new shard (exceeds budget)
        """
        # Scenario 1: current_tokens + chunk_tokens == budget (should fit)
        chunk1 = {"chunk_id": "c1", "content": "x" * 159_996}  # 39,999 tokens
        chunk2 = {"chunk_id": "c2", "content": "x" * 4}  # 1 token
        cluster_groups = [[chunk1, chunk2]]

        shards = gap_detection_service._shard_by_tokens(cluster_groups)

        assert len(shards) == 1, "Expected 1 shard when total equals budget"
        assert len(shards[0]) == 2, "Both chunks should fit in single shard"

        # Scenario 2: current_tokens + chunk_tokens > budget (should split)
        chunk3 = {"chunk_id": "c3", "content": "x" * 159_996}  # 39,999 tokens
        chunk4 = {"chunk_id": "c4", "content": "x" * 8}  # 2 tokens
        cluster_groups2 = [[chunk3, chunk4]]

        shards2 = gap_detection_service._shard_by_tokens(cluster_groups2)

        assert len(shards2) == 2, "Expected 2 shards when total exceeds budget by 1"
        assert len(shards2[0]) == 1, "First shard should have 1 chunk"
        assert len(shards2[1]) == 1, "Second shard should have 1 chunk"
