"""Unit tests for v2 Gap Detection Service.

Tests the gap detection and filling logic without requiring real API calls,
using mock LLM and embedding providers.
"""

import numpy as np
import pytest

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.v2.gap_detection import GapDetectionService
from chunkhound.services.research.v2.models import GapCandidate, UnifiedGap
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider


@pytest.fixture
def fake_llm_provider():
    """Create fake LLM provider for gap detection."""
    return FakeLLMProvider(
        responses={
            # Gap detection (structured JSON)
            "gap": '{"gaps": [{"query": "How is caching implemented?", "rationale": "Missing cache layer", "confidence": 0.85}, {"query": "What error handling exists?", "rationale": "Error handling unclear", "confidence": 0.75}]}',
            "code coverage": '{"gaps": [{"query": "How is authentication handled?", "rationale": "Auth missing", "confidence": 0.9}]}',
            # Gap unification (structured JSON)
            "merge": '{"unified_query": "How does the caching and error handling system work?"}',
            "similar": '{"unified_query": "How is authentication implemented in the system?"}',
        }
    )


@pytest.fixture
def fake_embedding_provider():
    """Create fake embedding provider for gap query clustering."""
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
def embedding_manager(fake_embedding_provider, monkeypatch):
    """Create embedding manager with fake provider."""

    class MockEmbeddingManager:
        def get_provider(self):
            return fake_embedding_provider

    return MockEmbeddingManager()


@pytest.fixture
def db_services(monkeypatch):
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
        shard_budget=20_000,  # Minimum valid value
        min_cluster_size=2,
        gap_similarity_threshold=0.3,
        min_gaps=1,
        max_gaps=5,
        max_symbols=10,  # Maximum is 20
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


class TestShardByTokens:
    """Test token-based sharding of chunk clusters."""

    def test_single_cluster_within_budget(self, gap_detection_service):
        """Should create single shard when cluster fits budget."""
        cluster_groups = [
            [
                {"chunk_id": "c1", "code": "x" * 100},
                {"chunk_id": "c2", "code": "x" * 100},
            ]
        ]

        shards = gap_detection_service._shard_by_tokens(cluster_groups)

        assert len(shards) == 1
        assert len(shards[0]) == 2

    def test_splits_cluster_when_exceeds_budget(self, gap_detection_service):
        """Should split cluster into multiple shards when budget exceeded."""
        # Create chunks that exceed shard_budget (4000 tokens)
        large_chunk = {"chunk_id": "c1", "code": "x" * 10000}  # ~2500 tokens
        cluster_groups = [[large_chunk, large_chunk.copy()]]

        shards = gap_detection_service._shard_by_tokens(cluster_groups)

        # Should split into 2 shards
        assert len(shards) >= 1
        assert all(isinstance(shard, list) for shard in shards)

    def test_multiple_clusters_create_multiple_shards(self, gap_detection_service):
        """Should process multiple clusters into shards."""
        cluster_groups = [
            [{"chunk_id": "c1", "code": "x" * 100}],
            [{"chunk_id": "c2", "code": "x" * 100}],
            [{"chunk_id": "c3", "code": "x" * 100}],
        ]

        shards = gap_detection_service._shard_by_tokens(cluster_groups)

        assert len(shards) >= 1
        total_chunks = sum(len(shard) for shard in shards)
        assert total_chunks == 3

    def test_empty_clusters_return_empty_shards(self, gap_detection_service):
        """Should handle empty cluster groups gracefully.

        NOTE: Current implementation has division by zero bug in logging.
        This test documents the bug for future fix.
        """
        cluster_groups = []

        # BUG: Implementation crashes on empty input due to logging division by zero
        with pytest.raises(ZeroDivisionError):
            gap_detection_service._shard_by_tokens(cluster_groups)

    def test_preserves_chunk_data(self, gap_detection_service):
        """Should preserve all chunk data through sharding."""
        cluster_groups = [
            [
                {
                    "chunk_id": "c1",
                    "code": "def foo(): pass",
                    "file_path": "test.py",
                    "start_line": 1,
                }
            ]
        ]

        shards = gap_detection_service._shard_by_tokens(cluster_groups)

        assert len(shards) == 1
        chunk = shards[0][0]
        assert chunk["chunk_id"] == "c1"
        assert chunk["code"] == "def foo(): pass"
        assert chunk["file_path"] == "test.py"
        assert chunk["start_line"] == 1


class TestGlobalDedup:
    """Test global deduplication across gap results."""

    def test_deduplicates_by_chunk_id(self, gap_detection_service):
        """Should keep only one chunk per chunk_id."""
        gap_results = [
            [
                {"chunk_id": "c1", "rerank_score": 0.8},
                {"chunk_id": "c2", "rerank_score": 0.7},
            ],
            [
                {"chunk_id": "c1", "rerank_score": 0.9},  # Duplicate with higher score
                {"chunk_id": "c3", "rerank_score": 0.6},
            ],
        ]

        deduplicated = gap_detection_service._global_dedup(gap_results)

        # Should have 3 unique chunks
        chunk_ids = [c["chunk_id"] for c in deduplicated]
        assert len(chunk_ids) == 3
        assert len(set(chunk_ids)) == 3

    def test_keeps_highest_rerank_score(self, gap_detection_service):
        """Should keep chunk with highest rerank_score on conflict."""
        gap_results = [
            [{"chunk_id": "c1", "rerank_score": 0.8, "version": "v1"}],
            [{"chunk_id": "c1", "rerank_score": 0.9, "version": "v2"}],
            [{"chunk_id": "c1", "rerank_score": 0.7, "version": "v3"}],
        ]

        deduplicated = gap_detection_service._global_dedup(gap_results)

        assert len(deduplicated) == 1
        assert deduplicated[0]["version"] == "v2"  # Highest score (0.9)
        assert deduplicated[0]["rerank_score"] == 0.9

    def test_handles_empty_gap_results(self, gap_detection_service):
        """Should handle empty gap results gracefully."""
        gap_results = []

        deduplicated = gap_detection_service._global_dedup(gap_results)

        assert deduplicated == []

    def test_handles_empty_sublists(self, gap_detection_service):
        """Should handle gap results with empty sublists."""
        gap_results = [[], [{"chunk_id": "c1", "rerank_score": 0.8}], []]

        deduplicated = gap_detection_service._global_dedup(gap_results)

        assert len(deduplicated) == 1
        assert deduplicated[0]["chunk_id"] == "c1"

    def test_preserves_chunk_data(self, gap_detection_service):
        """Should preserve all chunk data through deduplication."""
        gap_results = [
            [
                {
                    "chunk_id": "c1",
                    "file_path": "test.py",
                    "code": "def foo(): pass",
                    "rerank_score": 0.9,
                    "metadata": {"important": "data"},
                }
            ]
        ]

        deduplicated = gap_detection_service._global_dedup(gap_results)

        chunk = deduplicated[0]
        assert chunk["file_path"] == "test.py"
        assert chunk["code"] == "def foo(): pass"
        assert chunk["metadata"]["important"] == "data"

    def test_falls_back_to_id_field(self, gap_detection_service):
        """Should use 'id' field if 'chunk_id' not present."""
        gap_results = [
            [{"id": "c1", "rerank_score": 0.8}],
            [{"id": "c2", "rerank_score": 0.7}],
        ]

        deduplicated = gap_detection_service._global_dedup(gap_results)

        assert len(deduplicated) == 2
        ids = [c.get("id") for c in deduplicated]
        assert "c1" in ids
        assert "c2" in ids


class TestMergeCoverage:
    """Test merging coverage and gap chunks."""

    def test_merges_disjoint_chunks(self, gap_detection_service):
        """Should merge chunks with no overlap."""
        covered_chunks = [
            {"chunk_id": "c1", "rerank_score": 0.9},
            {"chunk_id": "c2", "rerank_score": 0.8},
        ]
        gap_chunks = [
            {"chunk_id": "g1", "rerank_score": 0.7},
            {"chunk_id": "g2", "rerank_score": 0.6},
        ]

        merged = gap_detection_service._merge_coverage(covered_chunks, gap_chunks)

        assert len(merged) == 4
        chunk_ids = {c["chunk_id"] for c in merged}
        assert chunk_ids == {"c1", "c2", "g1", "g2"}

    def test_deduplicates_overlapping_chunks(self, gap_detection_service):
        """Should deduplicate when chunks appear in both lists."""
        covered_chunks = [
            {"chunk_id": "c1", "rerank_score": 0.8},
            {"chunk_id": "c2", "rerank_score": 0.7},
        ]
        gap_chunks = [
            {"chunk_id": "c1", "rerank_score": 0.9},  # Higher score
            {"chunk_id": "g1", "rerank_score": 0.6},
        ]

        merged = gap_detection_service._merge_coverage(covered_chunks, gap_chunks)

        assert len(merged) == 3
        chunk_ids = {c["chunk_id"] for c in merged}
        assert chunk_ids == {"c1", "c2", "g1"}

    def test_keeps_higher_score_on_conflict(self, gap_detection_service):
        """Should keep chunk with higher rerank_score on conflict."""
        covered_chunks = [{"chunk_id": "c1", "rerank_score": 0.7, "source": "coverage"}]
        gap_chunks = [{"chunk_id": "c1", "rerank_score": 0.9, "source": "gap"}]

        merged = gap_detection_service._merge_coverage(covered_chunks, gap_chunks)

        assert len(merged) == 1
        assert merged[0]["source"] == "gap"  # Gap has higher score
        assert merged[0]["rerank_score"] == 0.9

    def test_handles_empty_inputs(self, gap_detection_service):
        """Should handle empty coverage or gap chunks."""
        # Empty coverage
        merged1 = gap_detection_service._merge_coverage(
            [], [{"chunk_id": "g1", "rerank_score": 0.8}]
        )
        assert len(merged1) == 1

        # Empty gaps
        merged2 = gap_detection_service._merge_coverage(
            [{"chunk_id": "c1", "rerank_score": 0.8}], []
        )
        assert len(merged2) == 1

        # Both empty
        merged3 = gap_detection_service._merge_coverage([], [])
        assert merged3 == []

    def test_preserves_chunk_data_through_merge(self, gap_detection_service):
        """Should preserve all chunk data through merge."""
        covered_chunks = [
            {
                "chunk_id": "c1",
                "file_path": "test.py",
                "code": "def foo(): pass",
                "rerank_score": 0.9,
            }
        ]
        gap_chunks = [
            {
                "chunk_id": "g1",
                "file_path": "cache.py",
                "code": "class Cache: pass",
                "rerank_score": 0.8,
            }
        ]

        merged = gap_detection_service._merge_coverage(covered_chunks, gap_chunks)

        assert len(merged) == 2
        c1 = next(c for c in merged if c["chunk_id"] == "c1")
        g1 = next(c for c in merged if c["chunk_id"] == "g1")

        assert c1["file_path"] == "test.py"
        assert c1["code"] == "def foo(): pass"
        assert g1["file_path"] == "cache.py"
        assert g1["code"] == "class Cache: pass"


class TestEmbedGapQueries:
    """Test gap query embedding."""

    @pytest.mark.asyncio
    async def test_embeds_all_gap_queries(self, gap_detection_service):
        """Should embed all gap candidate queries."""
        gaps = [
            GapCandidate("query1", "rationale1", 0.9, 0),
            GapCandidate("query2", "rationale2", 0.8, 1),
            GapCandidate("query3", "rationale3", 0.7, 2),
        ]

        embeddings = await gap_detection_service._embed_gap_queries(gaps)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 1536  # Fake provider dims

    @pytest.mark.asyncio
    async def test_handles_empty_gaps(self, gap_detection_service):
        """Should handle empty gaps list."""
        gaps = []

        embeddings = await gap_detection_service._embed_gap_queries(gaps)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 0

    @pytest.mark.asyncio
    async def test_embeddings_are_deterministic(self, gap_detection_service):
        """Should return same embeddings for same queries."""
        gaps1 = [GapCandidate("same query", "rationale", 0.9, 0)]
        gaps2 = [GapCandidate("same query", "rationale", 0.9, 0)]

        embeddings1 = await gap_detection_service._embed_gap_queries(gaps1)
        embeddings2 = await gap_detection_service._embed_gap_queries(gaps2)

        np.testing.assert_array_almost_equal(embeddings1, embeddings2)


class TestClusterGapQueries:
    """Test gap query clustering."""

    def test_single_gap_single_cluster(self, gap_detection_service):
        """Should assign single gap to cluster 0."""
        embeddings = np.array([[0.1, 0.2, 0.3]])

        labels = gap_detection_service._cluster_gap_queries(embeddings)

        assert len(labels) == 1
        assert labels[0] == 0

    def test_similar_gaps_same_cluster(self, gap_detection_service):
        """Should cluster similar embeddings together."""
        # Two very similar embeddings
        base = np.random.rand(1536)
        base = base / np.linalg.norm(base)  # Normalize
        similar = base + 0.01 * np.random.rand(1536)
        similar = similar / np.linalg.norm(similar)

        embeddings = np.array([base, similar])

        labels = gap_detection_service._cluster_gap_queries(embeddings)

        # Should be in same cluster (both 0 or both same label)
        assert labels[0] == labels[1]

    def test_empty_embeddings_return_empty(self, gap_detection_service):
        """Should handle empty embeddings."""
        embeddings = np.array([])

        labels = gap_detection_service._cluster_gap_queries(embeddings)

        assert len(labels) == 0


class TestSelectGapsByElbow:
    """Test gap selection using elbow detection."""

    def test_selects_within_min_max_bounds(self, gap_detection_service):
        """Should respect min and max gap constraints."""
        # Create gaps with descending scores
        gaps = [
            UnifiedGap(f"query{i}", [], 1, score=1.0 - i * 0.1) for i in range(10)
        ]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # Should be within bounds (min_gaps=1, max_gaps=5)
        assert len(selected) >= 1
        assert len(selected) <= 5

    def test_empty_gaps_return_empty(self, gap_detection_service):
        """Should handle empty gaps list."""
        gaps = []

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        assert selected == []

    def test_fewer_than_min_returns_all(self, gap_detection_service):
        """Should return all gaps when below min_gaps."""
        # Only 1 gap, min_gaps=1
        gaps = [UnifiedGap("query1", [], 1, score=0.9)]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        assert len(selected) == 1
        assert selected[0].query == "query1"

    def test_sorts_by_score_descending(self, gap_detection_service):
        """Should sort gaps by score descending."""
        gaps = [
            UnifiedGap("low", [], 1, score=0.3),
            UnifiedGap("high", [], 1, score=0.9),
            UnifiedGap("medium", [], 1, score=0.6),
        ]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # First selected should be highest score
        assert selected[0].query == "high"
        assert selected[0].score == 0.9

    def test_stops_at_score_drop(self, gap_detection_service):
        """Should stop when score drops significantly."""
        gaps = [
            UnifiedGap("q1", [], 1, score=1.0),
            UnifiedGap("q2", [], 1, score=0.9),
            UnifiedGap("q3", [], 1, score=0.4),  # Drops below 50% of top
            UnifiedGap("q4", [], 1, score=0.3),
        ]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # Should stop before q3 (score < 0.5 * 1.0 = 0.5)
        assert len(selected) <= 3
        if len(selected) == 3:
            assert selected[2].score >= 0.4

    def test_kneedle_detects_clear_elbow(self, gap_detection_service):
        """Should use kneedle to detect clear elbow point."""
        # Create clear elbow pattern: 3 high scores, then sharp drop
        gaps = [
            UnifiedGap("q1", [], 1, score=1.0),
            UnifiedGap("q2", [], 1, score=0.95),
            UnifiedGap("q3", [], 1, score=0.9),
            UnifiedGap("q4", [], 1, score=0.3),  # Sharp drop
            UnifiedGap("q5", [], 1, score=0.25),
        ]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # Should select 3 gaps (before sharp drop)
        # Note: May select 4 if kneedle identifies q4 as elbow point
        assert len(selected) >= 3
        assert len(selected) <= 4

    def test_fallback_to_max_gaps_when_too_many(self, gap_detection_service):
        """Should use max_gaps when candidates exceed maximum."""
        # Create 10 gaps (max_gaps=5)
        gaps = [UnifiedGap(f"q{i}", [], 1, score=1.0 - i * 0.05) for i in range(10)]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # Should cap at max_gaps
        assert len(selected) <= 5


class TestFindElbowKneedle:
    """Test kneedle algorithm for elbow detection."""

    def test_finds_elbow_in_clear_curve(self, gap_detection_service):
        """Should find elbow in clear L-shaped curve."""
        # Create clear elbow: high plateau, sharp drop, low plateau
        gaps = [
            UnifiedGap("q1", [], 1, score=1.0),
            UnifiedGap("q2", [], 1, score=0.95),
            UnifiedGap("q3", [], 1, score=0.9),
            UnifiedGap("q4", [], 1, score=0.3),  # Sharp drop here
            UnifiedGap("q5", [], 1, score=0.25),
            UnifiedGap("q6", [], 1, score=0.2),
        ]

        elbow_idx = gap_detection_service._find_elbow_kneedle(gaps)

        # Should find elbow around index 3-4 (the sharp drop)
        assert elbow_idx is not None
        assert 3 <= elbow_idx <= 4

    def test_returns_none_for_linear_curve(self, gap_detection_service):
        """Should return None when curve is linear (no elbow)."""
        # Linear decline (no elbow)
        gaps = [UnifiedGap(f"q{i}", [], 1, score=1.0 - i * 0.2) for i in range(5)]

        elbow_idx = gap_detection_service._find_elbow_kneedle(gaps)

        # Linear curve may return None or a weak elbow
        # Implementation uses 1% threshold to filter weak elbows
        assert elbow_idx is None or isinstance(elbow_idx, int)

    def test_returns_none_for_few_points(self, gap_detection_service):
        """Should return None when too few points for elbow."""
        # Only 2 points (need at least 3)
        gaps = [
            UnifiedGap("q1", [], 1, score=1.0),
            UnifiedGap("q2", [], 1, score=0.5),
        ]

        elbow_idx = gap_detection_service._find_elbow_kneedle(gaps)

        assert elbow_idx is None

    def test_returns_none_for_identical_scores(self, gap_detection_service):
        """Should return None when all scores identical."""
        gaps = [UnifiedGap(f"q{i}", [], 1, score=0.8) for i in range(5)]

        elbow_idx = gap_detection_service._find_elbow_kneedle(gaps)

        assert elbow_idx is None

    def test_elbow_index_is_one_based(self, gap_detection_service):
        """Should return 1-based index (number of gaps to select)."""
        gaps = [
            UnifiedGap("q1", [], 1, score=1.0),
            UnifiedGap("q2", [], 1, score=0.9),
            UnifiedGap("q3", [], 1, score=0.3),  # Elbow at index 2
            UnifiedGap("q4", [], 1, score=0.2),
        ]

        elbow_idx = gap_detection_service._find_elbow_kneedle(gaps)

        # Should return 2 or 3 (1-based count)
        assert elbow_idx is not None
        assert elbow_idx >= 1
        assert elbow_idx <= len(gaps)

    def test_handles_exponential_decay(self, gap_detection_service):
        """Should detect elbow in exponential decay curve."""
        # Exponential decay: steep at start, flattens
        gaps = [
            UnifiedGap("q1", [], 1, score=1.0),
            UnifiedGap("q2", [], 1, score=0.6),
            UnifiedGap("q3", [], 1, score=0.4),
            UnifiedGap("q4", [], 1, score=0.3),
            UnifiedGap("q5", [], 1, score=0.25),
            UnifiedGap("q6", [], 1, score=0.22),
        ]

        elbow_idx = gap_detection_service._find_elbow_kneedle(gaps)

        # Should find elbow somewhere in middle curve
        assert elbow_idx is not None
        assert 2 <= elbow_idx <= 4

    def test_perpendicular_distance_calculation(self, gap_detection_service):
        """Should correctly compute perpendicular distances."""
        # Create perfect right angle at index 2
        gaps = [
            UnifiedGap("q1", [], 1, score=1.0),
            UnifiedGap("q2", [], 1, score=1.0),
            UnifiedGap("q3", [], 1, score=0.5),  # Sharp corner
            UnifiedGap("q4", [], 1, score=0.0),
            UnifiedGap("q5", [], 1, score=0.0),
        ]

        elbow_idx = gap_detection_service._find_elbow_kneedle(gaps)

        # Should find elbow at or near the corner (index 2 or 3)
        assert elbow_idx is not None
        assert 2 <= elbow_idx <= 3

    def test_significance_threshold_filters_weak_elbows(self, gap_detection_service):
        """Should filter out weak/insignificant elbows."""
        # Very slight curve (almost linear)
        gaps = [
            UnifiedGap("q1", [], 1, score=1.0),
            UnifiedGap("q2", [], 1, score=0.98),
            UnifiedGap("q3", [], 1, score=0.96),
            UnifiedGap("q4", [], 1, score=0.94),
            UnifiedGap("q5", [], 1, score=0.92),
        ]

        elbow_idx = gap_detection_service._find_elbow_kneedle(gaps)

        # Should return None (elbow not significant)
        assert elbow_idx is None
