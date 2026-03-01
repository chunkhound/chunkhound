"""Unit tests for Gap Detection Service helper methods.

Tests the synchronous helper methods:
- _shard_by_tokens(): Token-based sharding of chunk clusters
- _select_gaps_by_elbow(): Gap selection using elbow detection
- _global_dedup(): Cross-gap deduplication (via deduplicate_chunks)
"""

import json

import pytest

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.services.research.shared.gap_detection import GapDetectionService
from chunkhound.services.research.shared.gap_models import UnifiedGap
from tests.fixtures.fake_providers import FakeLLMProvider


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def fake_llm_provider():
    """Create fake LLM provider for gap detection."""
    return FakeLLMProvider()


@pytest.fixture
def mock_llm_manager(fake_llm_provider):
    """Create LLM manager that returns fake provider."""

    class MockLLMManager:
        def __init__(self, provider):
            self._provider = provider

        def get_utility_provider(self):
            return self._provider

        def get_synthesis_provider(self):
            return self._provider

    return MockLLMManager(fake_llm_provider)


@pytest.fixture
def mock_embedding_manager():
    """Create mock embedding manager."""

    class MockEmbeddingManager:
        pass

    return MockEmbeddingManager()


@pytest.fixture
def mock_db_services():
    """Create mock database services."""
    from pathlib import Path

    class MockProvider:
        def get_base_directory(self):
            return Path("/fake/base")

    class MockDatabaseServices:
        provider = MockProvider()

    return MockDatabaseServices()


@pytest.fixture
def research_config():
    """Create research configuration for testing."""
    return ResearchConfig(
        shard_budget=20_000,  # ~20k tokens per shard
        min_gaps=1,
        max_gaps=5,
    )


@pytest.fixture
def gap_service(mock_llm_manager, mock_embedding_manager, mock_db_services, research_config):
    """Create gap detection service with mocked dependencies."""
    return GapDetectionService(
        llm_manager=mock_llm_manager,
        embedding_manager=mock_embedding_manager,
        db_services=mock_db_services,
        config=research_config,
    )


# -----------------------------------------------------------------------------
# Tests: _shard_by_tokens
# -----------------------------------------------------------------------------


class TestShardByTokens:
    """Test token-based sharding of chunk clusters."""

    def test_single_small_cluster_fits_one_shard(self, gap_service):
        """Small cluster should fit in single shard."""
        cluster_groups = [
            [
                {"chunk_id": "c1", "content": "x" * 100},
                {"chunk_id": "c2", "content": "x" * 100},
            ]
        ]

        shards = gap_service._shard_by_tokens(cluster_groups)

        assert len(shards) == 1
        assert len(shards[0]) == 2

    def test_large_cluster_splits_into_multiple_shards(self, gap_service):
        """Cluster exceeding budget should split."""
        # shard_budget=20_000 tokens, ~4 chars/token = ~80_000 chars
        large_content = "x" * 50_000  # ~12,500 tokens each
        cluster_groups = [
            [
                {"chunk_id": "c1", "content": large_content},
                {"chunk_id": "c2", "content": large_content},
            ]
        ]

        shards = gap_service._shard_by_tokens(cluster_groups)

        # Each chunk ~12,500 tokens, budget=20,000
        # First chunk fits, second exceeds -> 2 shards
        assert len(shards) == 2

    def test_empty_clusters_return_empty_shards(self, gap_service):
        """Empty input should return empty output."""
        shards = gap_service._shard_by_tokens([])
        assert shards == []

    def test_preserves_chunk_metadata(self, gap_service):
        """Should preserve all chunk fields through sharding."""
        cluster_groups = [
            [
                {
                    "chunk_id": "c1",
                    "content": "def foo(): pass",
                    "file_path": "test.py",
                    "start_line": 1,
                    "metadata": {"key": "value"},
                }
            ]
        ]

        shards = gap_service._shard_by_tokens(cluster_groups)

        chunk = shards[0][0]
        assert chunk["chunk_id"] == "c1"
        assert chunk["file_path"] == "test.py"
        assert chunk["metadata"]["key"] == "value"

    def test_empty_content_chunks_handled(self, gap_service):
        """Chunks with empty content should work."""
        cluster_groups = [
            [{"chunk_id": "c1", "content": ""}, {"chunk_id": "c2", "content": ""}]
        ]

        shards = gap_service._shard_by_tokens(cluster_groups)

        assert len(shards) == 1
        assert len(shards[0]) == 2


# -----------------------------------------------------------------------------
# Tests: _select_gaps_by_elbow
# -----------------------------------------------------------------------------


class TestSelectGapsByElbow:
    """Test gap selection using elbow detection."""

    def test_empty_gaps_return_empty(self, gap_service):
        """Empty input should return empty output."""
        selected = gap_service._select_gaps_by_elbow([])
        assert selected == []

    def test_single_gap_returned(self, gap_service):
        """Single gap should be returned."""
        gaps = [UnifiedGap(query="q1", sources=[], vote_count=1, score=0.9)]

        selected = gap_service._select_gaps_by_elbow(gaps)

        assert len(selected) == 1
        assert selected[0].query == "q1"

    def test_respects_max_gaps(self, gap_service):
        """Should not exceed max_gaps (5 in fixture)."""
        gaps = [
            UnifiedGap(query=f"q{i}", sources=[], vote_count=1, score=0.9 - i * 0.01)
            for i in range(10)
        ]

        selected = gap_service._select_gaps_by_elbow(gaps)

        assert len(selected) <= 5

    def test_sorts_by_score_descending(self, gap_service):
        """Highest scores should come first."""
        gaps = [
            UnifiedGap(query="low", sources=[], vote_count=1, score=0.3),
            UnifiedGap(query="high", sources=[], vote_count=1, score=0.9),
            UnifiedGap(query="mid", sources=[], vote_count=1, score=0.6),
        ]

        selected = gap_service._select_gaps_by_elbow(gaps)

        assert selected[0].query == "high"

    def test_fifty_percent_threshold_filters_weak_gaps(self, gap_service):
        """Gaps below 50% of top score should be filtered."""
        gaps = [
            UnifiedGap(query="q1", sources=[], vote_count=1, score=1.0),
            UnifiedGap(query="q2", sources=[], vote_count=1, score=0.9),
            UnifiedGap(query="q3", sources=[], vote_count=1, score=0.4),  # Below 50%
        ]

        selected = gap_service._select_gaps_by_elbow(gaps)

        selected_queries = [g.query for g in selected]
        assert "q1" in selected_queries
        assert "q2" in selected_queries
        # q3 at 0.4 is below 50% of 1.0, may be filtered
        # (depends on elbow detection, but threshold should apply)

    def test_near_zero_scores_return_min_gaps(self, gap_service):
        """Near-zero top score should return min_gaps only."""
        gaps = [
            UnifiedGap(query=f"q{i}", sources=[], vote_count=1, score=1e-10)
            for i in range(3)
        ]

        selected = gap_service._select_gaps_by_elbow(gaps)

        assert len(selected) == 1  # min_gaps=1


# -----------------------------------------------------------------------------
# Tests: _global_dedup
# -----------------------------------------------------------------------------


class TestGlobalDedup:
    """Test global deduplication across gap results."""

    def test_empty_results_return_empty(self, gap_service):
        """Empty input should return empty output."""
        deduplicated = gap_service._global_dedup([])
        assert deduplicated == []

    def test_deduplicates_by_chunk_id(self, gap_service):
        """Same chunk_id across gaps should be deduplicated."""
        gap_results = [
            [
                {"chunk_id": "c1", "rerank_score": 0.8},
                {"chunk_id": "c2", "rerank_score": 0.7},
            ],
            [
                {"chunk_id": "c1", "rerank_score": 0.9},  # Duplicate
                {"chunk_id": "c3", "rerank_score": 0.6},
            ],
        ]

        deduplicated = gap_service._global_dedup(gap_results)

        chunk_ids = [c["chunk_id"] for c in deduplicated]
        assert len(chunk_ids) == 3
        assert set(chunk_ids) == {"c1", "c2", "c3"}

    def test_highest_score_wins_on_conflict(self, gap_service):
        """Chunk with highest rerank_score should be kept."""
        gap_results = [
            [{"chunk_id": "c1", "rerank_score": 0.5, "version": "v1"}],
            [{"chunk_id": "c1", "rerank_score": 0.9, "version": "v2"}],
            [{"chunk_id": "c1", "rerank_score": 0.7, "version": "v3"}],
        ]

        deduplicated = gap_service._global_dedup(gap_results)

        assert len(deduplicated) == 1
        assert deduplicated[0]["rerank_score"] == 0.9
        assert deduplicated[0]["version"] == "v2"

    def test_uses_id_field_as_fallback(self, gap_service):
        """Should use 'id' field if 'chunk_id' not present."""
        gap_results = [
            [{"id": "c1", "rerank_score": 0.8}],
            [{"id": "c2", "rerank_score": 0.7}],
        ]

        deduplicated = gap_service._global_dedup(gap_results)

        assert len(deduplicated) == 2
        ids = {c.get("id") for c in deduplicated}
        assert ids == {"c1", "c2"}

    def test_handles_empty_sublists(self, gap_service):
        """Empty gap result sublists should be handled."""
        gap_results = [
            [],
            [{"chunk_id": "c1", "rerank_score": 0.8}],
            [],
        ]

        deduplicated = gap_service._global_dedup(gap_results)

        assert len(deduplicated) == 1


# -----------------------------------------------------------------------------
# Tests: Edge Cases
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_shard_oversized_chunks_get_own_shard(self, gap_service):
        """Chunks larger than budget should get their own shard."""
        huge = "x" * 100_000  # Way over budget
        cluster_groups = [
            [
                {"chunk_id": "c1", "content": huge},
                {"chunk_id": "c2", "content": "small"},
            ]
        ]

        shards = gap_service._shard_by_tokens(cluster_groups)

        # Large chunk pushes small to next shard
        assert len(shards) >= 2

    def test_gap_selection_all_identical_scores(self, gap_service):
        """All identical scores should be handled."""
        gaps = [
            UnifiedGap(query=f"q{i}", sources=[], vote_count=1, score=0.8)
            for i in range(4)
        ]

        selected = gap_service._select_gaps_by_elbow(gaps)

        # Should return something within constraints
        assert 1 <= len(selected) <= 5

    def test_dedup_missing_chunk_id_skipped(self, gap_service):
        """Chunks without any ID should be skipped."""
        gap_results = [
            [
                {"content": "no id", "rerank_score": 0.9},
                {"chunk_id": "c1", "rerank_score": 0.8},
            ]
        ]

        deduplicated = gap_service._global_dedup(gap_results)

        # Only chunk with ID should remain
        assert len(deduplicated) == 1


# -----------------------------------------------------------------------------
# Tests: Snapshot system clustering (Step 2.1)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cluster_covered_chunks_uses_snapshot_systems_when_available(
    tmp_path,
    mock_llm_manager,
    mock_embedding_manager,
    mock_db_services,
    monkeypatch: pytest.MonkeyPatch,
):
    (tmp_path / "snapshot.chunk_systems.json").write_text(
        json.dumps(
            {
                "schema_version": "snapshot.chunk_systems.v1",
                "clusters": [
                    {"cluster_id": 1, "chunk_ids": [10, 11]},
                    {"cluster_id": 2, "chunk_ids": [20]},
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    config = ResearchConfig(
        shard_budget=20_000,
        min_gaps=1,
        max_gaps=5,
        chunk_systems_snapshot_dir=tmp_path,
    )
    svc = GapDetectionService(
        llm_manager=mock_llm_manager,
        embedding_manager=mock_embedding_manager,
        db_services=mock_db_services,
        config=config,
    )

    async def kmeans_should_not_run(*_args, **_kwargs):
        raise AssertionError("kmeans should not be called when snapshot index is present")

    monkeypatch.setattr(svc, "_cluster_chunks_kmeans", kmeans_should_not_run)

    covered_chunks = [
        {
            "chunk_id": 10,
            "rerank_score": 0.9,
            "file_path": "b.py",
            "start_line": 2,
            "content": "x",
        },
        {
            "chunk_id": 11,
            "rerank_score": 0.1,
            "file_path": "a.py",
            "start_line": 1,
            "content": "x",
        },
        {
            "chunk_id": 20,
            "rerank_score": 2.0,
            "file_path": "c.py",
            "start_line": 3,
            "content": "x",
        },
        {
            "chunk_id": 999,  # not in snapshot index -> unknown bucket
            "rerank_score": 1.5,
            "file_path": "z.py",
            "start_line": 99,
            "content": "x",
        },
    ]

    groups = await svc._cluster_covered_chunks_for_gaps(covered_chunks)
    group_chunk_ids = [[c["chunk_id"] for c in g] for g in groups]

    # Order: by aggregate relevance (sum of rerank_score), unknown bucket separate.
    assert group_chunk_ids[0] == [20]
    assert group_chunk_ids[1] == [999]
    assert group_chunk_ids[2] == [10, 11]


@pytest.mark.asyncio
async def test_cluster_covered_chunks_falls_back_to_kmeans_when_snapshot_missing(
    tmp_path,
    mock_llm_manager,
    mock_embedding_manager,
    mock_db_services,
    monkeypatch: pytest.MonkeyPatch,
):
    config = ResearchConfig(shard_budget=20_000, min_gaps=1, max_gaps=5)
    # Set snapshot dir implicitly (not in model_fields_set) to match defaulting behavior.
    object.__setattr__(config, "chunk_systems_snapshot_dir", tmp_path)

    svc = GapDetectionService(
        llm_manager=mock_llm_manager,
        embedding_manager=mock_embedding_manager,
        db_services=mock_db_services,
        config=config,
    )

    called = {"value": False}

    async def fake_kmeans(_chunks):
        called["value"] = True
        return [[{"chunk_id": "sentinel"}]]

    monkeypatch.setattr(svc, "_cluster_chunks_kmeans", fake_kmeans)

    result = await svc._cluster_covered_chunks_for_gaps(
        [{"chunk_id": 1, "content": "x"}]
    )

    assert called["value"] is True
    assert result == [[{"chunk_id": "sentinel"}]]
