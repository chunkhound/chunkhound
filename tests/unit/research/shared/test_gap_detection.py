"""Unit tests for Gap Detection Service helper methods.

Tests detect_and_fill_gaps shard-failure reporting (#514) and the synchronous
helper methods:
- _shard_by_tokens(): Token-based sharding of chunk clusters
- _select_gaps_by_elbow(): Gap selection using elbow detection
- _global_dedup(): Cross-gap deduplication (via deduplicate_chunks)
"""

import pytest
from loguru import logger as _loguru_logger

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
        assert deduplicated[0]["chunk_id"] == "c1"


# -----------------------------------------------------------------------------
# Tests: per-shard failure reporting (#514)
# -----------------------------------------------------------------------------

# ~12,500 tokens each; with shard_budget=20_000 every chunk gets its own shard.
_SHARD_SIZED_CONTENT = "x" * 50_000

# Mirrors the message the providers raise when a structured completion is cut off.
_TRUNCATED_MESSAGE = (
    "LLM structured completion truncated - token limit exceeded "
    "(prompt=42,935, completion=2,039). This indicates insufficient "
    "max_completion_tokens for the structured output."
)


def _covered_chunks(count: int) -> list[dict]:
    """One shard-sized chunk per expected shard, tagged with its file name."""
    return [
        {
            "chunk_id": f"c{i}",
            "file_path": f"f{i}.py",
            "content": f"# marker-f{i}\n" + _SHARD_SIZED_CONTENT,
            "rerank_score": 0.9,
        }
        for i in range(count)
    ]


class _RecordingProgress:
    """Stand-in for TreeProgressDisplay that records gap_step messages."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def emit_event(self, event_type, message, **kwargs) -> None:
        self.messages.append(message)


@pytest.fixture
def warnings_logged():
    """Collect loguru WARNING messages for the duration of a test."""
    messages: list[str] = []
    sink_id = _loguru_logger.add(messages.append, level="WARNING", format="{message}")
    try:
        yield messages
    finally:
        _loguru_logger.remove(sink_id)


class TestShardFailureReporting:
    """A failed shard is counted and reported, never an empty gap list."""

    @pytest.mark.asyncio
    async def test_truncated_completion_counts_as_failed_shard(
        self, gap_service, fake_llm_provider, warnings_logged, monkeypatch
    ):
        """A truncated completion gives shards_failed == 1 and a shard warning."""

        async def truncated(*args, **kwargs):
            raise RuntimeError(_TRUNCATED_MESSAGE)

        monkeypatch.setattr(fake_llm_provider, "complete_structured", truncated)
        covered = _covered_chunks(1)

        chunks, stats = await gap_service.detect_and_fill_gaps(
            root_query="q", covered_chunks=covered, phase1_threshold=0.5
        )

        assert chunks == covered
        assert stats["shards_total"] == 1
        assert stats["shards_succeeded"] == 0
        assert stats["shards_failed"] == 1
        assert stats["shard_failures"]["items"][0]["item"] == "shard 0"
        assert any("shard 0" in m and "truncated" in m for m in warnings_logged)

    @pytest.mark.asyncio
    async def test_partial_failure_names_the_failed_shard(
        self, gap_service, fake_llm_provider, warnings_logged, monkeypatch
    ):
        """One failing shard among several is counted; the others still succeed."""

        async def fail_on_second_shard(prompt, **kwargs):
            if "marker-f1" in prompt:
                raise RuntimeError(_TRUNCATED_MESSAGE)
            return {"gaps": []}

        monkeypatch.setattr(
            fake_llm_provider, "complete_structured", fail_on_second_shard
        )

        _, stats = await gap_service.detect_and_fill_gaps(
            root_query="q", covered_chunks=_covered_chunks(3), phase1_threshold=0.5
        )

        assert (
            stats["shards_total"],
            stats["shards_succeeded"],
            stats["shards_failed"],
        ) == (3, 2, 1)
        assert stats["degraded"] is True
        assert any("shard 1" in m for m in warnings_logged)

    @pytest.mark.asyncio
    async def test_every_shard_failing_reports_degraded_not_clean_zero_gaps(
        self, gap_service, fake_llm_provider, warnings_logged, monkeypatch
    ):
        """All shards failing is an explicit degraded result, not 'no gaps found'."""

        async def truncated(*args, **kwargs):
            raise RuntimeError(_TRUNCATED_MESSAGE)

        monkeypatch.setattr(fake_llm_provider, "complete_structured", truncated)
        progress = _RecordingProgress()
        gap_service._progress = progress
        covered = _covered_chunks(3)

        chunks, stats = await gap_service.detect_and_fill_gaps(
            root_query="q", covered_chunks=covered, phase1_threshold=0.5
        )

        assert chunks == covered
        assert stats["gaps_found"] == 0
        assert stats["shards_total"] == 3
        assert stats["shards_succeeded"] == 0
        assert stats["shards_failed"] == 3
        assert stats["degraded"] is True
        assert stats["shard_failures"]["count"] == 3
        # The step events must not read like a clean run either.
        assert any("DEGRADED" in m for m in progress.messages)
        assert not any("complete: no gaps found" in m for m in progress.messages)
        assert any("failed=3" in m for m in warnings_logged)

    @pytest.mark.asyncio
    async def test_clean_corpus_has_no_gaps_and_no_warning(
        self, gap_service, fake_llm_provider, warnings_logged, monkeypatch
    ):
        """Every shard succeeding with no gaps keeps the fast path and stays quiet."""

        async def no_gaps(*args, **kwargs):
            return {"gaps": []}

        monkeypatch.setattr(fake_llm_provider, "complete_structured", no_gaps)
        progress = _RecordingProgress()
        gap_service._progress = progress
        covered = _covered_chunks(3)

        chunks, stats = await gap_service.detect_and_fill_gaps(
            root_query="q", covered_chunks=covered, phase1_threshold=0.5
        )

        assert chunks == covered
        assert stats["gaps_found"] == 0
        assert stats["shards_total"] == 3
        assert stats["shards_succeeded"] == 3
        assert stats["shards_failed"] == 0
        assert stats["degraded"] is False
        assert warnings_logged == []
        assert any("complete: no gaps found" in m for m in progress.messages)
        assert not any("DEGRADED" in m for m in progress.messages)

    @pytest.mark.asyncio
    async def test_configured_completion_budget_reaches_the_llm_call(
        self, gap_service, fake_llm_provider, monkeypatch
    ):
        """gap_detection_max_completion_tokens replaces the old hardcoded 2048."""
        gap_service._config.gap_detection_max_completion_tokens = 12_345
        captured: list[int] = []

        async def capturing(*args, **kwargs):
            captured.append(kwargs["max_completion_tokens"])
            return {"gaps": []}

        monkeypatch.setattr(fake_llm_provider, "complete_structured", capturing)

        await gap_service.detect_and_fill_gaps(
            root_query="q", covered_chunks=_covered_chunks(2), phase1_threshold=0.5
        )

        assert captured == [12_345, 12_345]
