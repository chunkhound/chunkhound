"""Edge case tests for gap selection logic.

Tests the _select_gaps_by_elbow() method to expose the zero score bug and
other edge cases in the gap selection heuristic.

BUG CONTEXT:
At line 584 in gap_detection.py:
    if sorted_gaps[i].score < 0.5 * sorted_gaps[0].score: break

When all scores are 0.0:
    0.5 * 0.0 = 0.0
    0.0 < 0.0 is False
    → Loop never breaks → selects ALL gaps incorrectly

Expected fix (documented in each test):
    Check for near-zero top score before applying heuristic
"""

import pytest

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.v2.gap_detection import GapDetectionService
from chunkhound.services.research.v2.models import GapCandidate, UnifiedGap
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider


@pytest.fixture
def fake_llm_provider():
    """Create fake LLM provider for gap detection."""
    return FakeLLMProvider(responses={})


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
    """Create research configuration with typical min/max gaps."""
    return ResearchConfig(
        shard_budget=20_000,
        min_cluster_size=2,
        gap_similarity_threshold=0.3,
        min_gaps=1,  # At minimum, select 1 gap
        max_gaps=10,  # At maximum, select 10 gaps
        max_symbols=10,
        query_expansion_enabled=True,
        target_tokens=10000,
        max_chunks_per_file_repr=5,
        max_tokens_per_file_repr=2000,
        max_boundary_expansion_lines=300,
        max_compression_iterations=3,
    )


@pytest.fixture
def research_config_zero_min():
    """Create research configuration with min_gaps=0 for testing zero selection."""
    return ResearchConfig(
        shard_budget=20_000,
        min_cluster_size=2,
        gap_similarity_threshold=0.3,
        min_gaps=0,  # Allow selecting 0 gaps
        max_gaps=10,
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
    """Create gap detection service with standard config."""
    return GapDetectionService(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=research_config,
    )


@pytest.fixture
def gap_detection_service_zero_min(
    llm_manager, embedding_manager, db_services, research_config_zero_min
):
    """Create gap detection service with min_gaps=0."""
    return GapDetectionService(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=research_config_zero_min,
    )


def create_gap(query: str, confidence: float, source_shard: int = 0) -> UnifiedGap:
    """Helper to create UnifiedGap with specific confidence score.

    Args:
        query: Gap query string
        confidence: Confidence score (becomes gap.score)
        source_shard: Source shard identifier

    Returns:
        UnifiedGap with score set to confidence
    """
    candidate = GapCandidate(
        query=query,
        rationale=f"rationale for {query}",
        confidence=confidence,
        source_shard=source_shard,
    )
    return UnifiedGap(
        query=query,
        sources=[candidate],
        vote_count=1,
        avg_confidence=confidence,
        score=confidence,
    )


class TestAllZeroScores:
    """Test gap selection when all gaps have confidence 0.0.

    BUG: When all scores are 0.0, the condition `0.0 < 0.5 * 0.0` evaluates to
    `0.0 < 0.0` which is False, causing the loop to continue and select ALL gaps.

    EXPECTED FIX: Check if top score is near-zero before applying 50% heuristic:
        if sorted_gaps[0].score <= 0.01:  # Near-zero confidence
            logger.warning("Top gap has near-zero confidence, selecting min_gaps only")
            return sorted_gaps[:min_gaps]
    """

    def test_all_zero_scores_should_select_min_gaps_only(self, gap_detection_service):
        """BUG: All gaps with 0.0 confidence should select only min_gaps, not all gaps.

        Current buggy behavior: Selects ALL 5 gaps (loop never breaks)
        Expected behavior: Select min_gaps=1 gap only (no confidence to select more)
        """
        gaps = [create_gap(f"query{i}", 0.0) for i in range(5)]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # EXPECTED: Should select min_gaps=1 only (no confidence for more)
        # ACTUAL BUG: Selects all 5 gaps (0.0 < 0.0 is False, loop never breaks)
        assert len(selected) == 1, (
            f"Expected min_gaps=1 selected with all-zero scores, got {len(selected)}. "
            "This exposes the bug: 0.0 < 0.5*0.0 is False, loop doesn't break."
        )

    def test_all_zero_scores_with_min_gaps_zero(self, gap_detection_service_zero_min):
        """All gaps with 0.0 confidence and min_gaps=0 should select 0 gaps.

        When min_gaps=0 and all scores are 0.0, no confidence to select any
        gaps.
        """
        gaps = [create_gap(f"query{i}", 0.0) for i in range(5)]

        selected = gap_detection_service_zero_min._select_gaps_by_elbow(gaps)

        # EXPECTED: min_gaps=0 means we can select nothing when confidence zero
        # ACTUAL BUG: May select all gaps due to 0.0 < 0.0 = False
        assert len(selected) == 0, (
            f"Expected 0 gaps (min_gaps=0, all-zero scores), got {len(selected)}"
            ". With no confidence, should select nothing."
        )

    def test_many_zero_scores_caps_at_max_gaps(self, gap_detection_service):
        """BUG: Even with 50 gaps at 0.0 confidence, should not exceed max_gaps.

        Tests interaction between zero-score bug and max_gaps constraint.
        """
        # Create 50 gaps with 0.0 confidence (max_gaps=10)
        gaps = [create_gap(f"query{i}", 0.0) for i in range(50)]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # EXPECTED: min_gaps=1 (or max_gaps=10 if bug triggers)
        # ACTUAL BUG: May select up to max_gaps due to zero-score bug in fallback
        assert len(selected) <= 10, (
            f"Expected at most max_gaps=10 selected, got {len(selected)}. "
            "Even with bug, max_gaps constraint should hold."
        )

        # More specific assertion: With fix, should select min_gaps=1 only
        assert len(selected) == 1, (
            f"Expected min_gaps=1 with all-zero scores, got {len(selected)}. "
            "Bug may cause selection of more gaps than justified by confidence."
        )


class TestNearZeroScores:
    """Test gap selection with near-zero scores (0.0, 0.0, 0.01, 0.02).

    Tests that selection logic handles scores very close to zero without
    selecting all gaps.
    """

    def test_near_zero_scores_with_small_variation(self, gap_detection_service):
        """BUG: Gaps with near-zero scores should use min_gaps logic.

        Tests: [0.0, 0.0, 0.01, 0.02]

        Current behavior: May select too many gaps due to 0.0 threshold issue.
        Expected: Select min_gaps=1 when top score is near-zero.
        """
        gaps = [
            create_gap("query1", 0.02),  # Barely above zero
            create_gap("query2", 0.01),  # Almost zero
            create_gap("query3", 0.0),
            create_gap("query4", 0.0),
        ]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # EXPECTED: Top score 0.02 is too low to justify multiple selections
        # 50% threshold: 0.01 < 0.5 * 0.02 = 0.01 is False (boundary case)
        # With fix: Should select min_gaps=1 when top score <= 0.01 threshold
        assert len(selected) >= 1, "Should select at least min_gaps=1"
        assert len(selected) <= 2, (
            f"Expected 1-2 gaps with near-zero scores, got {len(selected)}. "
            "Very low confidence should limit selection."
        )

    def test_near_zero_with_one_outlier(self, gap_detection_service):
        """Single higher-confidence gap among near-zero scores is selective.

        [0.1, 0.01, 0.0, 0.0] - First gap has 10x confidence of second.
        """
        gaps = [
            create_gap("query1", 0.1),  # 10x higher than others
            create_gap("query2", 0.01),
            create_gap("query3", 0.0),
            create_gap("query4", 0.0),
        ]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # 50% threshold: 0.01 < 0.5 * 0.1 = 0.05, so should stop at query1
        assert len(selected) == 1, (
            f"Expected 1 gap (second gap drops below 50%), got {len(selected)}"
            ". Should stop at first gap when score drops significantly."
        )
        assert selected[0].query == "query1"
        assert selected[0].score == 0.1


class TestSingleGapZeroScore:
    """Test gap selection with single gap at 0.0 confidence.

    Tests min_gaps logic when only one gap exists with zero confidence.
    """

    def test_single_gap_zero_score_min_gaps_one(self, gap_detection_service):
        """Single gap with 0.0 confidence and min_gaps=1 should select that gap.

        Even with zero confidence, min_gaps=1 forces selection.
        """
        gaps = [create_gap("query1", 0.0)]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # EXPECTED: len(gaps) <= min_gaps, so return all gaps (special case)
        assert len(selected) == 1, (
            f"Expected 1 gap selected (len <= min_gaps), got {len(selected)}. "
            "Should return all gaps when count <= min_gaps."
        )
        assert selected[0].score == 0.0

    def test_single_gap_zero_score_min_gaps_zero(self, gap_detection_service_zero_min):
        """Single gap with 0.0 confidence and min_gaps=0 should select 0 gaps.

        With min_gaps=0, zero confidence justifies selecting nothing.
        """
        gaps = [create_gap("query1", 0.0)]

        selected = gap_detection_service_zero_min._select_gaps_by_elbow(gaps)

        # EXPECTED: With min_gaps=0 and zero confidence, select nothing
        # ACTUAL: len(gaps) <= min_gaps (1 <= 0 is False), falls to heuristic
        # Heuristic: starts with sorted_gaps[:0] = [], loop runs but 0.0 < 0.0 is False
        # Bug means we append all remaining gaps (just 1 gap)
        assert len(selected) == 0, (
            f"Expected 0 gaps selected (min_gaps=0, zero score), got {len(selected)}. "
            "Should select nothing when min_gaps=0 and confidence is zero."
        )


class TestIdenticalNonZeroScores:
    """Test gap selection when all gaps have identical non-zero scores.

    Tests elbow fallback behavior when no score variation exists.
    """

    def test_all_identical_scores_uses_max_gaps_limit(self, gap_detection_service):
        """All gaps with identical score 0.5 should respect max_gaps limit.

        Kneedle should return None (no elbow in flat distribution).
        Fallback heuristic: 0.5 < 0.5 * 0.5 = 0.25 is False, selects all gaps.
        """
        gaps = [create_gap(f"query{i}", 0.5) for i in range(8)]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # EXPECTED: Kneedle returns None (flat), fallback selects all
        # All gaps have same score, so 50% heuristic never breaks (0.5 < 0.25 is False)
        # Should select all 8 gaps (within max_gaps=10)
        assert len(selected) == 8, (
            f"Expected 8 gaps selected (all identical scores), got {len(selected)}. "
            "Fallback heuristic should select all when no score drop."
        )

    def test_identical_scores_exceeding_max_gaps(self, gap_detection_service):
        """15 gaps with identical score should cap at max_gaps=10.

        Tests that max_gaps constraint applies even when all scores are identical.
        """
        gaps = [create_gap(f"query{i}", 0.7) for i in range(15)]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # EXPECTED: More than max_gaps, use kneedle on first 10
        # Kneedle on flat distribution returns None, fallback to max_gaps
        assert len(selected) == 10, (
            f"Expected max_gaps=10 (15 candidates, all identical), got "
            f"{len(selected)}. Should cap at max_gaps when over limit."
        )

    def test_identical_scores_at_boundary(self, gap_detection_service):
        """Exactly max_gaps candidates with identical scores should select all.

        10 gaps with identical scores (max_gaps=10).
        """
        gaps = [create_gap(f"query{i}", 0.6) for i in range(10)]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # EXPECTED: len(gaps) <= max_gaps, kneedle None, fallback selects all
        assert len(selected) == 10, (
            f"Expected 10 gaps (at max_gaps limit, identical scores), got "
            f"{len(selected)}. Should select all at boundary, no variation."
        )


class TestMixedScoresNearThreshold:
    """Test gap selection with scores near the 50% drop threshold.

    Tests boundary conditions for the 50% heuristic: [1.0, 0.55, 0.51, 0.49, 0.1]
    """

    def test_scores_crossing_fifty_percent_threshold(self, gap_detection_service):
        """Scores that cross 50% threshold should stop at boundary.

        [1.0, 0.55, 0.51, 0.49, 0.1]
        50% of top: 0.5
        Should stop before 0.49 (first score < 0.5)
        """
        gaps = [
            create_gap("query1", 1.0),
            create_gap("query2", 0.55),
            create_gap("query3", 0.51),
            create_gap("query4", 0.49),  # Below 50% threshold
            create_gap("query5", 0.1),
        ]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # EXPECTED: Stop before query4 (0.49 < 0.5)
        # Kneedle might find elbow at different point
        assert len(selected) >= 1, "Should select at least min_gaps=1"
        assert len(selected) <= 3, (
            f"Expected 1-3 gaps selected (stop at 50% threshold), got {len(selected)}. "
            "Should stop when score drops below 50% of top score."
        )

        # Verify we didn't select gaps below threshold
        if len(selected) >= 3:
            # If we selected 3, verify last one is above threshold
            assert selected[-1].score >= 0.5, (
                f"Last selected gap has score {selected[-1].score}, expected >= 0.5. "
                "Should not select gaps below 50% threshold."
            )

    def test_scores_just_above_threshold(self, gap_detection_service):
        """Scores just above 50% threshold should be selected.

        [1.0, 0.51, 0.50, 0.49]
        50% of top: 0.5
        Should select first 3 gaps (0.51, 0.50 >= 0.5; stop before 0.49)
        """
        gaps = [
            create_gap("query1", 1.0),
            create_gap("query2", 0.51),
            create_gap("query3", 0.50),  # Exactly at boundary
            create_gap("query4", 0.49),  # Just below
        ]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # EXPECTED: Select first 3 (stop before 0.49)
        # Note: Floating point comparison might cause edge cases
        assert len(selected) >= 1, "Should select at least min_gaps=1"
        assert len(selected) <= 3, (
            f"Expected 1-3 gaps selected (boundary test), got {len(selected)}. "
            "Boundary case: 0.50 >= 0.50 should be included, 0.49 < 0.50 excluded."
        )

    def test_gradual_decline_with_clear_drop(self, gap_detection_service):
        """Gradual decline followed by sharp drop should stop at drop.

        [1.0, 0.9, 0.85, 0.82, 0.3, 0.2]
        50% threshold: 0.5
        Should select first 4 gaps (stop before 0.3)
        """
        gaps = [
            create_gap("query1", 1.0),
            create_gap("query2", 0.9),
            create_gap("query3", 0.85),
            create_gap("query4", 0.82),
            create_gap("query5", 0.3),  # Sharp drop here
            create_gap("query6", 0.2),
        ]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # EXPECTED: Kneedle likely finds elbow around gap 4-5
        # Fallback heuristic: stop before 0.3 (below 50% threshold)
        assert len(selected) >= 1, "Should select at least min_gaps=1"
        assert len(selected) <= 5, (
            f"Expected 1-5 gaps selected (stop at sharp drop), got {len(selected)}. "
            "Should detect sharp drop and stop selection."
        )

        # Verify last selected gap is above threshold
        assert selected[-1].score >= 0.5, (
            f"Last selected gap has score {selected[-1].score}, expected >= 0.5. "
            "Should not include gaps below 50% threshold."
        )

    def test_alternating_scores_above_below_threshold(self, gap_detection_service):
        """Scores alternating above/below threshold should stop at first below.

        [1.0, 0.6, 0.4, 0.7]
        50% threshold: 0.5
        Should select first 2 gaps (stop at 0.4)
        """
        gaps = [
            create_gap("query1", 1.0),
            create_gap("query2", 0.6),
            create_gap("query3", 0.4),  # Below threshold
            create_gap("query4", 0.7),  # Above again (not selected)
        ]

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # EXPECTED: Stop at first gap below threshold (0.4 < 0.5)
        # Note: Gaps are sorted by score descending in implementation
        # So actual order will be [1.0, 0.7, 0.6, 0.4]
        # This test verifies sorting works correctly

        # After sorting: [1.0, 0.7, 0.6, 0.4]
        # 50% threshold: 0.5
        # Should select all first 3 (stop before 0.4)
        assert len(selected) >= 1, "Should select at least min_gaps=1"
        assert len(selected) <= 3, (
            f"Expected 1-3 gaps (sorted, stop at threshold), got "
            f"{len(selected)}. Should sort and apply threshold."
        )
        # Verify gaps are sorted descending
        selected_scores = [g.score for g in selected]
        assert selected_scores == sorted(selected_scores, reverse=True), (
            "Selected gaps should be sorted by score descending"
        )
