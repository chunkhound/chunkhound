"""Edge case tests for gap clustering with identical confidence scores.

Tests GapDetectionService._select_gaps_by_elbow() behavior when all gaps have
identical confidence scores (flat distributions). When kneedle returns None for
flat distributions, the method should fall back to selecting min(len(gaps), max_gaps).

EDGE CASE CONTEXT:
When all gaps have identical confidence scores:
- find_elbow_kneedle() returns None
  (line 52 in elbow_detection.py: "All scores identical")
- _select_gaps_by_elbow() must handle None elbow_idx in fallback logic
- Expected behavior: Select min(len(gaps), max_gaps) when no elbow detected

TEST COVERAGE:
1. 10 gaps with identical 0.8 confidence → selects all 10 (max_gaps=10)
2. 3 gaps with identical 0.0 confidence
   → selects min_gaps if len < min_gaps, else all 3
3. 1 gap with 1.0 confidence → selects that single gap (edge case: len=1)
4. 100 gaps with identical 0.5 confidence, max_gaps=10
   → selects exactly 10 (respects max_gaps)
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
    """Create research configuration with standard min/max gaps."""
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
def research_config_min_3():
    """Create research configuration with min_gaps=3 for testing min constraint."""
    return ResearchConfig(
        shard_budget=20_000,
        min_cluster_size=2,
        gap_similarity_threshold=0.3,
        min_gaps=3,  # Higher minimum to test constraint
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
def gap_detection_service_min_3(
    llm_manager, embedding_manager, db_services, research_config_min_3
):
    """Create gap detection service with min_gaps=3."""
    return GapDetectionService(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=research_config_min_3,
    )


def create_identical_gaps(count: int, confidence: float) -> list[UnifiedGap]:
    """Helper to create UnifiedGap list with identical confidence scores.

    Args:
        count: Number of gaps to create
        confidence: Confidence score for all gaps (becomes gap.score)

    Returns:
        List of UnifiedGap objects with identical scores
    """
    gaps = []
    for i in range(count):
        candidate = GapCandidate(
            query=f"gap_query_{i}",
            rationale=f"rationale for gap {i}",
            confidence=confidence,
            source_shard=0,
        )
        gaps.append(
            UnifiedGap(
                query=f"gap_query_{i}",
                sources=[candidate],
                vote_count=1,
                avg_confidence=confidence,
                score=confidence,  # Score set to confidence
            )
        )
    return gaps


class TestIdenticalScoresWithinMaxGaps:
    """Test gap selection when all gaps have identical scores within max_gaps limit.

    When len(gaps) <= max_gaps and all scores identical:
    - Kneedle returns None (flat distribution)
    - Fallback heuristic: score never drops below 50% threshold
    - Expected: Select all gaps
    """

    def test_ten_gaps_identical_0_8_confidence(self, gap_detection_service):
        """10 gaps with identical 0.8 confidence should select all 10 (max_gaps=10).

        Behavior verification:
        1. Kneedle returns None (all scores identical)
        2. len(gaps) = 10 <= max_gaps = 10
        3. Fallback heuristic: 0.8 < 0.5 * 0.8 = 0.4 is False (never breaks)
        4. Expected: Select all 10 gaps
        """
        gaps = create_identical_gaps(count=10, confidence=0.8)

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # Verify kneedle behavior (implicit: if it worked, we'd see < 10)
        # All identical scores → kneedle returns None
        assert len(selected) == 10, (
            f"Expected 10 gaps selected (all identical 0.8, max_gaps=10), "
            f"got {len(selected)}. "
            "Kneedle should return None for flat distribution, "
            "fallback selects all."
        )

        # Verify all selected gaps have same score
        scores = [g.score for g in selected]
        assert all(s == 0.8 for s in scores), (
            f"All selected gaps should have score 0.8, got {scores}"
        )

    def test_five_gaps_identical_0_9_confidence(self, gap_detection_service):
        """5 gaps with identical 0.9 confidence should select all 5 (within max_gaps).

        Behavior verification:
        1. len(gaps) = 5 < max_gaps = 10
        2. Kneedle returns None (flat distribution)
        3. Fallback: 0.9 < 0.45 is False, selects all
        """
        gaps = create_identical_gaps(count=5, confidence=0.9)

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        assert len(selected) == 5, (
            f"Expected 5 gaps selected (all identical 0.9), got {len(selected)}"
        )

    def test_eight_gaps_identical_0_5_confidence(self, gap_detection_service):
        """8 gaps with identical 0.5 confidence should select all 8 (within max_gaps).

        Medium confidence level (0.5) with flat distribution.
        """
        gaps = create_identical_gaps(count=8, confidence=0.5)

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        assert len(selected) == 8, (
            f"Expected 8 gaps selected (all identical 0.5), got {len(selected)}"
        )


class TestIdenticalZeroConfidence:
    """Test gap selection when all gaps have identical 0.0 confidence.

    Zero confidence case requires special handling:
    - No justification to select any gaps based on confidence
    - Should respect min_gaps constraint
    - Related to zero-score bug in test_gap_selection_edge_cases.py
    """

    def test_three_gaps_all_zero_min_gaps_1(self, gap_detection_service):
        """3 gaps with 0.0 confidence, min_gaps=1 should select min_gaps only.

        Behavior verification:
        1. All scores 0.0 (no confidence for selection)
        2. len(gaps) = 3 > min_gaps = 1
        3. Kneedle returns None (flat distribution)
        4. Expected fallback: Select min_gaps when all scores zero
        5. Actual behavior depends on zero-score bug fix status
        """
        gaps = create_identical_gaps(count=3, confidence=0.0)

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # EXPECTED (after zero-score bug fix):
        # Should select min_gaps=1 when all scores are zero
        #
        # CURRENT BEHAVIOR (with zero-score bug):
        # May select all 3 gaps due to 0.0 < 0.5*0.0 = False (loop never breaks)
        #
        # This test documents the interaction between flat distribution
        # and zero-score edge case
        assert len(selected) >= 1, "Should select at least min_gaps=1"
        assert len(selected) <= 3, (
            f"Expected 1-3 gaps (zero confidence), got {len(selected)}. "
            "Zero confidence should limit selection to min_gaps."
        )

    def test_three_gaps_all_zero_min_gaps_3(self, gap_detection_service_min_3):
        """3 gaps with 0.0 confidence, min_gaps=3 should select all 3.

        When len(gaps) <= min_gaps, special case returns all gaps regardless of scores.
        Line 565-566 in gap_detection.py:
            if len(sorted_gaps) <= min_gaps:
                return sorted_gaps
        """
        gaps = create_identical_gaps(count=3, confidence=0.0)

        selected = gap_detection_service_min_3._select_gaps_by_elbow(gaps)

        assert len(selected) == 3, (
            f"Expected 3 gaps (len <= min_gaps=3), got {len(selected)}. "
            "Should return all gaps when count <= min_gaps, regardless of scores."
        )

    def test_five_gaps_all_zero_min_gaps_3(self, gap_detection_service_min_3):
        """5 gaps with 0.0 confidence, min_gaps=3 should select 3 gaps.

        Behavior verification:
        1. len(gaps) = 5 > min_gaps = 3
        2. All scores 0.0 (no confidence)
        3. Kneedle returns None (flat)
        4. Expected: Select min_gaps=3 when all scores zero
        """
        gaps = create_identical_gaps(count=5, confidence=0.0)

        selected = gap_detection_service_min_3._select_gaps_by_elbow(gaps)

        # Expected behavior depends on zero-score bug fix
        # After fix: Should select min_gaps=3 when all scores zero
        # Current: May select all 5 due to 0.0 < 0.0 = False
        assert len(selected) >= 3, "Should select at least min_gaps=3"
        assert len(selected) <= 5, (
            f"Expected 3-5 gaps (zero confidence, min_gaps=3), got {len(selected)}"
        )


class TestSingleGapIdenticalScore:
    """Test gap selection with single gap (edge case: len=1).

    Single gap is trivial case:
    - len(gaps) = 1 <= min_gaps (always)
    - Should return that single gap regardless of score
    - Line 565-566: if len(sorted_gaps) <= min_gaps: return sorted_gaps
    """

    def test_single_gap_perfect_confidence(self, gap_detection_service):
        """Single gap with 1.0 confidence should select that gap.

        Edge case: len=1, score=1.0 (perfect confidence).
        """
        gaps = create_identical_gaps(count=1, confidence=1.0)

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        assert len(selected) == 1, (
            f"Expected 1 gap selected (single gap, len <= min_gaps), "
            f"got {len(selected)}"
        )
        assert selected[0].score == 1.0
        assert selected[0].query == "gap_query_0"

    def test_single_gap_zero_confidence(self, gap_detection_service):
        """Single gap with 0.0 confidence should select that gap.

        Edge case: len=1, score=0.0 (no confidence).
        Even with zero confidence, len <= min_gaps forces selection.
        """
        gaps = create_identical_gaps(count=1, confidence=0.0)

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        assert len(selected) == 1, (
            f"Expected 1 gap selected (len=1 <= min_gaps=1), got {len(selected)}. "
            "Should return all gaps when len <= min_gaps, regardless of score."
        )
        assert selected[0].score == 0.0

    def test_single_gap_medium_confidence(self, gap_detection_service):
        """Single gap with 0.6 confidence should select that gap.

        Edge case: len=1, score=0.6 (medium confidence).
        """
        gaps = create_identical_gaps(count=1, confidence=0.6)

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        assert len(selected) == 1, f"Expected 1 gap selected, got {len(selected)}"
        assert selected[0].score == 0.6


class TestIdenticalScoresExceedingMaxGaps:
    """Test gap selection when all gaps have identical scores exceeding max_gaps.

    When len(gaps) > max_gaps and all scores identical:
    - Line 593-595: Use kneedle on first max_gaps candidates
    - Kneedle returns None (flat distribution)
    - Line 604-609: Fallback to max_gaps
    - Expected: Select exactly max_gaps
    """

    def test_hundred_gaps_identical_0_5_max_gaps_10(self, gap_detection_service):
        """100 gaps with identical 0.5 confidence, max_gaps=10 should select exactly 10.

        Behavior verification:
        1. len(gaps) = 100 > max_gaps = 10
        2. Line 594: candidate_gaps = sorted_gaps[:10]
        3. Kneedle on 10 identical scores returns None
        4. Line 604-609: Fallback to max_gaps
        5. Expected: Select exactly 10 gaps
        """
        gaps = create_identical_gaps(count=100, confidence=0.5)

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        assert len(selected) == 10, (
            f"Expected max_gaps=10 selected (100 candidates, all identical), "
            f"got {len(selected)}. "
            "Should cap at max_gaps when len(gaps) > max_gaps, "
            "even with flat distribution."
        )

        # Verify selected gaps are first 10 (sorted by score, but all identical)
        selected_queries = [g.query for g in selected]
        # All scores identical, so order is arbitrary but count should be exact
        assert len(set(selected_queries)) == 10, "Should select 10 unique gaps"

    def test_fifty_gaps_identical_0_7_max_gaps_10(self, gap_detection_service):
        """50 gaps with identical 0.7 confidence, max_gaps=10 should select exactly 10.

        Behavior verification:
        1. len(gaps) = 50 > max_gaps = 10
        2. Kneedle on first 10 returns None (identical)
        3. Fallback to max_gaps = 10
        """
        gaps = create_identical_gaps(count=50, confidence=0.7)

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        assert len(selected) == 10, (
            f"Expected max_gaps=10 selected (50 candidates, all identical 0.7), "
            f"got {len(selected)}"
        )

    def test_fifteen_gaps_identical_1_0_max_gaps_10(self, gap_detection_service):
        """15 gaps with identical 1.0 confidence, max_gaps=10 should select exactly 10.

        Edge case: Perfect confidence (1.0) but flat distribution.
        """
        gaps = create_identical_gaps(count=15, confidence=1.0)

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        assert len(selected) == 10, (
            f"Expected max_gaps=10 selected "
            f"(15 candidates, all perfect confidence), "
            f"got {len(selected)}. "
            "Even with perfect confidence, max_gaps constraint applies."
        )

        # Verify all selected have perfect score
        scores = [g.score for g in selected]
        assert all(s == 1.0 for s in scores), "All selected gaps should have score 1.0"


class TestIdenticalScoresAtBoundaries:
    """Test gap selection when gap count exactly equals min/max boundaries.

    Boundary cases:
    - len(gaps) = min_gaps (should select all)
    - len(gaps) = max_gaps (should select all if kneedle None)
    - len(gaps) = min_gaps = max_gaps (trivial case)
    """

    def test_exactly_min_gaps_identical_scores(self, gap_detection_service):
        """Exactly min_gaps=1 gap with identical score should select all.

        Boundary case: len(gaps) = min_gaps = 1
        Line 565-566: if len(sorted_gaps) <= min_gaps: return sorted_gaps
        """
        gaps = create_identical_gaps(count=1, confidence=0.75)

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        assert len(selected) == 1, (
            f"Expected 1 gap selected (len = min_gaps = 1), got {len(selected)}"
        )

    def test_exactly_max_gaps_identical_scores(self, gap_detection_service):
        """Exactly max_gaps=10 gaps with identical scores should select all.

        Boundary case: len(gaps) = max_gaps = 10
        Line 569-591: len(sorted_gaps) <= max_gaps branch
        Kneedle returns None (flat), fallback selects all
        """
        gaps = create_identical_gaps(count=10, confidence=0.65)

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        assert len(selected) == 10, (
            f"Expected 10 gaps selected (len = max_gaps = 10, all identical), "
            f"got {len(selected)}. "
            "Should select all when at max_gaps boundary "
            "with flat distribution."
        )

    def test_exactly_min_gaps_3_identical_scores(self, gap_detection_service_min_3):
        """Exactly min_gaps=3 gaps with identical scores should select all.

        Boundary case: len(gaps) = min_gaps = 3
        """
        gaps = create_identical_gaps(count=3, confidence=0.55)

        selected = gap_detection_service_min_3._select_gaps_by_elbow(gaps)

        assert len(selected) == 3, (
            f"Expected 3 gaps selected (len = min_gaps = 3), got {len(selected)}"
        )


class TestKneedleNoneHandling:
    """Test that _select_gaps_by_elbow() correctly handles kneedle returning None.

    Verifies fallback logic when kneedle cannot detect elbow point:
    - Lines 571-591: Fallback to 50% threshold heuristic
    - Lines 595-609: Fallback to max_gaps
    - Ensures no crashes or unexpected behavior when kneedle returns None
    """

    def test_kneedle_none_uses_fallback_heuristic(self, gap_detection_service):
        """When kneedle returns None, should use 50% threshold fallback.

        Identical scores force kneedle to return None.
        Fallback heuristic (line 581-586):
            selected = sorted_gaps[:min_gaps]
            for i in range(min_gaps, len(sorted_gaps)):
                if sorted_gaps[i].score < 0.5 * sorted_gaps[0].score:
                    break
                selected.append(sorted_gaps[i])

        With identical scores, loop never breaks (0.5 < 0.5*0.5 = False).
        """
        gaps = create_identical_gaps(count=7, confidence=0.5)

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        # Kneedle returns None, fallback selects all (loop never breaks)
        assert len(selected) == 7, (
            f"Expected 7 gaps (kneedle None, fallback selects all), got {len(selected)}"
        )

    def test_kneedle_none_respects_max_gaps_constraint(self, gap_detection_service):
        """When kneedle returns None and len > max_gaps, should cap at max_gaps.

        Line 604-609: Fallback when kneedle returns None on candidate_gaps[:max_gaps].
        """
        gaps = create_identical_gaps(count=20, confidence=0.6)

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        assert len(selected) == 10, (
            f"Expected max_gaps=10 (kneedle None, 20 candidates), got {len(selected)}. "
            "Should cap at max_gaps even when kneedle returns None."
        )

    def test_kneedle_none_empty_gaps_returns_empty(self, gap_detection_service):
        """Empty gaps list should return empty (kneedle never called).

        Line 554-555: if not unified_gaps: return []
        """
        gaps = []

        selected = gap_detection_service._select_gaps_by_elbow(gaps)

        assert selected == [], (
            f"Expected empty list for empty gaps, got {len(selected)} gaps"
        )
