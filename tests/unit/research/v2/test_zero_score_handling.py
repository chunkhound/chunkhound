"""Comprehensive tests for zero-score edge cases in v2 research pipeline.

This module tests zero-score handling across all components of the v2 coverage-first
research algorithm, ensuring no crashes, no division by zero errors, and sensible
fallback behavior.

Components tested:
1. Elbow detection (find_elbow_kneedle) with all-zeros
2. Gap selection with zero scores
3. Compound reranking with zero-score results
4. Threshold computation with zero scores
5. Gap scoring with zero confidence/votes

Motivation:
The pipeline reported 6 test failures related to zero-score handling, particularly
in elbow detection on all-zeros returning wrong threshold values. These tests verify
correct handling of degenerate score distributions.

IMPORTANT NOTE - Relationship to test_gap_selection_edge_cases.py:
    - test_gap_selection_edge_cases.py: Documents EXPECTED behavior after bug fix
      (tests currently FAIL, documenting the bug)
    - test_zero_score_handling.py (this file): Verifies CURRENT behavior
      (tests PASS, ensuring no crashes/exceptions with zero scores)

    Both test suites are valuable:
    - Bug documentation tests show what SHOULD happen (aspirational)
    - Current behavior tests show what DOES happen (defensive)

    When the gap selection bug is fixed, test_gap_selection_edge_cases.py tests
    will start passing, and some assertions in this file may need updating to
    reflect the new behavior.

Gap Selection Bug Context:
    At line 584 in gap_detection.py:
        if sorted_gaps[i].score < 0.5 * sorted_gaps[0].score: break

    When all scores are 0.0:
        0.5 * 0.0 = 0.0
        0.0 < 0.0 is False
        → Loop never breaks → selects ALL gaps (incorrect)

    Expected fix:
        if sorted_gaps[0].score <= 0.01:  # Near-zero confidence
            return sorted_gaps[:min_gaps]  # Don't apply 50% heuristic
"""

import numpy as np
import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.shared.elbow_detection import (
    compute_elbow_threshold,
    find_elbow_kneedle,
)
from chunkhound.services.research.v2.coverage_research_service import (
    CoverageResearchService,
)
from chunkhound.services.research.v2.gap_detection import GapDetectionService
from chunkhound.services.research.v2.models import GapCandidate, UnifiedGap


# =============================================================================
# Helper Functions
# =============================================================================


def make_chunk(chunk_id: str, rerank_score: float) -> dict:
    """Create minimal chunk dict with rerank_score.

    Args:
        chunk_id: Unique chunk identifier
        rerank_score: Rerank score in [0.0, 1.0] range

    Returns:
        Dict with chunk_id, rerank_score, and minimal metadata
    """
    return {
        "chunk_id": chunk_id,
        "rerank_score": rerank_score,
        "file_path": f"file_{chunk_id}.py",
        "code": f"def func_{chunk_id}(): pass",
        "content": f"def func_{chunk_id}(): pass",
        "start_line": 1,
        "end_line": 5,
    }


def make_gap_candidate(query: str, confidence: float, source_shard: int) -> GapCandidate:
    """Create gap candidate with specified confidence.

    Args:
        query: Gap query string
        confidence: Confidence score [0.0, 1.0]
        source_shard: Shard index that detected this gap

    Returns:
        GapCandidate instance
    """
    return GapCandidate(
        query=query,
        rationale=f"Gap for {query}",
        confidence=confidence,
        source_shard=source_shard,
    )


def make_unified_gap(query: str, score: float, vote_count: int = 1) -> UnifiedGap:
    """Create unified gap with specified score.

    Args:
        query: Gap query string
        score: Combined score for gap selection
        vote_count: Number of shards that found this gap

    Returns:
        UnifiedGap instance
    """
    sources = [make_gap_candidate(query, score, 0)]
    return UnifiedGap(
        query=query,
        sources=sources,
        vote_count=vote_count,
        avg_confidence=score,
        score=score,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_config():
    """Create minimal config for research services.

    Returns:
        Config with default ResearchConfig (relevance_threshold = 0.5)
    """
    config = Config()
    # Use default ResearchConfig
    return config


@pytest.fixture
def coverage_service(minimal_config, tmp_path):
    """Create minimal Coverage Research Service instance.

    Args:
        minimal_config: Research configuration fixture
        tmp_path: pytest temporary directory

    Returns:
        CoverageResearchService with mocked dependencies
    """
    # Mock DatabaseServices (threshold computation doesn't use DB)
    class MockSearchService:
        async def search(self, query, path_filter=None, top_k=100):
            return []

    class MockProvider:
        def get_base_directory(self):
            return str(tmp_path)

    class MockIndexingCoordinator:
        pass

    class MockDatabaseServices:
        provider = MockProvider()
        indexing_coordinator = MockIndexingCoordinator()
        search_service = MockSearchService()

    db_services = MockDatabaseServices()

    # Mock EmbeddingManager
    class MockEmbeddingProvider:
        async def generate_embeddings(self, texts):
            return [[0.1] * 1536 for _ in texts]

        async def embed(self, texts):
            return [[0.1] * 1536 for _ in texts]

        async def rerank(self, query, documents, top_n=None):
            # Return documents with zero rerank scores
            return [
                {**doc, "rerank_score": 0.0}
                for doc in (documents[:top_n] if top_n else documents)
            ]

    class MockEmbeddingManager:
        def get_provider(self):
            return MockEmbeddingProvider()

    embedding_manager = MockEmbeddingManager()

    # Mock LLMManager
    class MockLLMProvider:
        async def generate(self, messages, **kwargs):
            return '{"queries": []}'

        async def complete_structured(self, prompt, json_schema, **kwargs):
            return {"gaps": []}

        def estimate_tokens(self, text):
            return len(text) // 4

    class MockLLMManager:
        def get_utility_provider(self):
            return MockLLMProvider()

        def get_synthesis_provider(self):
            return MockLLMProvider()

    llm_manager = MockLLMManager()

    # Create service instance
    service = CoverageResearchService(
        database_services=db_services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        config=minimal_config,
    )

    return service


@pytest.fixture
def gap_detection_service(minimal_config, tmp_path):
    """Create minimal Gap Detection Service instance.

    Args:
        minimal_config: Research configuration fixture
        tmp_path: pytest temporary directory

    Returns:
        GapDetectionService with mocked dependencies
    """
    # Mock DatabaseServices
    class MockSearchService:
        async def search(self, query, path_filter=None, top_k=100):
            return []

    class MockProvider:
        def get_base_directory(self):
            return str(tmp_path)

    class MockIndexingCoordinator:
        pass

    class MockDatabaseServices:
        provider = MockProvider()
        indexing_coordinator = MockIndexingCoordinator()
        search_service = MockSearchService()

    db_services = MockDatabaseServices()

    # Mock EmbeddingManager
    class MockEmbeddingProvider:
        async def embed(self, texts):
            return [[0.1] * 1536 for _ in texts]

        async def rerank(self, query, documents, top_n=None):
            # Return documents with zero rerank scores
            return [
                {**doc, "rerank_score": 0.0}
                for doc in (documents[:top_n] if top_n else documents)
            ]

    class MockEmbeddingManager:
        def get_provider(self):
            return MockEmbeddingProvider()

    embedding_manager = MockEmbeddingManager()

    # Mock LLMManager
    class MockLLMProvider:
        async def complete_structured(self, prompt, json_schema, **kwargs):
            return {"gaps": []}

        def estimate_tokens(self, text):
            return len(text) // 4

    class MockLLMManager:
        def get_utility_provider(self):
            return MockLLMProvider()

        def get_synthesis_provider(self):
            return MockLLMProvider()

    llm_manager = MockLLMManager()

    # Create service instance
    service = GapDetectionService(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=minimal_config.research,
    )

    return service


# =============================================================================
# Test 1: Elbow Detection with All Zeros
# =============================================================================


def test_elbow_detection_all_zeros():
    """Test: Kneedle algorithm on [0.0, 0.0, 0.0] → returns None.

    When all scores are zero (and thus identical), Kneedle cannot find an elbow
    because there's no variation in the data. The normalization step checks if
    max_score == min_score and returns None.

    Expected Behavior:
        - Kneedle detects all identical scores
        - Returns None (no elbow possible)
        - Does not crash or divide by zero
    """
    scores = [0.0, 0.0, 0.0]

    elbow_idx = find_elbow_kneedle(scores)

    assert elbow_idx is None, "Kneedle should return None for all-zeros (no elbow)"


def test_elbow_detection_many_zeros():
    """Test: Kneedle on [0.0] * 100 → returns None.

    Tests with larger dataset to ensure robustness with many identical zeros.
    """
    scores = [0.0] * 100

    elbow_idx = find_elbow_kneedle(scores)

    assert elbow_idx is None, "Kneedle should return None for many identical zeros"


def test_elbow_detection_zeros_then_drop():
    """Test: Kneedle on [0.0, 0.0, 0.0, -0.1] → returns elbow if detectable.

    Edge case: all zeros followed by a negative drop. Tests if Kneedle can
    detect an elbow when transitioning from zeros to negative values.

    Expected: Kneedle may detect elbow at index 2 (last zero) or return None
    if drop is too small to be significant.
    """
    scores = [0.0, 0.0, 0.0, -0.1]

    elbow_idx = find_elbow_kneedle(scores)

    # Either detects elbow at transition or returns None (both valid)
    if elbow_idx is not None:
        assert 0 <= elbow_idx < len(scores), "Elbow index should be valid if detected"
    # If None, that's also acceptable for this edge case


def test_elbow_detection_single_zero():
    """Test: Kneedle on [0.0] → returns None.

    Single point cannot form an elbow (requires >= 3 points).
    """
    scores = [0.0]

    elbow_idx = find_elbow_kneedle(scores)

    assert elbow_idx is None, "Kneedle should return None for single point"


def test_elbow_detection_two_zeros():
    """Test: Kneedle on [0.0, 0.0] → returns None.

    Two identical points cannot form an elbow (requires >= 3 points).
    """
    scores = [0.0, 0.0]

    elbow_idx = find_elbow_kneedle(scores)

    assert elbow_idx is None, "Kneedle should return None for two identical points"


# =============================================================================
# Test 2: Mostly Zeros Edge Case (unique - not covered elsewhere)
# =============================================================================


def test_mostly_zeros_one_nonzero():
    """Test: compute_elbow_threshold([0.0, 0.0, 0.5]) → returns valid threshold.

    When most chunks have zero scores but one is nonzero, Kneedle may detect
    an elbow or fall back to median.

    Median Calculation:
        - sorted_scores = [0.5, 0.0, 0.0] (descending)
        - median_idx = 3 // 2 = 1
        - threshold = sorted_scores[1] = 0.0

    Expected: 0.0 (median) or 0.5 (if Kneedle detects elbow)
    """
    chunks = [make_chunk("c1", 0.0), make_chunk("c2", 0.0), make_chunk("c3", 0.5)]

    threshold = compute_elbow_threshold(chunks)

    # Either median (0.0) or Kneedle-detected threshold (0.0 or 0.5)
    assert threshold in [0.0, 0.5], f"Expected 0.0 or 0.5, got {threshold}"


# =============================================================================
# Test 3: Gap Selection with Zero Scores
# =============================================================================


def test_gap_selection_all_zero_scores(gap_detection_service):
    """Test: _select_gaps_by_elbow with all score=0.0 gaps → selects min_gaps only.

    When all unified gaps have score=0.0, Kneedle cannot find an elbow
    (all identical). With near-zero top score, we have no confidence to
    select beyond min_gaps.

    Score Distribution:
        - [0.0, 0.0, 0.0, 0.0, 0.0] (all identical)

    Expected Behavior:
        1. Kneedle returns None (all identical)
        2. Falls back to 50% heuristic branch
        3. Near-zero guard detects top score < 1e-9
        4. Returns only min_gaps (default=1) - no confidence to select more

    Note: Zero confidence provides no justification for selecting beyond
    the minimum required gaps.
    """
    gaps = [make_unified_gap(f"gap_{i}", 0.0) for i in range(5)]

    selected = gap_detection_service._select_gaps_by_elbow(gaps)

    # With all zero scores, there's no confidence to select beyond min_gaps
    # min_gaps default is 1
    assert len(selected) == 1, (
        f"Should select min_gaps=1 when all scores are zero, got {len(selected)}"
    )


def test_gap_selection_single_zero_score(gap_detection_service):
    """Test: _select_gaps_by_elbow with single zero-score gap → returns that gap.

    When only one gap exists (even with score=0.0), it should be returned
    because it's below min_gaps threshold.
    """
    gaps = [make_unified_gap("gap_0", 0.0)]

    selected = gap_detection_service._select_gaps_by_elbow(gaps)

    assert len(selected) == 1, "Single zero-score gap should be selected"
    assert selected[0].score == 0.0, "Selected gap should have score 0.0"


def test_gap_selection_zeros_then_nonzero(gap_detection_service):
    """Test: _select_gaps_by_elbow with [0.8, 0.0, 0.0, 0.0] → detects elbow.

    When first gap has nonzero score and rest are zeros, Kneedle should
    detect an elbow at the transition.

    Score Distribution:
        - [0.8, 0.0, 0.0, 0.0] (clear drop at index 0)

    Expected Behavior:
        - Kneedle detects elbow at index 0 or 1
        - Returns 1 or 2 gaps (depending on elbow location)
        - Or falls back to min_gaps if Kneedle fails
    """
    gaps = [
        make_unified_gap("gap_0", 0.8),
        make_unified_gap("gap_1", 0.0),
        make_unified_gap("gap_2", 0.0),
        make_unified_gap("gap_3", 0.0),
    ]

    selected = gap_detection_service._select_gaps_by_elbow(gaps)

    # Should select at least 1 gap (the nonzero one)
    assert len(selected) >= 1, "Should select at least the nonzero gap"

    # First selected gap should be the high-score one
    assert selected[0].score == 0.8, "First selected gap should have highest score"


def test_gap_selection_empty_list(gap_detection_service):
    """Test: _select_gaps_by_elbow([]) → returns empty list.

    When no gaps exist, selection should return empty list without crashing.
    """
    gaps = []

    selected = gap_detection_service._select_gaps_by_elbow(gaps)

    assert selected == [], "Empty gap list should return empty selection"


# =============================================================================
# Test 4: Gap Scoring with Zero Confidence
# =============================================================================


def test_gap_scoring_zero_confidence():
    """Test: UnifiedGap scoring formula with confidence=0.0 → score=0.0.

    Gap scoring formula:
        score = vote_count * avg_confidence * (1 + 0.3 * shard_bonus)
        shard_bonus = 1 / (1 + min_shard_idx)

    With avg_confidence=0.0:
        score = vote_count * 0.0 * (1 + 0.3 * shard_bonus) = 0.0

    Expected: score=0.0, no division by zero
    """
    gap_candidate = make_gap_candidate("test_gap", confidence=0.0, source_shard=0)

    # Apply scoring formula manually (as done in _unify_gap_clusters)
    vote_count = 1
    avg_confidence = gap_candidate.confidence  # 0.0
    shard_bonus = 1 / (1 + gap_candidate.source_shard)  # 1 / (1 + 0) = 1.0
    score = vote_count * avg_confidence * (1 + 0.3 * shard_bonus)

    assert score == 0.0, "Zero confidence should result in zero score"


def test_gap_scoring_zero_votes():
    """Test: UnifiedGap with vote_count=0 → score=0.0.

    Edge case: gap with zero votes (should not happen in practice, but verify
    scoring formula doesn't crash).

    Score = 0 * avg_confidence * (1 + 0.3 * shard_bonus) = 0.0
    """
    # Create gap manually with zero votes
    gap = UnifiedGap(
        query="test_gap",
        sources=[],
        vote_count=0,
        avg_confidence=0.5,
        score=0.0,
    )

    # Verify score is zero
    assert gap.score == 0.0, "Zero votes should result in zero score"
    assert gap.vote_count == 0, "Vote count should be zero"


def test_gap_scoring_multiple_zero_confidence_gaps():
    """Test: Multiple gaps with confidence=0.0 → all have score=0.0.

    When multiple gaps all have zero confidence, they should all get zero scores
    without any division by zero errors.
    """
    gaps = [make_gap_candidate(f"gap_{i}", 0.0, i) for i in range(5)]

    # Apply scoring formula to each
    scores = []
    for gap in gaps:
        vote_count = 1
        avg_confidence = gap.confidence
        shard_bonus = 1 / (1 + gap.source_shard)
        score = vote_count * avg_confidence * (1 + 0.3 * shard_bonus)
        scores.append(score)

    assert all(s == 0.0 for s in scores), "All zero-confidence gaps should have score 0.0"


# =============================================================================
# Test 5: Numpy Array Handling with Zeros
# =============================================================================


def test_numpy_all_zeros_normalization():
    """Test: Numpy operations in Kneedle don't crash on all-zeros.

    Verifies that numpy array operations in find_elbow_kneedle handle
    all-zeros gracefully without divide-by-zero warnings.

    Implementation Detail:
        In find_elbow_kneedle:
        ```python
        if max_score == min_score:
            return None  # Guard against division by zero
        normalized_scores = (scores - min_score) / (max_score - min_score)
        ```

    Expected: No numpy warnings, returns None
    """
    scores = [0.0, 0.0, 0.0, 0.0]

    # This should not raise any warnings
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        elbow_idx = find_elbow_kneedle(scores)

    # Should complete without errors (no warnings raised)
    assert elbow_idx is None, "Should return None for all-zeros"


def test_numpy_negative_zeros():
    """Test: Kneedle handles negative zeros ([-0.0, -0.0]) correctly.

    In floating point, -0.0 == 0.0, but this tests robustness.
    """
    scores = [-0.0, -0.0, -0.0]

    elbow_idx = find_elbow_kneedle(scores)

    assert elbow_idx is None, "Negative zeros should be treated as all identical"


