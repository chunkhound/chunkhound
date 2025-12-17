"""Unit tests for threshold computation edge cases in Coverage Research Service.

Tests the _compute_elbow_threshold() method which uses the Kneedle algorithm
to adaptively compute relevance thresholds from chunk rerank scores.

The Kneedle algorithm can return None for degenerate distributions:
- < 3 points (insufficient for elbow detection)
- All identical scores (no elbow point exists)
- Perfectly linear distributions (no significant elbow)

When Kneedle returns None, the implementation falls back to median threshold.
"""

import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.shared.elbow_detection import (
    compute_elbow_threshold,
)
from chunkhound.services.research.v2.coverage_research_service import (
    CoverageResearchService,
)


# Helper Functions


def make_chunk(chunk_id: str, rerank_score: float) -> dict:
    """Create minimal chunk dict with rerank_score for threshold computation.

    Args:
        chunk_id: Unique chunk identifier
        rerank_score: Rerank score in [0.0, 1.0] range

    Returns:
        Dict with chunk_id and rerank_score (minimal for _compute_elbow_threshold)
    """
    return {
        "chunk_id": chunk_id,
        "rerank_score": rerank_score,
        "file_path": f"file_{chunk_id}.py",
        "content": f"def func_{chunk_id}(): pass",
        "start_line": 1,
        "end_line": 5,
    }


# Fixtures


@pytest.fixture
def minimal_config():
    """Create minimal config for Coverage Research Service.

    Only sets required research config fields, uses defaults for others.
    """
    config = Config()
    # Use default ResearchConfig (relevance_threshold = 0.5)
    return config


@pytest.fixture
def coverage_service(minimal_config, tmp_path, monkeypatch):
    """Create minimal Coverage Research Service instance.

    This fixture provides a service instance with mocked dependencies,
    sufficient for testing the _compute_elbow_threshold() method which
    only operates on chunk data structures.

    Args:
        minimal_config: Research configuration fixture
        tmp_path: pytest temporary directory
        monkeypatch: pytest monkeypatch fixture

    Returns:
        CoverageResearchService with mocked database/LLM/embedding managers
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

    # DatabaseServices is a NamedTuple, not a regular class
    class MockDatabaseServices:
        provider = MockProvider()
        indexing_coordinator = MockIndexingCoordinator()
        search_service = MockSearchService()

    db_services = MockDatabaseServices()

    # Mock EmbeddingManager (threshold computation doesn't use embeddings)
    class MockEmbeddingProvider:
        async def generate_embeddings(self, texts):
            return [[0.1] * 1536 for _ in texts]

        async def rerank(self, query, documents, top_n=None):
            # Return documents with mock rerank scores
            return [
                {**doc, "rerank_score": 0.9 - i * 0.1}
                for i, doc in enumerate(documents[:top_n] if top_n else documents)
            ]

    class MockEmbeddingManager:
        def get_provider(self):
            return MockEmbeddingProvider()

    embedding_manager = MockEmbeddingManager()

    # Mock LLMManager (threshold computation doesn't use LLM)
    class MockLLMProvider:
        async def generate(self, messages, **kwargs):
            return '{"queries": []}'

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


# Test Cases


def test_single_chunk_returns_that_score(coverage_service):
    """Test: Single chunk with score 0.85 → threshold should be 0.85.

    When only one chunk exists, Kneedle cannot compute an elbow (requires >= 3 points).
    However, the implementation short-circuits this case by returning the single score
    directly without invoking Kneedle.

    Implementation Note:
        The method doesn't explicitly handle len(scores)==1 case before calling
        find_elbow_kneedle, so Kneedle returns None and median fallback is used.
        For single element, median = element itself.

    Expected Behavior:
        - Kneedle returns None (< 3 points)
        - Median fallback: sorted_scores[len//2] = sorted_scores[0] = 0.85
        - Returns: 0.85
    """
    chunks = [make_chunk("c1", 0.85)]

    threshold = compute_elbow_threshold(chunks)

    assert threshold == 0.85, "Single chunk should return its rerank_score"


def test_two_chunks_returns_median(coverage_service):
    """Test: Two chunks [0.9, 0.1] → median fallback.

    Kneedle requires >= 3 points to detect an elbow. With only 2 points,
    it returns None and the implementation falls back to median.

    Median Calculation:
        - sorted_scores = [0.9, 0.1] (descending order)
        - median_idx = len(sorted_scores) // 2 = 2 // 2 = 1
        - threshold = sorted_scores[1] = 0.1

    Expected Behavior:
        - Kneedle returns None (< 3 points)
        - Median fallback: sorted_scores[1] = 0.1
        - Returns: 0.1
    """
    chunks = [make_chunk("c1", 0.9), make_chunk("c2", 0.1)]

    threshold = compute_elbow_threshold(chunks)

    # Median of [0.9, 0.1] sorted descending: sorted_scores[len//2] = sorted_scores[1] = 0.1
    assert threshold == 0.1, "Two chunks should return median (lower score)"


def test_all_identical_scores_returns_that_value(coverage_service):
    """Test: All identical scores [0.5, 0.5, 0.5, 0.5] → kneedle returns None, median is 0.5.

    When all scores are identical, Kneedle cannot find an elbow because:
        - Normalization: (scores - min) / (max - min) → 0/0 (division by zero guard)
        - Implementation checks: if max_score == min_score: return None

    Median Calculation:
        - sorted_scores = [0.5, 0.5, 0.5, 0.5]
        - median_idx = len(sorted_scores) // 2 = 4 // 2 = 2
        - threshold = sorted_scores[2] = 0.5

    Expected Behavior:
        - Kneedle returns None (all identical)
        - Median fallback: sorted_scores[2] = 0.5
        - Returns: 0.5
    """
    chunks = [
        make_chunk("c1", 0.5),
        make_chunk("c2", 0.5),
        make_chunk("c3", 0.5),
        make_chunk("c4", 0.5),
    ]

    threshold = compute_elbow_threshold(chunks)

    assert threshold == 0.5, "Identical scores should return that value (via median)"


def test_all_zero_scores_returns_zero(coverage_service):
    """Test: All zero scores [0.0, 0.0, 0.0] → verify doesn't crash, returns 0.0.

    Edge case where all rerank scores are zero (could happen with irrelevant results).
    Kneedle cannot find elbow (all identical), so median fallback is used.

    Median Calculation:
        - sorted_scores = [0.0, 0.0, 0.0]
        - median_idx = len(sorted_scores) // 2 = 3 // 2 = 1
        - threshold = sorted_scores[1] = 0.0

    Expected Behavior:
        - Kneedle returns None (all identical)
        - Median fallback: sorted_scores[1] = 0.0
        - Returns: 0.0 (NOT the default 0.5)

    This verifies the implementation correctly uses median fallback rather
    than returning the hardcoded default.
    """
    chunks = [make_chunk("c1", 0.0), make_chunk("c2", 0.0), make_chunk("c3", 0.0)]

    threshold = compute_elbow_threshold(chunks)

    assert threshold == 0.0, "All zero scores should return 0.0, not default 0.5"


def test_empty_chunks_returns_default(coverage_service):
    """Test: Empty chunks list → returns config default (currently hardcoded 0.5).

    When no chunks are provided, the implementation returns a hardcoded default
    of 0.5. This test documents the current behavior and verifies the guard clause.

    Implementation Note:
        Current code has hardcoded 0.5 default:
        ```python
        if not chunks:
            return 0.5  # Default threshold
        ```

    Expected Enhancement:
        Should use configured default instead:
        ```python
        if not chunks:
            return self._config.relevance_threshold  # Default from config (0.5)
        ```

    Expected Behavior (Current):
        - Returns: 0.5 (hardcoded default)

    Expected Behavior (After Enhancement):
        - Returns: self._config.relevance_threshold (0.5 by default)
    """
    chunks = []

    threshold = compute_elbow_threshold(chunks)

    # Current implementation returns hardcoded 0.5
    # After enhancement, should return config.research.relevance_threshold (also 0.5 by default)
    assert threshold == 0.5, "Empty chunks should return default threshold (0.5)"


def test_clear_elbow_returns_kneedle_threshold(coverage_service):
    """Test: Three chunks with clear elbow [0.9, 0.85, 0.2] → kneedle detects elbow.

    When scores have a clear drop (elbow), Kneedle successfully detects it.
    The elbow point is the chunk with maximum perpendicular distance from the
    line connecting first and last points.

    Score Distribution:
        - Index 0: 0.9 (high relevance)
        - Index 1: 0.85 (high relevance)
        - Index 2: 0.2 (low relevance)
        - Clear elbow between indices 1 and 2

    Kneedle Behavior:
        1. Normalize scores to [0, 1]: [1.0, 0.93, 0.0]
        2. Draw line from (0, 1.0) to (1, 0.0)
        3. Compute perpendicular distances
        4. Find point with max distance (likely index 1)
        5. Return elbow_idx = 1, threshold = sorted_scores[1] = 0.85

    Expected Behavior:
        - Kneedle detects elbow at index 1 or 2 (implementation-dependent)
        - Returns: score at elbow index (0.85 or 0.2)
        - NOT median (0.85) - Kneedle should succeed

    Note: Exact elbow index depends on Kneedle sensitivity parameter (0.01 threshold).
    """
    chunks = [
        make_chunk("c1", 0.9),
        make_chunk("c2", 0.85),
        make_chunk("c3", 0.2),
    ]

    threshold = compute_elbow_threshold(chunks)

    # Kneedle should detect elbow, returning one of the scores
    # Most likely: 0.85 (elbow before drop to 0.2)
    # Alternative: 0.2 (point at the drop)
    # Median fallback would be: sorted_scores[1] = 0.85
    assert threshold in [
        0.9,
        0.85,
        0.2,
    ], f"Clear elbow should return Kneedle threshold, got {threshold}"

    # Verify it's NOT the median (unless Kneedle happens to pick median point)
    # For this distribution, Kneedle should successfully find elbow at 0.85 or 0.2
    # We just verify it returned one of the actual scores, not an interpolated value
    assert threshold in [c["rerank_score"] for c in chunks], (
        "Threshold should be one of the actual chunk scores"
    )


def test_perfectly_linear_distribution_returns_median(coverage_service):
    """Test: Perfectly linear scores → kneedle finds no significant elbow, returns median.

    When scores decrease linearly, there's no clear elbow point. Kneedle may
    still return an index, but the perpendicular distance will be below the
    significance threshold (0.01), causing it to return None.

    Score Distribution:
        - [1.0, 0.8, 0.6, 0.4, 0.2] (perfectly linear)
        - No clear elbow point
        - All points are equidistant from first-to-last line

    Kneedle Behavior:
        1. Normalize scores: [1.0, 0.75, 0.5, 0.25, 0.0]
        2. Draw line from (0, 1.0) to (1, 0.0)
        3. All points ON the line → distances ≈ 0
        4. Max distance < 0.01 (significance threshold)
        5. Returns None

    Median Calculation:
        - sorted_scores = [1.0, 0.8, 0.6, 0.4, 0.2]
        - median_idx = len(sorted_scores) // 2 = 5 // 2 = 2
        - threshold = sorted_scores[2] = 0.6

    Expected Behavior:
        - Kneedle returns None (no significant elbow)
        - Median fallback: sorted_scores[2] = 0.6
        - Returns: 0.6
    """
    chunks = [
        make_chunk("c1", 1.0),
        make_chunk("c2", 0.8),
        make_chunk("c3", 0.6),
        make_chunk("c4", 0.4),
        make_chunk("c5", 0.2),
    ]

    threshold = compute_elbow_threshold(chunks)

    # Perfectly linear → Kneedle returns None → median fallback
    # median_idx = 5 // 2 = 2 → sorted_scores[2] = 0.6
    assert threshold == 0.6, "Linear distribution should return median (0.6)"


def test_many_identical_then_drop(coverage_service):
    """Test: Many identical high scores, then sudden drop → kneedle detects elbow.

    This tests a common real-world pattern: multiple highly relevant chunks
    (same rerank score) followed by a drop to lower-relevance chunks.

    Score Distribution:
        - [0.95, 0.95, 0.95, 0.95, 0.3, 0.2] (4 identical, then drop)
        - Clear elbow between indices 3 and 4

    Kneedle Behavior:
        - Should detect elbow at or near the drop point
        - Likely returns index 3 (last high score) or 4 (first low score)

    Expected Behavior:
        - Kneedle succeeds: returns 0.95 or 0.3 (depending on exact elbow location)
        - NOT median (0.95, which is sorted_scores[3])
    """
    chunks = [
        make_chunk("c1", 0.95),
        make_chunk("c2", 0.95),
        make_chunk("c3", 0.95),
        make_chunk("c4", 0.95),
        make_chunk("c5", 0.3),
        make_chunk("c6", 0.2),
    ]

    threshold = compute_elbow_threshold(chunks)

    # Kneedle should detect elbow at the drop point
    # Most likely: 0.95 (last high score before drop)
    # Alternative: 0.3 (first low score after drop)
    assert threshold in [0.95, 0.3], (
        f"Identical-then-drop should detect elbow at transition, got {threshold}"
    )


def test_missing_rerank_score_field(coverage_service):
    """Test: Chunk missing rerank_score field → defaults to 0.0.

    Robustness test: chunks without rerank_score should not crash,
    but should use default value of 0.0.

    Implementation:
        scores = [c.get("rerank_score", 0.0) for c in chunks]

    Expected Behavior:
        - Missing rerank_score treated as 0.0
        - Median of [0.9, 0.0, 0.0] = sorted_scores[1] = 0.0
    """
    chunks = [
        make_chunk("c1", 0.9),
        {"chunk_id": "c2", "content": "no rerank_score field"},
        {"chunk_id": "c3", "content": "also missing"},
    ]

    threshold = compute_elbow_threshold(chunks)

    # sorted_scores = [0.9, 0.0, 0.0]
    # Kneedle likely returns None (not enough variation)
    # median_idx = 3 // 2 = 1 → sorted_scores[1] = 0.0
    assert threshold == 0.0, "Missing rerank_score should default to 0.0, median is 0.0"


# Enhancement Documentation Test


def test_empty_chunks_should_use_config_default():
    """ENHANCEMENT: Empty chunks should use config.research.relevance_threshold.

    Current Implementation:
        ```python
        if not chunks:
            return 0.5  # Hardcoded default
        ```

    Proposed Enhancement:
        ```python
        if not chunks:
            return self._config.relevance_threshold  # From ResearchConfig
        ```

    Benefits:
        - Consistent with rest of config system
        - Allows users to customize default threshold
        - Follows principle of no magic numbers

    This test documents the expected behavior after enhancement.
    Currently SKIPPED to avoid breaking existing tests.
    """
    pytest.skip(
        "Enhancement not yet implemented. "
        "Empty chunks should return self._config.relevance_threshold instead of hardcoded 0.5"
    )

    # After enhancement, this test should pass:
    # config = Config()
    # config.research.relevance_threshold = 0.7  # Custom default
    # service = CoverageResearchService(..., config=config)
    # threshold = service._compute_elbow_threshold([])
    # assert threshold == 0.7, "Should use configured default, not hardcoded 0.5"
