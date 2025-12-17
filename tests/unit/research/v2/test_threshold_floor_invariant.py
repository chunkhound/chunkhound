"""Unit tests for threshold floor enforcement invariant in Gap Detection Service.

Tests that the algorithm invariant `effective_threshold >= phase1_threshold` is always
maintained when filling gaps. This ensures gap results meet or exceed the quality bar
from Phase 1 coverage.

Algorithm Spec Reference:
- docs/algorithm-coverage-first-research.md lines 346-358
- Step 2.7 in gap_detection.py: _fill_single_gap method

Key Invariant:
    effective_threshold = max(phase1_threshold, gap_threshold)

This prevents:
- Noise from low-relevance chunks that happen to match gap query
- Quality degradation in gap-filled context vs original coverage
"""

import pytest
from unittest.mock import AsyncMock, patch

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.v2.gap_detection import GapDetectionService
from chunkhound.services.research.v2.models import GapCandidate, UnifiedGap
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider


# Fixtures


@pytest.fixture
def fake_llm_provider():
    """Create fake LLM provider for gap detection."""
    return FakeLLMProvider(
        responses={
            "gap": '{"gaps": [{"query": "Test gap", "rationale": "Test", "confidence": 0.8}]}',
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


@pytest.fixture
def gap_detection_service(llm_manager, embedding_manager, db_services, research_config):
    """Create gap detection service with mocked dependencies."""
    return GapDetectionService(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=research_config,
    )


# Helper Functions


def make_chunk(chunk_id: str, rerank_score: float) -> dict:
    """Create minimal chunk dict with rerank_score.

    Args:
        chunk_id: Unique chunk identifier
        rerank_score: Rerank score in [0.0, 1.0] range

    Returns:
        Dict with chunk_id and rerank_score
    """
    return {
        "chunk_id": chunk_id,
        "file_path": f"file_{chunk_id}.py",
        "code": f"def func_{chunk_id}(): pass",
        "rerank_score": rerank_score,
        "start_line": 1,
        "end_line": 5,
    }


def make_gap(query: str, score: float = 0.8) -> UnifiedGap:
    """Create minimal UnifiedGap for testing.

    Args:
        query: Gap query string
        score: Gap score (default 0.8)

    Returns:
        UnifiedGap instance
    """
    candidate = GapCandidate(
        query=query,
        rationale="Test gap",
        confidence=0.8,
        source_shard=0,
    )
    return UnifiedGap(
        query=query,
        sources=[candidate],
        vote_count=1,
        avg_confidence=0.8,
        score=score,
    )


# Test Cases


@pytest.mark.asyncio
async def test_gap_threshold_below_phase1_uses_phase1(gap_detection_service):
    """Test: Gap elbow=0.3, phase1=0.5 → effective_threshold=0.5.

    When gap's elbow threshold is lower than phase1, the effective threshold
    should use phase1 as the floor. This ensures gap results meet the same
    quality bar as Phase 1 coverage.

    Setup:
        - phase1_threshold = 0.5
        - Mock elbow detection to return 0.3
        - Gap chunks with scores [0.7, 0.6, 0.4, 0.2]

    Expected:
        - gap_threshold = 0.3 (from mocked elbow)
        - effective_threshold = max(0.5, 0.3) = 0.5
        - Only chunks with score >= 0.5 are kept ([0.7, 0.6])
    """
    gap = make_gap("test gap")
    phase1_threshold = 0.5

    # Mock chunks returned from unified search
    mock_chunks = [
        make_chunk("c1", 0.7),
        make_chunk("c2", 0.6),
        make_chunk("c3", 0.4),
        make_chunk("c4", 0.2),
    ]

    # Mock unified search to return chunks
    with patch.object(
        gap_detection_service._unified_search,
        "unified_search",
        new_callable=AsyncMock,
        return_value=mock_chunks,
    ):
        # Mock elbow detection to return 0.3 (below phase1)
        with patch(
            "chunkhound.services.research.v2.gap_detection.compute_elbow_threshold",
            return_value=0.3,
        ):
            result = await gap_detection_service._fill_single_gap(
                root_query="root query",
                gap=gap,
                phase1_threshold=phase1_threshold,
                path_filter=None,
            )

    # Verify effective threshold was 0.5 (phase1), not 0.3 (gap elbow)
    assert len(result) == 2, "Should filter to chunks >= 0.5"
    assert all(c["rerank_score"] >= 0.5 for c in result), "All chunks should be >= 0.5"
    assert result[0]["chunk_id"] == "c1"
    assert result[1]["chunk_id"] == "c2"


@pytest.mark.asyncio
async def test_gap_threshold_equals_phase1(gap_detection_service):
    """Test: Gap elbow=0.5, phase1=0.5 → effective_threshold=0.5.

    When gap elbow equals phase1, effective threshold should be 0.5.
    This is a boundary case where both thresholds align.

    Setup:
        - phase1_threshold = 0.5
        - Mock elbow detection to return 0.5
        - Gap chunks with scores [0.8, 0.6, 0.5, 0.3]

    Expected:
        - gap_threshold = 0.5 (from mocked elbow)
        - effective_threshold = max(0.5, 0.5) = 0.5
        - Chunks with score >= 0.5 are kept ([0.8, 0.6, 0.5])
    """
    gap = make_gap("test gap")
    phase1_threshold = 0.5

    mock_chunks = [
        make_chunk("c1", 0.8),
        make_chunk("c2", 0.6),
        make_chunk("c3", 0.5),  # Exactly at threshold
        make_chunk("c4", 0.3),
    ]

    with patch.object(
        gap_detection_service._unified_search,
        "unified_search",
        new_callable=AsyncMock,
        return_value=mock_chunks,
    ):
        with patch(
            "chunkhound.services.research.v2.gap_detection.compute_elbow_threshold",
            return_value=0.5,
        ):
            result = await gap_detection_service._fill_single_gap(
                root_query="root query",
                gap=gap,
                phase1_threshold=phase1_threshold,
                path_filter=None,
            )

    # Verify chunks at or above 0.5 are kept
    assert len(result) == 3, "Should keep chunks >= 0.5 (inclusive)"
    assert all(c["rerank_score"] >= 0.5 for c in result)
    assert result[0]["chunk_id"] == "c1"
    assert result[1]["chunk_id"] == "c2"
    assert result[2]["chunk_id"] == "c3"


@pytest.mark.asyncio
async def test_gap_threshold_above_phase1_uses_gap(gap_detection_service):
    """Test: Gap elbow=0.7, phase1=0.5 → effective_threshold=0.7.

    When gap's elbow threshold is higher than phase1, the effective threshold
    should use the gap threshold. This allows gap fills to be MORE selective
    than Phase 1 when the gap results have a clear quality cutoff.

    Setup:
        - phase1_threshold = 0.5
        - Mock elbow detection to return 0.7
        - Gap chunks with scores [0.9, 0.8, 0.6, 0.4]

    Expected:
        - gap_threshold = 0.7 (from mocked elbow)
        - effective_threshold = max(0.5, 0.7) = 0.7
        - Only chunks with score >= 0.7 are kept ([0.9, 0.8])
    """
    gap = make_gap("test gap")
    phase1_threshold = 0.5

    mock_chunks = [
        make_chunk("c1", 0.9),
        make_chunk("c2", 0.8),
        make_chunk("c3", 0.6),
        make_chunk("c4", 0.4),
    ]

    with patch.object(
        gap_detection_service._unified_search,
        "unified_search",
        new_callable=AsyncMock,
        return_value=mock_chunks,
    ):
        with patch(
            "chunkhound.services.research.v2.gap_detection.compute_elbow_threshold",
            return_value=0.7,
        ):
            result = await gap_detection_service._fill_single_gap(
                root_query="root query",
                gap=gap,
                phase1_threshold=phase1_threshold,
                path_filter=None,
            )

    # Verify effective threshold was 0.7 (gap), not 0.5 (phase1)
    assert len(result) == 2, "Should filter to chunks >= 0.7"
    assert all(c["rerank_score"] >= 0.7 for c in result)
    assert result[0]["chunk_id"] == "c1"
    assert result[1]["chunk_id"] == "c2"


@pytest.mark.asyncio
async def test_threshold_floor_across_multiple_gaps(gap_detection_service):
    """Test: Different gaps have different elbows, each uses max(gap_elbow, phase1).

    This tests that the threshold floor invariant is applied independently to
    each gap fill. Different gaps may have different elbow thresholds, but all
    must respect the phase1 floor.

    Setup:
        - phase1_threshold = 0.5
        - Gap 1: elbow=0.3, chunks=[0.9, 0.6, 0.4]
        - Gap 2: elbow=0.7, chunks=[0.95, 0.8, 0.6]
        - Gap 3: elbow=0.5, chunks=[0.85, 0.5, 0.3]

    Expected:
        - Gap 1: effective=max(0.5, 0.3)=0.5 → keeps [0.9, 0.6]
        - Gap 2: effective=max(0.5, 0.7)=0.7 → keeps [0.95, 0.8]
        - Gap 3: effective=max(0.5, 0.5)=0.5 → keeps [0.85, 0.5]
    """
    phase1_threshold = 0.5

    # Define gap scenarios
    gap_scenarios = [
        {
            "gap": make_gap("gap1"),
            "chunks": [
                make_chunk("g1c1", 0.9),
                make_chunk("g1c2", 0.6),
                make_chunk("g1c3", 0.4),
            ],
            "elbow": 0.3,
            "expected_count": 2,  # 0.9, 0.6 (>= 0.5)
        },
        {
            "gap": make_gap("gap2"),
            "chunks": [
                make_chunk("g2c1", 0.95),
                make_chunk("g2c2", 0.8),
                make_chunk("g2c3", 0.6),
            ],
            "elbow": 0.7,
            "expected_count": 2,  # 0.95, 0.8 (>= 0.7)
        },
        {
            "gap": make_gap("gap3"),
            "chunks": [
                make_chunk("g3c1", 0.85),
                make_chunk("g3c2", 0.5),
                make_chunk("g3c3", 0.3),
            ],
            "elbow": 0.5,
            "expected_count": 2,  # 0.85, 0.5 (>= 0.5)
        },
    ]

    # Test each gap independently
    for scenario in gap_scenarios:
        with patch.object(
            gap_detection_service._unified_search,
            "unified_search",
            new_callable=AsyncMock,
            return_value=scenario["chunks"],
        ):
            with patch(
                "chunkhound.services.research.v2.gap_detection.compute_elbow_threshold",
                return_value=scenario["elbow"],
            ):
                result = await gap_detection_service._fill_single_gap(
                    root_query="root query",
                    gap=scenario["gap"],
                    phase1_threshold=phase1_threshold,
                    path_filter=None,
                )

        # Verify threshold floor was respected
        effective_threshold = max(phase1_threshold, scenario["elbow"])
        assert len(result) == scenario["expected_count"], (
            f"Gap with elbow={scenario['elbow']} should have "
            f"{scenario['expected_count']} chunks (threshold={effective_threshold})"
        )
        assert all(c["rerank_score"] >= effective_threshold for c in result), (
            f"All chunks should be >= {effective_threshold}"
        )


@pytest.mark.asyncio
async def test_empty_chunks_respects_phase1_floor(gap_detection_service):
    """Test: When unified search returns no chunks, no error occurs.

    Edge case: If unified search returns empty results, the method should
    handle it gracefully without crashing.

    Setup:
        - phase1_threshold = 0.5
        - Unified search returns []

    Expected:
        - No elbow computation (guard clause)
        - Returns empty list
        - No crash
    """
    gap = make_gap("test gap")
    phase1_threshold = 0.5

    with patch.object(
        gap_detection_service._unified_search,
        "unified_search",
        new_callable=AsyncMock,
        return_value=[],
    ):
        result = await gap_detection_service._fill_single_gap(
            root_query="root query",
            gap=gap,
            phase1_threshold=phase1_threshold,
            path_filter=None,
        )

    assert result == [], "Should return empty list when no chunks found"


@pytest.mark.asyncio
async def test_chunks_missing_rerank_score_treated_as_zero(gap_detection_service):
    """Test: Chunks missing rerank_score field are treated as 0.0 and filtered.

    Robustness test: Ensures chunks without rerank_score don't crash the
    threshold filtering logic, but are correctly treated as 0.0.

    Setup:
        - phase1_threshold = 0.5
        - Mock elbow to return 0.5
        - Chunks with scores [0.8, missing, 0.6, missing]

    Expected:
        - Missing scores treated as 0.0
        - Only chunks >= 0.5 kept ([0.8, 0.6])
    """
    gap = make_gap("test gap")
    phase1_threshold = 0.5

    mock_chunks = [
        make_chunk("c1", 0.8),
        {"chunk_id": "c2", "code": "no score", "file_path": "file2.py"},
        make_chunk("c3", 0.6),
        {"chunk_id": "c4", "code": "also no score", "file_path": "file4.py"},
    ]

    with patch.object(
        gap_detection_service._unified_search,
        "unified_search",
        new_callable=AsyncMock,
        return_value=mock_chunks,
    ):
        with patch(
            "chunkhound.services.research.v2.gap_detection.compute_elbow_threshold",
            return_value=0.5,
        ):
            result = await gap_detection_service._fill_single_gap(
                root_query="root query",
                gap=gap,
                phase1_threshold=phase1_threshold,
                path_filter=None,
            )

    # Only chunks with score >= 0.5 should remain (c1, c3)
    assert len(result) == 2
    assert result[0]["chunk_id"] == "c1"
    assert result[1]["chunk_id"] == "c3"


@pytest.mark.asyncio
async def test_extreme_phase1_threshold_filters_all_chunks(gap_detection_service):
    """Test: Very high phase1_threshold filters out all gap chunks.

    Edge case: If phase1_threshold is very high (e.g., 0.95), even high-scoring
    gap chunks may be filtered out.

    Setup:
        - phase1_threshold = 0.95
        - Mock elbow to return 0.7
        - Gap chunks with scores [0.9, 0.8, 0.7]

    Expected:
        - effective_threshold = max(0.95, 0.7) = 0.95
        - No chunks pass threshold
        - Returns empty list
    """
    gap = make_gap("test gap")
    phase1_threshold = 0.95

    mock_chunks = [
        make_chunk("c1", 0.9),
        make_chunk("c2", 0.8),
        make_chunk("c3", 0.7),
    ]

    with patch.object(
        gap_detection_service._unified_search,
        "unified_search",
        new_callable=AsyncMock,
        return_value=mock_chunks,
    ):
        with patch(
            "chunkhound.services.research.v2.gap_detection.compute_elbow_threshold",
            return_value=0.7,
        ):
            result = await gap_detection_service._fill_single_gap(
                root_query="root query",
                gap=gap,
                phase1_threshold=phase1_threshold,
                path_filter=None,
            )

    assert result == [], "Should filter all chunks when phase1_threshold very high"


@pytest.mark.asyncio
async def test_very_low_phase1_threshold_uses_gap_elbow(gap_detection_service):
    """Test: Very low phase1_threshold allows gap elbow to dominate.

    Edge case: If phase1_threshold is very low (e.g., 0.1), the gap's elbow
    threshold should dominate and provide meaningful filtering.

    Setup:
        - phase1_threshold = 0.1
        - Mock elbow to return 0.6
        - Gap chunks with scores [0.9, 0.7, 0.5, 0.3]

    Expected:
        - effective_threshold = max(0.1, 0.6) = 0.6
        - Only chunks >= 0.6 kept ([0.9, 0.7])
    """
    gap = make_gap("test gap")
    phase1_threshold = 0.1

    mock_chunks = [
        make_chunk("c1", 0.9),
        make_chunk("c2", 0.7),
        make_chunk("c3", 0.5),
        make_chunk("c4", 0.3),
    ]

    with patch.object(
        gap_detection_service._unified_search,
        "unified_search",
        new_callable=AsyncMock,
        return_value=mock_chunks,
    ):
        with patch(
            "chunkhound.services.research.v2.gap_detection.compute_elbow_threshold",
            return_value=0.6,
        ):
            result = await gap_detection_service._fill_single_gap(
                root_query="root query",
                gap=gap,
                phase1_threshold=phase1_threshold,
                path_filter=None,
            )

    assert len(result) == 2, "Should use gap elbow (0.6) not phase1 (0.1)"
    assert all(c["rerank_score"] >= 0.6 for c in result)
    assert result[0]["chunk_id"] == "c1"
    assert result[1]["chunk_id"] == "c2"


@pytest.mark.asyncio
async def test_invariant_holds_with_elbow_fallback_to_median(gap_detection_service):
    """Test: Threshold floor invariant holds even when elbow uses median fallback.

    When Kneedle algorithm returns None (no clear elbow), compute_elbow_threshold
    falls back to median. The threshold floor invariant should still hold.

    Setup:
        - phase1_threshold = 0.5
        - Perfectly linear scores → Kneedle returns None → median=0.6
        - Gap chunks with scores [1.0, 0.8, 0.6, 0.4, 0.2]

    Expected:
        - gap_threshold = 0.6 (median fallback)
        - effective_threshold = max(0.5, 0.6) = 0.6
        - Only chunks >= 0.6 kept
    """
    gap = make_gap("test gap")
    phase1_threshold = 0.5

    # Perfectly linear distribution triggers median fallback
    mock_chunks = [
        make_chunk("c1", 1.0),
        make_chunk("c2", 0.8),
        make_chunk("c3", 0.6),
        make_chunk("c4", 0.4),
        make_chunk("c5", 0.2),
    ]

    with patch.object(
        gap_detection_service._unified_search,
        "unified_search",
        new_callable=AsyncMock,
        return_value=mock_chunks,
    ):
        # Don't mock compute_elbow_threshold - let it use real logic
        # This will trigger Kneedle → None → median fallback
        result = await gap_detection_service._fill_single_gap(
            root_query="root query",
            gap=gap,
            phase1_threshold=phase1_threshold,
            path_filter=None,
        )

    # With perfectly linear scores [1.0, 0.8, 0.6, 0.4, 0.2]:
    # median_idx = 5 // 2 = 2 → sorted_scores[2] = 0.6
    # effective_threshold = max(0.5, 0.6) = 0.6
    assert len(result) == 3, "Should keep chunks >= 0.6"
    assert all(c["rerank_score"] >= 0.6 for c in result)
    assert result[0]["chunk_id"] == "c1"
    assert result[1]["chunk_id"] == "c2"
    assert result[2]["chunk_id"] == "c3"
