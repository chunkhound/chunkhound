"""Unit tests for v2 Gap Detection Service - Async Failure Handling in Parallel Gap Fills.

Tests edge cases where individual gap fills fail asynchronously during parallel execution.
Verifies that asyncio.gather() with return_exceptions=True properly handles failures while
preserving successful results.

Implementation Details:
- _fill_gaps_parallel uses asyncio.gather(*tasks, return_exceptions=True) at line 664
- Exceptions are caught per-task and converted to empty lists
- Other gap fills continue unaffected
- No shared mutable state between gap fills ensures isolation
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.v2.gap_detection import GapDetectionService
from chunkhound.services.research.v2.models import UnifiedGap, GapCandidate
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider


@pytest.fixture
def fake_llm_provider():
    """Create fake LLM provider for gap detection."""
    return FakeLLMProvider(
        responses={
            "gap": '{"gaps": [{"query": "test gap", "rationale": "test", "confidence": 0.85}]}',
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


@pytest.fixture
def covered_chunks():
    """Create sample covered chunks from Phase 1."""
    return [
        {
            "chunk_id": "c1",
            "code": "def foo(): pass",
            "file_path": "test.py",
            "rerank_score": 0.9,
        },
        {
            "chunk_id": "c2",
            "code": "def bar(): pass",
            "file_path": "test.py",
            "rerank_score": 0.8,
        },
    ]


class TestSingleGapFailurePreservesOthers:
    """Test that when one gap fill raises an exception, other gap results are preserved.

    Verifies the critical isolation property: each gap fill is independent and failures
    don't cascade. Uses asyncio.gather(return_exceptions=True) to catch exceptions.
    """

    @pytest.mark.asyncio
    async def test_single_gap_fill_exception_preserves_others(
        self, gap_detection_service, covered_chunks
    ):
        """Should preserve successful gap results when one gap fill raises an exception.

        Scenario: 3 gaps selected
        - gap1 succeeds with 3 chunks
        - gap2 raises ValueError (simulated failure)
        - gap3 succeeds with 2 chunks

        Expected behavior:
        - gap1 and gap3 results preserved
        - gap2 logged as warning, returns empty list
        - gaps_filled = 2 (only successful gaps)
        - chunks_added = 5 (3 + 2 from successful gaps)

        Implementation check:
        - asyncio.gather() uses return_exceptions=True (line 664)
        - Exceptions are caught and converted to empty lists (lines 669-673)
        """
        # Mock _fill_single_gap to selectively fail
        original_fill_single_gap = gap_detection_service._fill_single_gap
        call_count = 0

        async def mock_fill_single_gap(root_query, gap, phase1_threshold, path_filter):
            nonlocal call_count
            call_count += 1

            # Gap 2 (call_count == 2) raises exception
            if call_count == 2:
                raise ValueError("Simulated unified search failure")

            # Gap 1 and Gap 3 succeed with different chunk counts
            if call_count == 1:
                return [
                    {"chunk_id": "g1_c1", "rerank_score": 0.85, "code": "chunk1"},
                    {"chunk_id": "g1_c2", "rerank_score": 0.80, "code": "chunk2"},
                    {"chunk_id": "g1_c3", "rerank_score": 0.75, "code": "chunk3"},
                ]
            else:  # call_count == 3
                return [
                    {"chunk_id": "g3_c1", "rerank_score": 0.78, "code": "chunk4"},
                    {"chunk_id": "g3_c2", "rerank_score": 0.72, "code": "chunk5"},
                ]

        # Create unified gaps
        unified_gaps = [
            UnifiedGap(
                "cache implementation?",
                [GapCandidate("cache", "need cache", 0.9, 0)],
                1,
                0.9,
                score=0.9,
            ),
            UnifiedGap(
                "error handling?",
                [GapCandidate("errors", "need errors", 0.8, 0)],
                1,
                0.8,
                score=0.8,
            ),
            UnifiedGap(
                "logging?",
                [GapCandidate("logs", "need logs", 0.7, 0)],
                1,
                0.7,
                score=0.7,
            ),
        ]

        with patch.object(
            gap_detection_service,
            "_fill_single_gap",
            side_effect=mock_fill_single_gap,
        ):
            with patch.object(
                gap_detection_service,
                "_cluster_chunks_hdbscan",
                new_callable=AsyncMock,
                return_value=[covered_chunks],
            ):
                with patch.object(
                    gap_detection_service,
                    "_select_gaps_by_elbow",
                    return_value=unified_gaps,
                ):
                    all_chunks, gap_stats = (
                        await gap_detection_service.detect_and_fill_gaps(
                            root_query="How does the system work?",
                            covered_chunks=covered_chunks,
                            phase1_threshold=0.6,
                        )
                    )

        # Verify _fill_single_gap was called 3 times
        assert call_count == 3

        # Verify stats reflect only successful gap fills
        assert gap_stats["gaps_selected"] == 3
        assert gap_stats["gaps_filled"] == 2  # Only gap1 and gap3 succeeded
        assert gap_stats["chunks_added"] == 5  # 3 + 2

        # Verify merged chunks (2 coverage + 5 gap = 7 total)
        assert len(all_chunks) == 7

        # Verify all gap chunks are present
        gap_chunk_ids = {c["chunk_id"] for c in all_chunks if c["chunk_id"].startswith("g")}
        assert len(gap_chunk_ids) == 5
        assert gap_chunk_ids == {"g1_c1", "g1_c2", "g1_c3", "g3_c1", "g3_c2"}


class TestAllGapFailuresGraceful:
    """Test graceful handling when all gap fills fail.

    Verifies that complete failure doesn't crash the system and returns
    original coverage chunks intact.
    """

    @pytest.mark.asyncio
    async def test_all_gap_fills_fail_gracefully(
        self, gap_detection_service, covered_chunks
    ):
        """Should return original coverage when all gap fills fail.

        Scenario: 3 gaps selected, all raise different exceptions

        Expected behavior:
        - No crashes, graceful degradation
        - gaps_filled = 0
        - chunks_added = 0
        - Returns original covered_chunks unmodified
        - All exceptions logged as warnings
        """
        # Mock _fill_single_gap to always fail with different exceptions
        call_count = 0
        exceptions = [
            RuntimeError("Database connection lost"),
            ValueError("Invalid search query format"),
            asyncio.TimeoutError("Search timeout exceeded"),
        ]

        async def mock_fill_single_gap(root_query, gap, phase1_threshold, path_filter):
            nonlocal call_count
            exc = exceptions[call_count]
            call_count += 1
            raise exc

        unified_gaps = [
            UnifiedGap("query1", [], 1, 0.9, score=0.9),
            UnifiedGap("query2", [], 1, 0.8, score=0.8),
            UnifiedGap("query3", [], 1, 0.7, score=0.7),
        ]

        with patch.object(
            gap_detection_service,
            "_fill_single_gap",
            side_effect=mock_fill_single_gap,
        ):
            with patch.object(
                gap_detection_service,
                "_cluster_chunks_hdbscan",
                new_callable=AsyncMock,
                return_value=[covered_chunks],
            ):
                with patch.object(
                    gap_detection_service,
                    "_select_gaps_by_elbow",
                    return_value=unified_gaps,
                ):
                    all_chunks, gap_stats = (
                        await gap_detection_service.detect_and_fill_gaps(
                            root_query="test query",
                            covered_chunks=covered_chunks,
                            phase1_threshold=0.6,
                        )
                    )

        # Verify all gaps attempted
        assert call_count == 3

        # Verify stats reflect complete failure
        assert gap_stats["gaps_selected"] == 3
        assert gap_stats["gaps_filled"] == 0  # All failed
        assert gap_stats["chunks_added"] == 0  # No chunks added

        # Verify original coverage unchanged
        assert len(all_chunks) == len(covered_chunks)
        assert all_chunks == covered_chunks


class TestAsyncTimeoutFailures:
    """Test handling of asyncio.TimeoutError in individual gap fills.

    Verifies that timeout in one gap doesn't affect others and is handled
    as a normal exception.
    """

    @pytest.mark.asyncio
    async def test_timeout_in_one_gap_fill(self, gap_detection_service, covered_chunks):
        """Should handle asyncio.TimeoutError in one gap while others complete.

        Scenario: 3 gaps where gap2 times out

        Expected behavior:
        - Timeout caught by return_exceptions=True
        - Converted to empty list for that gap
        - Other gaps complete successfully
        - gaps_filled = 2 (excluding timed-out gap)
        """
        call_count = 0

        async def mock_fill_single_gap(root_query, gap, phase1_threshold, path_filter):
            nonlocal call_count
            call_count += 1

            if call_count == 2:
                # Simulate timeout
                raise asyncio.TimeoutError("Gap fill search timed out")

            # Other gaps succeed
            return [
                {"chunk_id": f"g{call_count}_c1", "rerank_score": 0.8, "code": f"chunk{call_count}"}
            ]

        unified_gaps = [
            UnifiedGap("fast query", [], 1, 0.9, score=0.9),
            UnifiedGap("slow query", [], 1, 0.8, score=0.8),  # Times out
            UnifiedGap("medium query", [], 1, 0.7, score=0.7),
        ]

        with patch.object(
            gap_detection_service,
            "_fill_single_gap",
            side_effect=mock_fill_single_gap,
        ):
            with patch.object(
                gap_detection_service,
                "_cluster_chunks_hdbscan",
                new_callable=AsyncMock,
                return_value=[covered_chunks],
            ):
                with patch.object(
                    gap_detection_service,
                    "_select_gaps_by_elbow",
                    return_value=unified_gaps,
                ):
                    all_chunks, gap_stats = (
                        await gap_detection_service.detect_and_fill_gaps(
                            root_query="test query",
                            covered_chunks=covered_chunks,
                            phase1_threshold=0.6,
                        )
                    )

        # Verify all gaps attempted
        assert call_count == 3

        # Verify timeout handled gracefully
        assert gap_stats["gaps_selected"] == 3
        assert gap_stats["gaps_filled"] == 2  # Gap 1 and 3 succeeded
        assert gap_stats["chunks_added"] == 2  # 1 chunk from each successful gap

        # Verify chunks from successful gaps only
        gap_chunk_ids = {c["chunk_id"] for c in all_chunks if c["chunk_id"].startswith("g")}
        assert gap_chunk_ids == {"g1_c1", "g3_c1"}


class TestAsyncGatherImplementation:
    """Test that implementation correctly uses asyncio.gather with return_exceptions=True.

    Verifies the implementation pattern at lines 663-664 in gap_detection.py.
    """

    @pytest.mark.asyncio
    async def test_asyncio_gather_uses_return_exceptions(
        self, gap_detection_service, covered_chunks
    ):
        """Should verify asyncio.gather() uses return_exceptions=True pattern.

        This test verifies the implementation detail: asyncio.gather() is called
        with return_exceptions=True, which returns exceptions as results instead
        of propagating them.

        Expected behavior:
        - Exceptions don't propagate to caller
        - Exceptions appear as Exception instances in results
        - Processing loop handles both Exception and list results
        - No unhandled exceptions crash the pipeline
        """
        # Create mix of success and failure scenarios
        call_count = 0

        async def mock_fill_single_gap(root_query, gap, phase1_threshold, path_filter):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return [{"chunk_id": "g1", "rerank_score": 0.9, "code": "success1"}]
            elif call_count == 2:
                raise ValueError("Deliberate failure for testing")
            elif call_count == 3:
                return [{"chunk_id": "g3", "rerank_score": 0.85, "code": "success3"}]
            else:  # call_count == 4
                raise RuntimeError("Another deliberate failure")

        unified_gaps = [
            UnifiedGap("success1", [], 1, 0.9, score=0.9),
            UnifiedGap("fail1", [], 1, 0.8, score=0.8),
            UnifiedGap("success2", [], 1, 0.7, score=0.7),
            UnifiedGap("fail2", [], 1, 0.6, score=0.6),
        ]

        with patch.object(
            gap_detection_service,
            "_fill_single_gap",
            side_effect=mock_fill_single_gap,
        ):
            with patch.object(
                gap_detection_service,
                "_cluster_chunks_hdbscan",
                new_callable=AsyncMock,
                return_value=[covered_chunks],
            ):
                with patch.object(
                    gap_detection_service,
                    "_select_gaps_by_elbow",
                    return_value=unified_gaps,
                ):
                    # This should NOT raise an exception
                    all_chunks, gap_stats = (
                        await gap_detection_service.detect_and_fill_gaps(
                            root_query="test query",
                            covered_chunks=covered_chunks,
                            phase1_threshold=0.6,
                        )
                    )

        # Verify all gaps attempted
        assert call_count == 4

        # Verify correct handling of mixed results
        assert gap_stats["gaps_selected"] == 4
        assert gap_stats["gaps_filled"] == 2  # Only successes counted
        assert gap_stats["chunks_added"] == 2  # Only successful chunks

        # Verify only successful gap chunks present
        gap_chunk_ids = {c["chunk_id"] for c in all_chunks if c["chunk_id"].startswith("g")}
        assert gap_chunk_ids == {"g1", "g3"}


class TestExceptionTypeHandling:
    """Test handling of different exception types during parallel gap fills.

    Verifies that various exception types are all handled uniformly.
    """

    @pytest.mark.asyncio
    async def test_various_exception_types_handled_uniformly(
        self, gap_detection_service, covered_chunks
    ):
        """Should handle different exception types uniformly.

        Scenario: Different exception types across gap fills
        - RuntimeError (infrastructure failure)
        - ValueError (invalid data)
        - asyncio.TimeoutError (timeout)
        - KeyError (missing data)
        - Exception (generic)

        Expected behavior:
        - All exception types converted to empty lists
        - All logged as warnings
        - No distinction in handling (all treated as failures)
        """
        call_count = 0
        exceptions = [
            RuntimeError("Infrastructure failure"),
            ValueError("Invalid query format"),
            asyncio.TimeoutError("Operation timed out"),
            KeyError("Missing configuration"),
            Exception("Generic error"),
        ]

        async def mock_fill_single_gap(root_query, gap, phase1_threshold, path_filter):
            nonlocal call_count
            exc = exceptions[call_count]
            call_count += 1
            raise exc

        unified_gaps = [
            UnifiedGap(f"query{i}", [], 1, 0.9 - i * 0.1, score=0.9 - i * 0.1)
            for i in range(5)
        ]

        with patch.object(
            gap_detection_service,
            "_fill_single_gap",
            side_effect=mock_fill_single_gap,
        ):
            with patch.object(
                gap_detection_service,
                "_cluster_chunks_hdbscan",
                new_callable=AsyncMock,
                return_value=[covered_chunks],
            ):
                with patch.object(
                    gap_detection_service,
                    "_select_gaps_by_elbow",
                    return_value=unified_gaps,
                ):
                    all_chunks, gap_stats = (
                        await gap_detection_service.detect_and_fill_gaps(
                            root_query="test query",
                            covered_chunks=covered_chunks,
                            phase1_threshold=0.6,
                        )
                    )

        # Verify all exception types handled
        assert call_count == 5

        # All gaps failed uniformly
        assert gap_stats["gaps_selected"] == 5
        assert gap_stats["gaps_filled"] == 0
        assert gap_stats["chunks_added"] == 0

        # Only original coverage remains
        assert len(all_chunks) == len(covered_chunks)


class TestPartialSuccessScenarios:
    """Test realistic scenarios with mix of successes, failures, and empty results."""

    @pytest.mark.asyncio
    async def test_realistic_mixed_scenario(
        self, gap_detection_service, covered_chunks
    ):
        """Should handle realistic scenario with successes, failures, and empties.

        Scenario: 5 gaps with different outcomes
        - gap1: Success with 3 chunks
        - gap2: Exception (ValueError)
        - gap3: Success but filtered to 0 chunks (empty result)
        - gap4: Success with 1 chunk
        - gap5: Exception (TimeoutError)

        Expected behavior:
        - gaps_filled = 2 (gap1 and gap4, gap3 returned empty)
        - chunks_added = 4 (3 + 1)
        - Exceptions logged but don't crash pipeline
        - Empty results don't count as filled
        """
        call_count = 0

        async def mock_fill_single_gap(root_query, gap, phase1_threshold, path_filter):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # Gap 1: Success with 3 chunks
                return [
                    {"chunk_id": "g1_c1", "rerank_score": 0.9, "code": "chunk1"},
                    {"chunk_id": "g1_c2", "rerank_score": 0.85, "code": "chunk2"},
                    {"chunk_id": "g1_c3", "rerank_score": 0.8, "code": "chunk3"},
                ]
            elif call_count == 2:
                # Gap 2: Exception
                raise ValueError("Search failed")
            elif call_count == 3:
                # Gap 3: Success but empty (all filtered by threshold)
                return []
            elif call_count == 4:
                # Gap 4: Success with 1 chunk
                return [
                    {"chunk_id": "g4_c1", "rerank_score": 0.88, "code": "chunk4"},
                ]
            else:  # call_count == 5
                # Gap 5: Timeout
                raise asyncio.TimeoutError("Search timed out")

        unified_gaps = [
            UnifiedGap(f"query{i}", [], 1, 0.9 - i * 0.1, score=0.9 - i * 0.1)
            for i in range(5)
        ]

        with patch.object(
            gap_detection_service,
            "_fill_single_gap",
            side_effect=mock_fill_single_gap,
        ):
            with patch.object(
                gap_detection_service,
                "_cluster_chunks_hdbscan",
                new_callable=AsyncMock,
                return_value=[covered_chunks],
            ):
                with patch.object(
                    gap_detection_service,
                    "_select_gaps_by_elbow",
                    return_value=unified_gaps,
                ):
                    all_chunks, gap_stats = (
                        await gap_detection_service.detect_and_fill_gaps(
                            root_query="test query",
                            covered_chunks=covered_chunks,
                            phase1_threshold=0.6,
                        )
                    )

        # Verify all gaps attempted
        assert call_count == 5

        # Verify stats
        assert gap_stats["gaps_selected"] == 5
        assert gap_stats["gaps_filled"] == 2  # Only gap1 and gap4 returned non-empty
        assert gap_stats["chunks_added"] == 4  # 3 + 1

        # Verify chunks
        gap_chunk_ids = {c["chunk_id"] for c in all_chunks if c["chunk_id"].startswith("g")}
        assert gap_chunk_ids == {"g1_c1", "g1_c2", "g1_c3", "g4_c1"}
        assert len(all_chunks) == len(covered_chunks) + 4


class TestIsolationInvariants:
    """Test that gap fills are truly independent with no shared state."""

    @pytest.mark.asyncio
    async def test_no_shared_state_between_gap_fills(
        self, gap_detection_service, covered_chunks
    ):
        """Should verify gap fills don't share mutable state.

        Tests the critical invariant from gap_detection.py line 643:
        "Gap fills are INDEPENDENT - no shared mutable state"

        Scenario: Track state modifications during gap fills
        Expected: Each gap fill operates on independent data
        """
        # Track which gaps were processed
        processed_gaps = []

        async def mock_fill_single_gap(root_query, gap, phase1_threshold, path_filter):
            # Record gap processing
            processed_gaps.append(gap.query)

            # Simulate some async work
            await asyncio.sleep(0.001)

            # Return unique chunks per gap
            return [
                {"chunk_id": f"{gap.query}_chunk", "rerank_score": 0.8, "code": gap.query}
            ]

        unified_gaps = [
            UnifiedGap("gap1", [], 1, 0.9, score=0.9),
            UnifiedGap("gap2", [], 1, 0.8, score=0.8),
            UnifiedGap("gap3", [], 1, 0.7, score=0.7),
        ]

        with patch.object(
            gap_detection_service,
            "_fill_single_gap",
            side_effect=mock_fill_single_gap,
        ):
            with patch.object(
                gap_detection_service,
                "_cluster_chunks_hdbscan",
                new_callable=AsyncMock,
                return_value=[covered_chunks],
            ):
                with patch.object(
                    gap_detection_service,
                    "_select_gaps_by_elbow",
                    return_value=unified_gaps,
                ):
                    all_chunks, gap_stats = (
                        await gap_detection_service.detect_and_fill_gaps(
                            root_query="test query",
                            covered_chunks=covered_chunks,
                            phase1_threshold=0.6,
                        )
                    )

        # Verify all gaps processed
        assert len(processed_gaps) == 3
        assert set(processed_gaps) == {"gap1", "gap2", "gap3"}

        # Verify each gap produced unique chunks
        gap_chunk_ids = {c["chunk_id"] for c in all_chunks if "_chunk" in c["chunk_id"]}
        assert gap_chunk_ids == {"gap1_chunk", "gap2_chunk", "gap3_chunk"}

        # Verify stats
        assert gap_stats["gaps_filled"] == 3
        assert gap_stats["chunks_added"] == 3
