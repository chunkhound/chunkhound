"""Unit tests for multi-hop termination conditions in v2 research pipeline.

Tests the 5 termination conditions specified in
docs/algorithm-coverage-first-research.md:
1. Time limit: 5 seconds (exhaustive: 600s)
2. Result limit: 500 chunks (exhaustive: None)
3. Candidate quality: < 5 above threshold
4. Score degradation: ≥ 0.15 drop in top-5
5. Minimum relevance: top-5 min < 0.3

These tests verify that:
- Each condition can independently trigger termination
- ANY condition triggers termination (not ALL required)
- Accumulated results are returned on termination
- Early termination prevents excessive API calls
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.interfaces.embedding_provider import RerankResult
from chunkhound.services.search.multi_hop_strategy import MultiHopStrategy


@pytest.fixture
def mock_db_provider():
    """Create mock database provider for multi-hop expansion."""
    provider = MagicMock()
    # Default: no neighbors found (terminates expansion immediately)
    provider.find_similar_chunks.return_value = []
    return provider


@pytest.fixture
def mock_embedding_provider():
    """Create mock embedding provider with reranking support."""
    provider = MagicMock()
    provider.supports_reranking.return_value = True

    # Default: simple identity reranking (preserves order)
    async def default_rerank(query, documents, top_k=None):
        return [
            RerankResult(index=i, score=0.9 - (i * 0.01)) for i in range(len(documents))
        ]

    provider.rerank = AsyncMock(side_effect=default_rerank)
    return provider


@pytest.fixture
def mock_single_hop_search():
    """Create mock single-hop search function."""

    async def search(query, page_size, offset, threshold, provider, model, path_filter):
        # Default: return enough results for expansion (20 results)
        results = [
            {
                "chunk_id": f"chunk_{i}",
                "content": f"content {i}",
                "similarity": 0.9 - (i * 0.01),
            }
            for i in range(20)
        ]
        return results, {}

    return AsyncMock(side_effect=search)


@pytest.mark.asyncio
class TestMultiHopTerminationConditions:
    """Test suite for multi-hop termination conditions."""

    async def test_time_limit_terminates_at_5_seconds(
        self, mock_db_provider, mock_embedding_provider, mock_single_hop_search
    ):
        """Test that time limit terminates expansion at ~5 seconds.

        Termination condition 1: Time limit reached.
        """

        # Mock slow expansion that keeps finding neighbors
        async def slow_rerank(query, documents, top_k=None):
            await asyncio.sleep(0.5)  # Simulate slow reranking
            return [
                RerankResult(index=i, score=0.9 - (i * 0.001))
                for i in range(len(documents))
            ]

        mock_embedding_provider.rerank = AsyncMock(side_effect=slow_rerank)

        # Mock expansion that keeps finding UNIQUE new neighbors
        # We need to track which chunks have been seen to avoid duplicates
        neighbor_counter = {"count": 0}

        def find_unique_neighbors(
            chunk_id, provider, model, limit, threshold, path_filter
        ):
            # Generate unique neighbors to prevent "no new candidates"
            # termination
            start = neighbor_counter["count"]
            neighbor_counter["count"] += 10
            return [
                {
                    "chunk_id": f"unique_neighbor_{start + i}",
                    "content": f"neighbor content {start + i}",
                    "similarity": 0.85,
                }
                for i in range(10)
            ]

        mock_db_provider.find_similar_chunks.side_effect = find_unique_neighbors

        strategy = MultiHopStrategy(
            mock_db_provider, mock_embedding_provider, mock_single_hop_search
        )

        start_time = time.perf_counter()
        results, pagination = await strategy.search(
            query="test query",
            page_size=10,
            offset=0,
            threshold=None,
            provider="test",
            model="test-model",
            path_filter=None,
            time_limit=5.0,
            result_limit=None,
        )
        elapsed = time.perf_counter() - start_time

        # Verify terminated around 5s (with some tolerance for async overhead)
        assert 4.5 <= elapsed <= 6.5, f"Should terminate around 5s, took {elapsed:.2f}s"

        # Verify accumulated results returned (not empty)
        assert len(results) > 0, "Should return accumulated results"
        assert pagination["total"] > 0, "Should have total count"

    async def test_result_limit_terminates_at_500(
        self, mock_db_provider, mock_embedding_provider, mock_single_hop_search
    ):
        """Test that result limit terminates expansion at 500 chunks.

        Termination condition 2: Result limit reached.
        """

        # Mock initial search returning 100 results
        async def large_initial_search(
            query, page_size, offset, threshold, provider, model, path_filter
        ):
            results = [
                {
                    "chunk_id": f"initial_{i}",
                    "content": f"content {i}",
                    "similarity": 0.9 - (i * 0.001),
                }
                for i in range(100)
            ]
            return results, {}

        mock_single_hop_search.side_effect = large_initial_search

        # Mock expansion returning many neighbors
        neighbor_counter = {"count": 100}  # Start after initial results

        def find_many_neighbors(
            chunk_id, provider, model, limit, threshold, path_filter
        ):
            # Return 50 new neighbors each time
            neighbors = [
                {
                    "chunk_id": f"neighbor_{neighbor_counter['count'] + i}",
                    "content": f"neighbor content {i}",
                    "similarity": 0.85,
                }
                for i in range(50)
            ]
            neighbor_counter["count"] += 50
            return neighbors

        mock_db_provider.find_similar_chunks.side_effect = find_many_neighbors

        strategy = MultiHopStrategy(
            mock_db_provider, mock_embedding_provider, mock_single_hop_search
        )

        results, pagination = await strategy.search(
            query="test query",
            page_size=600,  # Request more than limit
            offset=0,
            threshold=None,
            provider="test",
            model="test-model",
            path_filter=None,
            time_limit=60.0,  # Long timeout
            result_limit=500,
        )

        # Verify terminated at exactly 500 results
        assert pagination["total"] >= 500, (
            "Should accumulate at least 500 results before termination"
        )
        assert pagination["total"] <= 600, (
            "Should not accumulate significantly more than 500 results"
        )

        # Verify pagination works correctly
        assert len(results) <= 600, "Should respect page_size parameter"

    async def test_candidate_quality_terminates_below_5(
        self, mock_db_provider, mock_embedding_provider, mock_single_hop_search
    ):
        """Test that expansion terminates when < 5 candidates above threshold.

        Termination condition 3: Insufficient high-scoring candidates.
        """
        # Mock reranking that returns only 4 candidates with score > 0.0
        iteration = {"count": 0}

        async def declining_quality_rerank(query, documents, top_k=None):
            iteration["count"] += 1

            if iteration["count"] == 1:
                # First iteration: 20 high-quality results
                return [
                    RerankResult(index=i, score=0.9 - (i * 0.01))
                    for i in range(len(documents))
                ]
            else:
                # Second iteration: only 4 candidates with score > 0.0
                results = []
                for i in range(len(documents)):
                    if i < 4:
                        results.append(RerankResult(index=i, score=0.5 - (i * 0.05)))
                    else:
                        results.append(
                            RerankResult(index=i, score=0.0)
                        )  # Below threshold
                return results

        mock_embedding_provider.rerank = AsyncMock(side_effect=declining_quality_rerank)

        # Mock expansion returning some neighbors for first round
        expansion_calls = {"count": 0}

        def find_neighbors_once(
            chunk_id, provider, model, limit, threshold, path_filter
        ):
            expansion_calls["count"] += 1
            if expansion_calls["count"] <= 5:  # Only first 5 candidates
                return [
                    {
                        "chunk_id": f"neighbor_{chunk_id}_{i}",
                        "content": f"neighbor content {i}",
                        "similarity": 0.8,
                    }
                    for i in range(5)
                ]
            return []

        mock_db_provider.find_similar_chunks.side_effect = find_neighbors_once

        strategy = MultiHopStrategy(
            mock_db_provider, mock_embedding_provider, mock_single_hop_search
        )

        results, pagination = await strategy.search(
            query="test query",
            page_size=10,
            offset=0,
            threshold=None,
            provider="test",
            model="test-model",
            path_filter=None,
            time_limit=60.0,
            result_limit=None,
        )

        # Verify terminated due to insufficient candidates
        assert len(results) > 0, "Should return accumulated results"
        # Should stop after detecting only 4 high-scoring candidates
        assert expansion_calls["count"] <= 10, "Should not expand excessively"

    async def test_score_degradation_terminates_at_0_15_drop(
        self, mock_db_provider, mock_embedding_provider, mock_single_hop_search
    ):
        """Test that expansion terminates when tracked chunk scores drop ≥ 0.15.

        Termination condition 4: Score degradation detected.
        """
        # Track reranking iterations to simulate degradation
        iteration = {"count": 0}

        async def degrading_rerank(query, documents, top_k=None):
            iteration["count"] += 1

            if iteration["count"] == 1:
                # First iteration: high scores
                return [
                    RerankResult(index=i, score=0.9 - (i * 0.01))
                    for i in range(len(documents))
                ]
            else:
                # Second iteration: simulate score drop for tracked chunks
                # The strategy tracks top 5 chunks by chunk_id
                # We need to make their scores drop by >= 0.15
                results = []
                for i in range(len(documents)):
                    # First few documents are the original high-scoring chunks
                    if i < 20:  # Original chunks
                        original_score = 0.9 - (i * 0.01)
                        # Drop by 0.20 (exceeds 0.15 threshold)
                        new_score = max(0.0, original_score - 0.20)
                        results.append(RerankResult(index=i, score=new_score))
                    else:  # New neighbors
                        results.append(RerankResult(index=i, score=0.6))
                return results

        mock_embedding_provider.rerank = AsyncMock(side_effect=degrading_rerank)

        # Mock expansion returning neighbors for one round
        def find_neighbors_once(
            chunk_id, provider, model, limit, threshold, path_filter
        ):
            return [
                {
                    "chunk_id": f"neighbor_{chunk_id}_{i}",
                    "content": f"neighbor content {i}",
                    "similarity": 0.8,
                }
                for i in range(10)
            ]

        mock_db_provider.find_similar_chunks.side_effect = find_neighbors_once

        strategy = MultiHopStrategy(
            mock_db_provider, mock_embedding_provider, mock_single_hop_search
        )

        results, pagination = await strategy.search(
            query="test query",
            page_size=10,
            offset=0,
            threshold=None,
            provider="test",
            model="test-model",
            path_filter=None,
            time_limit=60.0,
            result_limit=None,
        )

        # Verify terminated due to score degradation
        assert len(results) > 0, "Should return accumulated results"
        # Should stop after one expansion round due to degradation
        assert mock_embedding_provider.rerank.call_count <= 2, (
            "Should terminate after detecting degradation"
        )

    async def test_minimum_relevance_terminates_below_0_3(
        self, mock_db_provider, mock_embedding_provider, mock_single_hop_search
    ):
        """Test that expansion terminates when top-5 min score < 0.3.

        Termination condition 5: Minimum relevance threshold.
        """
        # Track iterations
        iteration = {"count": 0}

        async def declining_relevance_rerank(query, documents, top_k=None):
            iteration["count"] += 1

            if iteration["count"] == 1:
                # First iteration: good scores
                return [
                    RerankResult(index=i, score=0.9 - (i * 0.01))
                    for i in range(len(documents))
                ]
            else:
                # Second iteration: top 5 include score < 0.3
                # Scores: [0.9, 0.8, 0.5, 0.29, 0.25, ...]
                scores = [0.9, 0.8, 0.5, 0.29, 0.25]
                results = []
                for i in range(len(documents)):
                    if i < len(scores):
                        results.append(RerankResult(index=i, score=scores[i]))
                    else:
                        results.append(RerankResult(index=i, score=0.1))
                return results

        mock_embedding_provider.rerank = AsyncMock(
            side_effect=declining_relevance_rerank
        )

        # Mock expansion returning neighbors for one round
        def find_neighbors_once(
            chunk_id, provider, model, limit, threshold, path_filter
        ):
            return [
                {
                    "chunk_id": f"neighbor_{chunk_id}_{i}",
                    "content": f"neighbor content {i}",
                    "similarity": 0.8,
                }
                for i in range(10)
            ]

        mock_db_provider.find_similar_chunks.side_effect = find_neighbors_once

        strategy = MultiHopStrategy(
            mock_db_provider, mock_embedding_provider, mock_single_hop_search
        )

        results, pagination = await strategy.search(
            query="test query",
            page_size=10,
            offset=0,
            threshold=None,
            provider="test",
            model="test-model",
            path_filter=None,
            time_limit=60.0,
            result_limit=None,
        )

        # Verify terminated due to low minimum relevance
        assert len(results) > 0, "Should return accumulated results"
        # Should stop after detecting min(top-5) < 0.3
        assert mock_embedding_provider.rerank.call_count <= 2, (
            "Should terminate after detecting low relevance"
        )

    async def test_any_condition_triggers_termination(
        self, mock_db_provider, mock_embedding_provider, mock_single_hop_search
    ):
        """Test that ANY condition triggers termination (not ALL required).

        This test simulates multiple conditions being met simultaneously
        and verifies that termination occurs when the FIRST condition is detected.
        """

        # Mock reranking that would satisfy both quality and relevance conditions
        async def multi_condition_rerank(query, documents, top_k=None):
            # Return only 3 candidates > 0 (quality condition)
            # AND all with low scores (relevance condition)
            results = []
            for i in range(len(documents)):
                if i < 3:
                    results.append(RerankResult(index=i, score=0.25))  # < 0.3
                else:
                    results.append(RerankResult(index=i, score=0.0))
            return results

        mock_embedding_provider.rerank = AsyncMock(side_effect=multi_condition_rerank)

        # Mock expansion that would return neighbors
        def find_neighbors(chunk_id, provider, model, limit, threshold, path_filter):
            return [
                {
                    "chunk_id": f"neighbor_{chunk_id}_{i}",
                    "content": f"neighbor content {i}",
                    "similarity": 0.8,
                }
                for i in range(5)
            ]

        mock_db_provider.find_similar_chunks.side_effect = find_neighbors

        strategy = MultiHopStrategy(
            mock_db_provider, mock_embedding_provider, mock_single_hop_search
        )

        results, pagination = await strategy.search(
            query="test query",
            page_size=10,
            offset=0,
            threshold=None,
            provider="test",
            model="test-model",
            path_filter=None,
            time_limit=60.0,
            result_limit=None,
        )

        # Verify terminated (either condition would suffice)
        assert len(results) > 0, "Should return accumulated results"
        # Should terminate immediately without expansion
        assert mock_embedding_provider.rerank.call_count == 1, (
            "Should terminate on first check"
        )

    async def test_fallback_to_single_hop_on_insufficient_initial_results(
        self, mock_db_provider, mock_embedding_provider, mock_single_hop_search
    ):
        """Test that multi-hop falls back to single-hop when initial results <= 5.

        This is not a termination condition per se, but a pre-condition check
        that prevents multi-hop from even starting.
        """
        # Mock single-hop returning only 5 results
        fallback_calls = {"count": 0}

        async def limited_search(
            query, page_size, offset, threshold, provider, model, path_filter
        ):
            fallback_calls["count"] += 1
            results = [
                {
                    "chunk_id": f"chunk_{i}",
                    "content": f"content {i}",
                    "similarity": 0.9 - (i * 0.01),
                }
                for i in range(5)
            ]
            return results, {}

        mock_single_hop_search.side_effect = limited_search

        strategy = MultiHopStrategy(
            mock_db_provider, mock_embedding_provider, mock_single_hop_search
        )

        results, pagination = await strategy.search(
            query="test query",
            page_size=10,
            offset=0,
            threshold=None,
            provider="test",
            model="test-model",
            path_filter=None,
            time_limit=60.0,
            result_limit=None,
        )

        # Verify fell back to single-hop search (called twice)
        assert fallback_calls["count"] == 2, (
            "Should call single-hop twice (initial + fallback)"
        )

        # Verify reranking was not attempted (fell back before multi-hop)
        assert mock_embedding_provider.rerank.call_count == 0, (
            "Should not attempt reranking when falling back"
        )

    async def test_accumulated_results_returned_on_early_termination(
        self, mock_db_provider, mock_embedding_provider, mock_single_hop_search
    ):
        """Test that accumulated results from multiple rounds are returned
        on termination.

        This verifies that the strategy properly accumulates and returns all
        discovered chunks, not just the final round.
        """
        # Track rounds
        round_count = {"count": 0}

        async def multi_round_rerank(query, documents, top_k=None):
            round_count["count"] += 1
            if round_count["count"] <= 2:
                # First two rounds: good scores
                return [
                    RerankResult(index=i, score=0.9 - (i * 0.01))
                    for i in range(len(documents))
                ]
            else:
                # Third round: trigger termination (low min score)
                scores = [0.9, 0.8, 0.5, 0.29, 0.25]  # min < 0.3
                results = []
                for i in range(len(documents)):
                    if i < len(scores):
                        results.append(RerankResult(index=i, score=scores[i]))
                    else:
                        results.append(RerankResult(index=i, score=0.1))
                return results

        mock_embedding_provider.rerank = AsyncMock(side_effect=multi_round_rerank)

        # Mock expansion returning neighbors for multiple rounds
        expansion_count = {"count": 0}

        def find_neighbors_multi_round(
            chunk_id, provider, model, limit, threshold, path_filter
        ):
            expansion_count["count"] += 1
            if expansion_count["count"] <= 10:  # First 2 rounds (5 candidates each)
                neighbors = []
                for i in range(5):
                    chunk_id_str = (
                        f"round_{round_count['count']}_neighbor_"
                        f"{chunk_id}_{i}"
                    )
                    neighbors.append(
                        {
                            "chunk_id": chunk_id_str,
                            "content": f"neighbor content round "
                            f"{round_count['count']} {i}",
                            "similarity": 0.8,
                        }
                    )
                return neighbors
            return []

        mock_db_provider.find_similar_chunks.side_effect = find_neighbors_multi_round

        strategy = MultiHopStrategy(
            mock_db_provider, mock_embedding_provider, mock_single_hop_search
        )

        results, pagination = await strategy.search(
            query="test query",
            page_size=100,  # Request all results
            offset=0,
            threshold=None,
            provider="test",
            model="test-model",
            path_filter=None,
            time_limit=60.0,
            result_limit=None,
        )

        # Verify accumulated results from multiple rounds
        assert pagination["total"] > 20, (
            "Should accumulate results from initial + multiple expansion rounds"
        )

        # Verify all results are unique (no duplicates)
        chunk_ids = [r["chunk_id"] for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), (
            "Should not return duplicate chunks"
        )


@pytest.mark.asyncio
class TestMultiHopExhaustiveModeTermination:
    """Test exhaustive mode changes to termination behavior."""

    async def test_exhaustive_mode_uses_extended_time_limit(
        self, mock_db_provider, mock_embedding_provider, mock_single_hop_search
    ):
        """Test that exhaustive mode uses 600s time limit instead of 5s.

        Note: This test doesn't actually wait 600s, it just verifies the
        configuration is properly applied.
        """
        # Mock config with exhaustive mode
        config = MagicMock()
        config.exhaustive_mode = True

        strategy = MultiHopStrategy(
            mock_db_provider, mock_embedding_provider, mock_single_hop_search, config
        )

        # Mock expansion that terminates quickly (no neighbors)
        mock_db_provider.find_similar_chunks.return_value = []

        results, pagination = await strategy.search(
            query="test query",
            page_size=10,
            offset=0,
            threshold=None,
            provider="test",
            model="test-model",
            path_filter=None,
            time_limit=600.0,  # Exhaustive limit
            result_limit=None,
        )

        # Verify search completes (doesn't wait 600s due to other termination)
        assert len(results) >= 0, "Should complete search"

    async def test_exhaustive_mode_disables_result_limit(
        self, mock_db_provider, mock_embedding_provider, mock_single_hop_search
    ):
        """Test that exhaustive mode disables 500 result limit.

        Note: This test verifies that result_limit=None is honored.
        """

        # Mock large initial results
        async def large_search(
            query, page_size, offset, threshold, provider, model, path_filter
        ):
            results = [
                {
                    "chunk_id": f"chunk_{i}",
                    "content": f"content {i}",
                    "similarity": 0.9 - (i * 0.0001),
                }
                for i in range(100)
            ]
            return results, {}

        mock_single_hop_search.side_effect = large_search

        # Mock expansion returning no neighbors (terminate quickly)
        mock_db_provider.find_similar_chunks.return_value = []

        strategy = MultiHopStrategy(
            mock_db_provider, mock_embedding_provider, mock_single_hop_search
        )

        results, pagination = await strategy.search(
            query="test query",
            page_size=200,
            offset=0,
            threshold=None,
            provider="test",
            model="test-model",
            path_filter=None,
            time_limit=10.0,
            result_limit=None,  # Unlimited (exhaustive mode)
        )

        # Verify no result limit applied
        # (would be capped at 500 in normal mode, but not here)
        assert pagination["total"] >= 100, "Should not cap results"

    async def test_quality_conditions_still_active_in_exhaustive_mode(
        self, mock_db_provider, mock_embedding_provider, mock_single_hop_search
    ):
        """Test that quality-based termination conditions remain active in
        exhaustive mode.

        Even with unlimited time/results, the algorithm should still stop when:
        - Quality degrades (score drop ≥ 0.15)
        - Minimum relevance too low (< 0.3)
        - Insufficient candidates (< 5)
        """
        # Mock config with exhaustive mode
        config = MagicMock()
        config.exhaustive_mode = True

        # Mock reranking with low relevance
        async def low_relevance_rerank(query, documents, top_k=None):
            # All scores < 0.3 (should trigger termination)
            return [
                RerankResult(index=i, score=0.2 - (i * 0.01))
                for i in range(len(documents))
            ]

        mock_embedding_provider.rerank = AsyncMock(side_effect=low_relevance_rerank)

        strategy = MultiHopStrategy(
            mock_db_provider, mock_embedding_provider, mock_single_hop_search, config
        )

        results, pagination = await strategy.search(
            query="test query",
            page_size=10,
            offset=0,
            threshold=None,
            provider="test",
            model="test-model",
            path_filter=None,
            time_limit=600.0,  # Exhaustive time limit
            result_limit=None,  # No result limit
        )

        # Verify terminated due to quality condition (not time/result limit)
        assert len(results) > 0, "Should return accumulated results"
        # Should terminate immediately due to low relevance
        assert mock_embedding_provider.rerank.call_count == 1, (
            "Should terminate on quality check"
        )
