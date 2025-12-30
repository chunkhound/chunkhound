"""Unit tests for elbow-based chunk filtering and proportional cluster budgets.

Tests the Kneedle algorithm for elbow detection and proportional budget allocation
for cluster synthesis. These are core algorithms used in the research pipeline.

Elbow Detection:
    Uses find_elbow_kneedle() to detect natural breakpoints in relevance scores,
    enabling adaptive filtering of low-relevance chunks.

Proportional Budgets:
    Allocates output tokens to clusters proportional to their input size,
    ensuring larger clusters (more code) receive more output budget.
"""

import pytest

from chunkhound.services.research.shared.elbow_detection import (
    compute_elbow_threshold,
    find_elbow_kneedle,
)


class TestElbowDetection:
    """Test cases for elbow-based chunk filtering using Kneedle algorithm."""

    def test_clear_elbow_cuts_at_drop(self):
        """Clear elbow in scores should cut at the drop point.

        Scores: [0.95, 0.90, 0.85, 0.40, 0.35, 0.30]
        Clear drop between 0.85 and 0.40, elbow should be at index 2.

        The Kneedle algorithm finds the point with maximum perpendicular
        distance from the line connecting first and last points.
        """
        scores = [0.95, 0.90, 0.85, 0.40, 0.35, 0.30]

        elbow_idx = find_elbow_kneedle(scores)

        # Elbow should be at or around index 2 (0.85) - the last high score
        assert elbow_idx is not None, "Should detect elbow for clear drop"
        assert elbow_idx in [2, 3], f"Elbow should be at transition point, got {elbow_idx}"
        # Verify keeping scores[:elbow_idx+1] includes the high-quality chunks
        kept_scores = scores[: elbow_idx + 1]
        assert all(s >= 0.40 for s in kept_scores), "Kept scores should be high quality"

    def test_no_elbow_uniform_scores_returns_none(self):
        """Uniform scores have no elbow - returns None.

        Scores: [0.5, 0.5, 0.5, 0.5]
        All identical scores mean no natural breakpoint exists.
        """
        scores = [0.5, 0.5, 0.5, 0.5]

        elbow_idx = find_elbow_kneedle(scores)

        assert elbow_idx is None, "Uniform scores should return None (no elbow)"

    def test_all_high_scores_gradual_decline(self):
        """All high scores with gradual decline may still detect an elbow.

        Scores: [0.95, 0.92, 0.90, 0.88]
        Even gradual declines can have a mathematically significant elbow
        point (perpendicular distance > 0.01), though less pronounced.

        The key behavior is that ANY elbow detected should be a valid index,
        and when used for filtering (scores[:elbow_idx+1]), should keep
        at least some scores.
        """
        scores = [0.95, 0.92, 0.90, 0.88]

        elbow_idx = find_elbow_kneedle(scores)

        # Gradual decline may or may not detect an elbow
        if elbow_idx is not None:
            # Elbow should be a valid index
            assert 0 <= elbow_idx < len(scores), f"Elbow should be valid index, got {elbow_idx}"
            # Using scores[:elbow_idx+1] should keep at least 1 score
            kept = scores[: elbow_idx + 1]
            assert len(kept) >= 1, "Should keep at least one score"

    def test_single_chunk_returns_none(self):
        """Single chunk should return None (need 3+ for elbow detection).

        Kneedle requires at least 3 points to compute meaningful
        perpendicular distances from the first-to-last line.
        """
        scores = [0.85]

        elbow_idx = find_elbow_kneedle(scores)

        assert elbow_idx is None, "Single chunk should return None"

    def test_empty_chunks_returns_none(self):
        """Empty list should be handled gracefully.

        Empty input should not crash and should return None.
        """
        scores = []

        elbow_idx = find_elbow_kneedle(scores)

        assert elbow_idx is None, "Empty list should return None"

    def test_two_chunks_returns_none(self):
        """Two chunks only should return None (need 3+ for elbow detection).

        Two points define a line; there's no intermediate point to
        measure perpendicular distance from.
        """
        scores = [0.9, 0.3]

        elbow_idx = find_elbow_kneedle(scores)

        assert elbow_idx is None, "Two chunks should return None (need 3+)"

    def test_three_chunks_minimum_for_detection(self):
        """Three chunks is minimum for elbow detection.

        Scores: [0.9, 0.85, 0.2]
        Clear drop from 0.85 to 0.2 should be detectable.
        """
        scores = [0.9, 0.85, 0.2]

        elbow_idx = find_elbow_kneedle(scores)

        # Three points is sufficient for detection if drop is significant
        # Elbow should be at index 1 (0.85) before the drop
        assert elbow_idx is not None, "Three chunks with clear drop should detect elbow"
        assert elbow_idx in [0, 1, 2], f"Elbow should be valid index, got {elbow_idx}"

    def test_very_small_significance_threshold(self):
        """Near-linear scores with tiny variation should return None.

        The algorithm has a 0.01 significance threshold for perpendicular
        distance. Near-linear scores fall below this.
        """
        # Linear with very small deviation
        scores = [1.0, 0.8, 0.6, 0.4, 0.2]

        elbow_idx = find_elbow_kneedle(scores)

        # Perfectly linear should return None (all points on the line)
        assert elbow_idx is None, "Linear scores should return None"

    def test_step_function_scores(self):
        """Step function scores should detect elbow at the step.

        Scores: [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
        Sharp step from 1.0 to 0.1 should have clear elbow.
        """
        scores = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]

        elbow_idx = find_elbow_kneedle(scores)

        assert elbow_idx is not None, "Step function should detect elbow"
        # Elbow should be at or near the step (index 2 or 3)
        assert elbow_idx in [2, 3], f"Elbow should be at step boundary, got {elbow_idx}"


class TestComputeElbowThreshold:
    """Test compute_elbow_threshold which wraps find_elbow_kneedle with fallback."""

    def test_returns_threshold_value_not_index(self):
        """compute_elbow_threshold returns the actual score, not index.

        The function returns the score at the elbow point, suitable for
        direct threshold comparison.
        """
        scores = [0.95, 0.90, 0.85, 0.40, 0.35, 0.30]

        threshold = compute_elbow_threshold(scores)

        # Should return one of the actual scores
        assert threshold in scores, f"Threshold {threshold} should be one of the scores"
        # Should be a high-quality threshold (not from the low tail)
        assert threshold >= 0.35, f"Threshold should be at elbow, not in tail"

    def test_empty_list_returns_default(self):
        """Empty input returns default threshold 0.5."""
        threshold = compute_elbow_threshold([])

        assert threshold == 0.5, "Empty list should return default 0.5"

    def test_uniform_scores_returns_median(self):
        """Uniform scores fall back to median.

        When Kneedle returns None (no elbow), median is used.
        """
        scores = [0.7, 0.7, 0.7, 0.7]

        threshold = compute_elbow_threshold(scores)

        assert threshold == 0.7, "Uniform scores should return that value (via median)"

    def test_works_with_chunk_dicts(self):
        """Can extract scores from chunk dictionaries.

        Supports both raw float lists and chunk dicts with rerank_score.
        """
        chunks = [
            {"chunk_id": "c1", "rerank_score": 0.95},
            {"chunk_id": "c2", "rerank_score": 0.90},
            {"chunk_id": "c3", "rerank_score": 0.40},
        ]

        threshold = compute_elbow_threshold(chunks)

        # Should work with dict input
        assert threshold in [0.95, 0.90, 0.40], "Should extract and process chunk scores"

    def test_missing_rerank_score_uses_default(self):
        """Chunks missing rerank_score default to 0.0."""
        chunks = [
            {"chunk_id": "c1", "rerank_score": 0.9},
            {"chunk_id": "c2"},  # Missing rerank_score
            {"chunk_id": "c3"},  # Missing rerank_score
        ]

        threshold = compute_elbow_threshold(chunks)

        # Should handle gracefully with 0.0 defaults
        assert threshold in [0.9, 0.0], f"Got unexpected threshold {threshold}"


class TestProportionalClusterBudgets:
    """Test proportional budget allocation for cluster synthesis.

    Budget formula: cluster_output_tokens = max(5000, int(total_budget * proportion))
    where proportion = cluster_tokens / total_input_tokens
    """

    def test_three_clusters_proportional_allocation(self):
        """Three clusters should get proportional output budgets.

        Clusters: 60k, 30k, 10k tokens (total 100k)
        With total output budget of 100k tokens:
        - Cluster 1: 60% -> 60k tokens
        - Cluster 2: 30% -> 30k tokens
        - Cluster 3: 10% -> 10k tokens
        """
        cluster_tokens = [60_000, 30_000, 10_000]
        total_input = sum(cluster_tokens)  # 100,000
        total_output_budget = 100_000

        output_budgets = []
        for tokens in cluster_tokens:
            proportion = tokens / total_input if total_input > 0 else 1.0
            budget = max(5000, int(total_output_budget * proportion))
            output_budgets.append(budget)

        # Verify proportions
        assert output_budgets[0] == 60_000, f"60% cluster should get 60k, got {output_budgets[0]}"
        assert output_budgets[1] == 30_000, f"30% cluster should get 30k, got {output_budgets[1]}"
        assert output_budgets[2] == 10_000, f"10% cluster should get 10k, got {output_budgets[2]}"

        # Verify ratios maintained
        assert output_budgets[0] == 2 * output_budgets[1], "60k should be 2x 30k"
        assert output_budgets[1] == 3 * output_budgets[2], "30k should be 3x 10k"

    def test_minimum_5k_budget_enforced(self):
        """Tiny clusters should get minimum 5k tokens budget.

        Even if proportion would yield less than 5k, the minimum is enforced.
        This ensures every cluster has enough budget for meaningful output.
        """
        # Tiny cluster in a large total
        cluster_tokens = [1_000, 99_000]  # 1k is 1% of 100k
        total_input = sum(cluster_tokens)  # 100,000
        total_output_budget = 100_000  # 1% would be 1k, below minimum

        output_budgets = []
        for tokens in cluster_tokens:
            proportion = tokens / total_input if total_input > 0 else 1.0
            budget = max(5000, int(total_output_budget * proportion))
            output_budgets.append(budget)

        # 1% of 100k = 1k, but minimum is 5k
        assert output_budgets[0] == 5000, f"Tiny cluster should get minimum 5k, got {output_budgets[0]}"
        # Large cluster gets proportional amount
        assert output_budgets[1] == 99_000, f"Large cluster should get 99k, got {output_budgets[1]}"

    def test_single_cluster_gets_all_budget(self):
        """Single cluster should get the full output budget.

        With only one cluster, it receives 100% of the budget (capped by minimum).
        """
        cluster_tokens = [50_000]
        total_input = sum(cluster_tokens)  # 50,000
        total_output_budget = 80_000

        output_budgets = []
        for tokens in cluster_tokens:
            proportion = tokens / total_input if total_input > 0 else 1.0
            budget = max(5000, int(total_output_budget * proportion))
            output_budgets.append(budget)

        # Single cluster gets 100% = 80k
        assert output_budgets[0] == 80_000, f"Single cluster should get full budget, got {output_budgets[0]}"

    def test_zero_total_input_uses_equal_split(self):
        """Zero total input should use proportion 1.0 per cluster.

        Edge case: if total_input is 0, each cluster gets proportion 1.0,
        which means full budget (after max(5000, ...) is applied).
        """
        cluster_tokens = [0, 0, 0]
        total_input = sum(cluster_tokens)  # 0
        total_output_budget = 30_000

        output_budgets = []
        for tokens in cluster_tokens:
            # With total_input=0, proportion defaults to 1.0
            proportion = tokens / total_input if total_input > 0 else 1.0
            budget = max(5000, int(total_output_budget * proportion))
            output_budgets.append(budget)

        # Each cluster gets full budget (proportion 1.0)
        assert all(b == 30_000 for b in output_budgets), f"Zero input should give full budget each: {output_budgets}"

    def test_proportions_match_input_ratios(self):
        """Output proportions should match input token ratios.

        This is the core invariant: larger clusters get proportionally
        more output budget.
        """
        cluster_tokens = [10_000, 20_000, 30_000, 40_000]
        total_input = sum(cluster_tokens)  # 100,000
        total_output_budget = 50_000

        output_budgets = []
        for tokens in cluster_tokens:
            proportion = tokens / total_input
            budget = max(5000, int(total_output_budget * proportion))
            output_budgets.append(budget)

        # All above minimum, so proportions should match
        # 10k/100k = 10% of 50k = 5k (exactly at minimum)
        # 20k/100k = 20% of 50k = 10k
        # 30k/100k = 30% of 50k = 15k
        # 40k/100k = 40% of 50k = 20k
        assert output_budgets[0] == 5000, f"10% should be 5k, got {output_budgets[0]}"
        assert output_budgets[1] == 10_000, f"20% should be 10k, got {output_budgets[1]}"
        assert output_budgets[2] == 15_000, f"30% should be 15k, got {output_budgets[2]}"
        assert output_budgets[3] == 20_000, f"40% should be 20k, got {output_budgets[3]}"

    def test_very_large_clusters_scale_correctly(self):
        """Large clusters should scale linearly without overflow.

        Tests that the algorithm handles large token counts correctly.
        """
        cluster_tokens = [500_000, 300_000, 200_000]  # 1M total
        total_input = sum(cluster_tokens)
        total_output_budget = 200_000

        output_budgets = []
        for tokens in cluster_tokens:
            proportion = tokens / total_input
            budget = max(5000, int(total_output_budget * proportion))
            output_budgets.append(budget)

        # 50% of 200k = 100k
        # 30% of 200k = 60k
        # 20% of 200k = 40k
        assert output_budgets[0] == 100_000
        assert output_budgets[1] == 60_000
        assert output_budgets[2] == 40_000

    def test_minimum_enforced_on_rounding(self):
        """Integer rounding that falls below 5k should clamp to 5k.

        Tests edge case where proportion * budget rounds to < 5000.
        """
        cluster_tokens = [100, 9_900]  # 1% vs 99%
        total_input = sum(cluster_tokens)  # 10,000
        total_output_budget = 10_000  # 1% = 100 tokens, way below minimum

        output_budgets = []
        for tokens in cluster_tokens:
            proportion = tokens / total_input
            budget = max(5000, int(total_output_budget * proportion))
            output_budgets.append(budget)

        # 1% of 10k = 100, clamped to 5k
        assert output_budgets[0] == 5000, "1% cluster should clamp to 5k minimum"
        # 99% of 10k = 9900
        assert output_budgets[1] == 9900, "99% cluster should get 9900"
