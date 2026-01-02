"""Tests for elbow-based chunk filtering utility.

This module tests the filter_chunks_by_elbow function which provides
unified elbow-based filtering for both BFS and WideCoverage strategies.
"""

import pytest

from chunkhound.services.research.shared.exploration.elbow_filter import (
    filter_chunks_by_elbow,
)


class TestFilterChunksByElbow:
    """Tests for filter_chunks_by_elbow core functionality."""

    def test_empty_input_returns_empty_with_passthrough_stats(self):
        """Empty input returns empty list with passthrough method.

        Empty input is a valid edge case that should not raise errors.
        """
        filtered, stats = filter_chunks_by_elbow([])

        assert filtered == []
        assert stats["method"] == "passthrough"
        assert stats["reason"] == "empty_input"
        assert stats["original_count"] == 0
        assert stats["filtered_count"] == 0

    def test_single_chunk_returns_all_too_few_chunks(self):
        """Single chunk returns unchanged with too_few_chunks reason.

        Elbow detection requires at least 3 points to find a knee.
        """
        chunks = [{"chunk_id": "c1", "score": 0.9, "content": "test"}]

        filtered, stats = filter_chunks_by_elbow(chunks)

        assert len(filtered) == 1
        assert filtered[0] == chunks[0]
        assert stats["method"] == "passthrough"
        assert stats["reason"] == "too_few_chunks"

    def test_two_chunks_returns_all_too_few_chunks(self):
        """Two chunks return unchanged with too_few_chunks reason.

        Elbow detection requires at least 3 points to find a knee.
        """
        chunks = [
            {"chunk_id": "c1", "score": 0.9},
            {"chunk_id": "c2", "score": 0.5},
        ]

        filtered, stats = filter_chunks_by_elbow(chunks)

        assert len(filtered) == 2
        assert stats["method"] == "passthrough"
        assert stats["reason"] == "too_few_chunks"

    def test_clear_elbow_filters_low_relevance_chunks(self):
        """Clear elbow in scores filters out low-relevance chunks.

        When there's a clear drop in scores, chunks below the elbow
        should be filtered out.
        """
        chunks = [
            {"chunk_id": "c1", "score": 0.95},
            {"chunk_id": "c2", "score": 0.90},
            {"chunk_id": "c3", "score": 0.85},
            {"chunk_id": "c4", "score": 0.40},  # Clear drop here
            {"chunk_id": "c5", "score": 0.35},
            {"chunk_id": "c6", "score": 0.30},
        ]

        filtered, stats = filter_chunks_by_elbow(chunks)

        assert stats["method"] == "elbow"
        assert stats["reason"] == "elbow_detected"
        assert len(filtered) < len(chunks), "Should filter some chunks"
        assert stats["filtered_count"] == len(filtered)
        assert stats["original_count"] == 6
        assert "cutoff_score" in stats
        assert "elbow_index" in stats

    def test_uniform_scores_returns_all_no_elbow(self):
        """Uniform scores return all chunks with no_elbow_detected reason.

        When all scores are identical, there's no natural breakpoint.
        """
        chunks = [
            {"chunk_id": "c1", "score": 0.5},
            {"chunk_id": "c2", "score": 0.5},
            {"chunk_id": "c3", "score": 0.5},
            {"chunk_id": "c4", "score": 0.5},
        ]

        filtered, stats = filter_chunks_by_elbow(chunks)

        assert len(filtered) == 4
        assert stats["method"] == "passthrough"
        assert stats["reason"] == "no_elbow_detected"

    def test_gradual_decline_may_return_all(self):
        """Gradual score decline may not detect a clear elbow.

        When scores decline gradually without a sharp drop,
        the algorithm may not find a significant elbow point.
        """
        chunks = [
            {"chunk_id": "c1", "score": 0.90},
            {"chunk_id": "c2", "score": 0.85},
            {"chunk_id": "c3", "score": 0.80},
            {"chunk_id": "c4", "score": 0.75},
            {"chunk_id": "c5", "score": 0.70},
        ]

        filtered, stats = filter_chunks_by_elbow(chunks)

        # Either elbow is detected or all are returned - both valid
        assert stats["method"] in ["elbow", "passthrough"]
        assert len(filtered) <= len(chunks)

    def test_unsorted_input_gets_sorted(self):
        """Chunks are sorted by score descending before processing.

        Input order doesn't matter - output is always score-sorted.
        """
        chunks = [
            {"chunk_id": "c3", "score": 0.50},
            {"chunk_id": "c1", "score": 0.90},
            {"chunk_id": "c2", "score": 0.70},
        ]

        filtered, stats = filter_chunks_by_elbow(chunks)

        # Verify output is sorted descending
        scores = [c["score"] for c in filtered]
        assert scores == sorted(scores, reverse=True), "Output should be sorted descending"
        assert filtered[0]["chunk_id"] == "c1", "Highest score should be first"

    def test_three_chunks_minimum_for_elbow_detection(self):
        """Three chunks is the minimum for elbow detection.

        With exactly 3 points, elbow detection can potentially work.
        """
        chunks = [
            {"chunk_id": "c1", "score": 0.95},
            {"chunk_id": "c2", "score": 0.50},  # Clear drop
            {"chunk_id": "c3", "score": 0.10},
        ]

        filtered, stats = filter_chunks_by_elbow(chunks)

        # Should attempt elbow detection (not passthrough due to count)
        assert stats["reason"] != "too_few_chunks"
        assert len(filtered) >= 1


class TestFilterChunksByElbowScoreKey:
    """Tests for configurable score key parameter."""

    def test_default_score_key(self):
        """Default score_key is 'score'.

        BFS strategy uses 'score' as the chunk score field.
        """
        chunks = [
            {"chunk_id": "c1", "score": 0.95},
            {"chunk_id": "c2", "score": 0.90},
            {"chunk_id": "c3", "score": 0.40},
        ]

        filtered, stats = filter_chunks_by_elbow(chunks)  # No score_key arg

        assert len(filtered) <= 3
        assert stats["original_count"] == 3

    def test_custom_rerank_score_key(self):
        """Custom score_key 'rerank_score' for WideCoverage strategy.

        WideCoverage uses 'rerank_score' from unified search.
        """
        chunks = [
            {"chunk_id": "c1", "rerank_score": 0.95},
            {"chunk_id": "c2", "rerank_score": 0.90},
            {"chunk_id": "c3", "rerank_score": 0.40},
        ]

        filtered, stats = filter_chunks_by_elbow(chunks, score_key="rerank_score")

        assert len(filtered) <= 3
        assert stats["original_count"] == 3

    def test_missing_score_key_uses_default_zero(self):
        """Missing score key defaults to 0.0.

        Chunks without the specified score key are treated as 0 score.
        """
        chunks = [
            {"chunk_id": "c1", "score": 0.95},
            {"chunk_id": "c2"},  # Missing 'score' key
            {"chunk_id": "c3", "score": 0.40},
        ]

        filtered, stats = filter_chunks_by_elbow(chunks)

        # Should not raise an error
        assert len(filtered) <= 3


class TestFilterChunksByElbowStats:
    """Tests for filter stats accuracy and structure."""

    def test_stats_counts_match_actual_results(self):
        """Stats counts accurately reflect filtered results.

        The filtered_count should match len(filtered_chunks).
        """
        chunks = [
            {"chunk_id": "c1", "score": 0.95},
            {"chunk_id": "c2", "score": 0.90},
            {"chunk_id": "c3", "score": 0.40},
            {"chunk_id": "c4", "score": 0.35},
        ]

        filtered, stats = filter_chunks_by_elbow(chunks)

        assert stats["filtered_count"] == len(filtered)
        assert stats["original_count"] == 4

    def test_elbow_stats_include_cutoff_and_index(self):
        """Elbow detection stats include cutoff_score and elbow_index.

        When elbow is detected, additional metadata is provided.
        """
        chunks = [
            {"chunk_id": "c1", "score": 0.95},
            {"chunk_id": "c2", "score": 0.90},
            {"chunk_id": "c3", "score": 0.85},
            {"chunk_id": "c4", "score": 0.40},  # Clear drop
            {"chunk_id": "c5", "score": 0.35},
        ]

        filtered, stats = filter_chunks_by_elbow(chunks)

        if stats["method"] == "elbow":
            assert "cutoff_score" in stats
            assert "elbow_index" in stats
            assert isinstance(stats["cutoff_score"], float)
            assert isinstance(stats["elbow_index"], int)
            assert stats["elbow_index"] >= 0

    def test_passthrough_stats_have_correct_structure(self):
        """Passthrough stats have required fields but no elbow metadata.

        When no elbow is detected, cutoff_score and elbow_index are omitted.
        """
        chunks = [{"chunk_id": "c1", "score": 0.5}]

        filtered, stats = filter_chunks_by_elbow(chunks)

        assert stats["method"] == "passthrough"
        assert "reason" in stats
        assert "original_count" in stats
        assert "filtered_count" in stats
        # Elbow-specific fields should NOT be present
        assert "cutoff_score" not in stats
        assert "elbow_index" not in stats


class TestFilterChunksByElbowIntegration:
    """Integration-style tests for realistic usage patterns."""

    def test_realistic_bfs_chunks(self):
        """Test with realistic BFS chunk structure.

        BFS chunks have 'score' from semantic search.
        """
        chunks = [
            {"chunk_id": "c1", "score": 0.92, "file_path": "src/main.py", "content": "def main()"},
            {"chunk_id": "c2", "score": 0.88, "file_path": "src/utils.py", "content": "def helper()"},
            {"chunk_id": "c3", "score": 0.45, "file_path": "tests/test.py", "content": "def test()"},
            {"chunk_id": "c4", "score": 0.42, "file_path": "docs/readme.md", "content": "# Docs"},
        ]

        filtered, stats = filter_chunks_by_elbow(chunks, score_key="score")

        # All chunk fields should be preserved
        for chunk in filtered:
            assert "chunk_id" in chunk
            assert "file_path" in chunk
            assert "content" in chunk

    def test_realistic_wide_coverage_chunks(self):
        """Test with realistic WideCoverage chunk structure.

        WideCoverage chunks have 'rerank_score' from unified search.
        """
        chunks = [
            {"chunk_id": "c1", "rerank_score": 0.95, "file_path": "src/api.py"},
            {"chunk_id": "c2", "rerank_score": 0.91, "file_path": "src/handler.py"},
            {"chunk_id": "c3", "rerank_score": 0.87, "file_path": "src/model.py"},
            {"chunk_id": "c4", "rerank_score": 0.35, "file_path": "tests/conftest.py"},
            {"chunk_id": "c5", "rerank_score": 0.30, "file_path": "setup.py"},
        ]

        filtered, stats = filter_chunks_by_elbow(chunks, score_key="rerank_score")

        assert stats["original_count"] == 5
        assert len(filtered) <= 5
        # High-scoring chunks should be in output
        assert any(c["chunk_id"] == "c1" for c in filtered)
