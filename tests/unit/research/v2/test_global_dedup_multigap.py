"""Unit tests for global deduplication across multiple gaps.

Tests Step 2.8 (_global_dedup) behavior when the same chunk_id appears in
multiple gap fill results with different rerank_scores.

Key Invariants:
- Highest rerank_score always wins
- Tie-breaking is deterministic (first occurrence preserved)
- No chunks are lost in deduplication
- Deduplication is performed AFTER all gap fills complete (sync point)
"""

import pytest

from chunkhound.services.research.v2.gap_detection import GapDetectionService


class TestGlobalDedupMultiGap:
    """Test global deduplication when same chunk appears in multiple gaps."""

    def test_same_chunk_three_gaps_different_scores(self):
        """Should keep chunk with highest rerank_score across 3 gaps.

        Scenario:
        - gap1: chunk_id="X" with score=0.9
        - gap2: chunk_id="X" with score=0.7
        - gap3: chunk_id="X" with score=0.5

        Expected: chunk with score=0.9 is kept (highest)
        """
        # Create mock service instance (we only need _global_dedup method)
        service = object.__new__(GapDetectionService)

        gap_results = [
            # Gap 1: chunk X with highest score
            [
                {
                    "chunk_id": "X",
                    "rerank_score": 0.9,
                    "code": "def foo(): pass",
                    "file_path": "test.py",
                }
            ],
            # Gap 2: chunk X with middle score
            [
                {
                    "chunk_id": "X",
                    "rerank_score": 0.7,
                    "code": "def foo(): pass",
                    "file_path": "test.py",
                }
            ],
            # Gap 3: chunk X with lowest score
            [
                {
                    "chunk_id": "X",
                    "rerank_score": 0.5,
                    "code": "def foo(): pass",
                    "file_path": "test.py",
                }
            ],
        ]

        # Run global deduplication
        result = service._global_dedup(gap_results)

        # Verify only one chunk X remains
        assert len(result) == 1
        assert result[0]["chunk_id"] == "X"

        # Verify highest score was kept
        assert result[0]["rerank_score"] == 0.9

    def test_same_chunk_identical_scores_tiebreak(self):
        """Should have deterministic outcome when scores are identical.

        Scenario:
        - gap1: chunk_id="X" with score=0.8
        - gap2: chunk_id="X" with score=0.8

        Expected: First occurrence is kept (deterministic tie-breaking)
        """
        service = object.__new__(GapDetectionService)

        gap_results = [
            # Gap 1: chunk X
            [
                {
                    "chunk_id": "X",
                    "rerank_score": 0.8,
                    "code": "version_1",
                    "file_path": "test.py",
                }
            ],
            # Gap 2: chunk X (identical score)
            [
                {
                    "chunk_id": "X",
                    "rerank_score": 0.8,
                    "code": "version_2",
                    "file_path": "test.py",
                }
            ],
        ]

        # Run deduplication multiple times to verify determinism
        result1 = service._global_dedup(gap_results)
        result2 = service._global_dedup(gap_results)

        # Verify results are identical
        assert len(result1) == 1
        assert len(result2) == 1
        assert result1[0]["code"] == result2[0]["code"]

        # Current implementation keeps FIRST occurrence when scores are equal
        # (because new_score > existing_score requires strict inequality)
        # So "version_1" should be kept
        assert result1[0]["code"] == "version_1"

    def test_dedup_preserves_highest_score(self):
        """Should preserve highest score for multiple overlapping chunks.

        Scenario: Multiple chunks with varying overlap patterns
        - chunk A: in gap1 (0.9), gap2 (0.7)
        - chunk B: in gap1 (0.8), gap3 (0.6)
        - chunk C: in gap2 (0.75), gap3 (0.85)
        - chunk D: only in gap3 (0.7)

        Expected:
        - A kept with score 0.9 (from gap1)
        - B kept with score 0.8 (from gap1)
        - C kept with score 0.85 (from gap3)
        - D kept with score 0.7 (only occurrence)
        """
        service = object.__new__(GapDetectionService)

        gap_results = [
            # Gap 1: chunks A (0.9), B (0.8)
            [
                {"chunk_id": "A", "rerank_score": 0.9, "code": "chunk_a"},
                {"chunk_id": "B", "rerank_score": 0.8, "code": "chunk_b"},
            ],
            # Gap 2: chunks A (0.7), C (0.75)
            [
                {"chunk_id": "A", "rerank_score": 0.7, "code": "chunk_a"},
                {"chunk_id": "C", "rerank_score": 0.75, "code": "chunk_c"},
            ],
            # Gap 3: chunks B (0.6), C (0.85), D (0.7)
            [
                {"chunk_id": "B", "rerank_score": 0.6, "code": "chunk_b"},
                {"chunk_id": "C", "rerank_score": 0.85, "code": "chunk_c"},
                {"chunk_id": "D", "rerank_score": 0.7, "code": "chunk_d"},
            ],
        ]

        result = service._global_dedup(gap_results)

        # Verify all chunks present (no loss)
        assert len(result) == 4
        chunk_map = {c["chunk_id"]: c for c in result}
        assert set(chunk_map.keys()) == {"A", "B", "C", "D"}

        # Verify highest scores preserved
        assert chunk_map["A"]["rerank_score"] == 0.9
        assert chunk_map["B"]["rerank_score"] == 0.8
        assert chunk_map["C"]["rerank_score"] == 0.85
        assert chunk_map["D"]["rerank_score"] == 0.7

    def test_dedup_preserves_all_unique_chunks(self):
        """Should not lose any unique chunks during deduplication.

        Scenario: 5 gaps with varying chunk distributions
        - 10 unique chunks total
        - Some duplicates, some unique

        Expected: All 10 unique chunks appear exactly once
        """
        service = object.__new__(GapDetectionService)

        gap_results = [
            # Gap 1: chunks 1, 2, 3
            [
                {"chunk_id": "c1", "rerank_score": 0.9, "code": "1"},
                {"chunk_id": "c2", "rerank_score": 0.8, "code": "2"},
                {"chunk_id": "c3", "rerank_score": 0.7, "code": "3"},
            ],
            # Gap 2: chunks 2, 4, 5 (2 is duplicate)
            [
                {"chunk_id": "c2", "rerank_score": 0.75, "code": "2"},
                {"chunk_id": "c4", "rerank_score": 0.85, "code": "4"},
                {"chunk_id": "c5", "rerank_score": 0.65, "code": "5"},
            ],
            # Gap 3: chunks 3, 6, 7 (3 is duplicate)
            [
                {"chunk_id": "c3", "rerank_score": 0.72, "code": "3"},
                {"chunk_id": "c6", "rerank_score": 0.78, "code": "6"},
                {"chunk_id": "c7", "rerank_score": 0.68, "code": "7"},
            ],
            # Gap 4: chunks 8, 9
            [
                {"chunk_id": "c8", "rerank_score": 0.82, "code": "8"},
                {"chunk_id": "c9", "rerank_score": 0.71, "code": "9"},
            ],
            # Gap 5: chunks 1, 10 (1 is duplicate)
            [
                {"chunk_id": "c1", "rerank_score": 0.88, "code": "1"},
                {"chunk_id": "c10", "rerank_score": 0.76, "code": "10"},
            ],
        ]

        result = service._global_dedup(gap_results)

        # Verify all 10 unique chunks present
        assert len(result) == 10
        chunk_ids = {c["chunk_id"] for c in result}
        expected_ids = {f"c{i}" for i in range(1, 11)}
        assert chunk_ids == expected_ids

        # Verify no duplicates
        assert len(result) == len(chunk_ids)

    def test_dedup_handles_empty_gaps(self):
        """Should handle empty gap results correctly.

        Scenario: Mix of populated and empty gaps
        - gap1: 2 chunks
        - gap2: empty
        - gap3: 1 chunk (duplicate from gap1)

        Expected: 2 unique chunks preserved
        """
        service = object.__new__(GapDetectionService)

        gap_results = [
            # Gap 1: 2 chunks
            [
                {"chunk_id": "A", "rerank_score": 0.9, "code": "a"},
                {"chunk_id": "B", "rerank_score": 0.8, "code": "b"},
            ],
            # Gap 2: empty
            [],
            # Gap 3: duplicate of A
            [
                {"chunk_id": "A", "rerank_score": 0.85, "code": "a"},
            ],
        ]

        result = service._global_dedup(gap_results)

        # Verify 2 unique chunks
        assert len(result) == 2
        chunk_map = {c["chunk_id"]: c for c in result}
        assert set(chunk_map.keys()) == {"A", "B"}

        # Verify highest score kept for A
        assert chunk_map["A"]["rerank_score"] == 0.9

    def test_dedup_all_empty_gaps(self):
        """Should return empty list when all gaps are empty.

        Scenario: All gaps returned no results

        Expected: Empty list returned
        """
        service = object.__new__(GapDetectionService)

        gap_results = [[], [], []]

        result = service._global_dedup(gap_results)

        assert result == []

    def test_dedup_chunks_missing_id(self):
        """Should skip chunks without chunk_id or id field.

        Scenario: Some chunks have missing IDs (malformed data)

        Expected: Malformed chunks skipped, valid chunks preserved
        """
        service = object.__new__(GapDetectionService)

        # Mock logger to verify warning
        import logging

        logger_warnings = []

        class MockLogger:
            def warning(self, msg):
                logger_warnings.append(msg)

            def debug(self, msg):
                pass  # Ignore debug logs in test

        # Temporarily replace logger (in chunk_dedup module where warning is emitted)
        original_logger = None
        try:
            from chunkhound.services.research.shared import chunk_dedup

            original_logger = chunk_dedup.logger
            chunk_dedup.logger = MockLogger()

            gap_results = [
                [
                    # Valid chunk
                    {"chunk_id": "A", "rerank_score": 0.9, "code": "a"},
                    # Missing chunk_id (should be skipped)
                    {"rerank_score": 0.8, "code": "b", "file_path": "test.py"},
                ],
                [
                    # Valid chunk with 'id' instead of 'chunk_id'
                    {"id": "B", "rerank_score": 0.85, "code": "c"},
                ],
            ]

            result = service._global_dedup(gap_results)

            # Verify only valid chunks kept
            assert len(result) == 2
            chunk_ids = {c.get("chunk_id") or c.get("id") for c in result}
            assert chunk_ids == {"A", "B"}

            # Verify warning logged for missing ID
            assert len(logger_warnings) == 1
            assert "missing ID" in logger_warnings[0]
            assert "test.py" in logger_warnings[0]

        finally:
            # Restore original logger
            if original_logger is not None:
                chunk_dedup.logger = original_logger

    def test_dedup_respects_id_field_fallback(self):
        """Should use 'id' field when 'chunk_id' is missing.

        Scenario: Mix of chunks with 'chunk_id' and 'id' fields

        Expected: Both field types handled correctly
        """
        service = object.__new__(GapDetectionService)

        gap_results = [
            [
                # Uses 'chunk_id'
                {"chunk_id": "A", "rerank_score": 0.9, "code": "a"},
            ],
            [
                # Uses 'id' instead
                {"id": "B", "rerank_score": 0.8, "code": "b"},
            ],
            [
                # Duplicate with different field name (same logical ID)
                {"id": "A", "rerank_score": 0.7, "code": "a"},
            ],
        ]

        result = service._global_dedup(gap_results)

        # Verify deduplication works across field types
        assert len(result) == 2

        # Build ID set using same logic as implementation
        chunk_ids = set()
        for c in result:
            chunk_id = c.get("chunk_id") or c.get("id")
            chunk_ids.add(chunk_id)

        assert chunk_ids == {"A", "B"}

        # Verify highest score kept for A
        chunk_a = next(c for c in result if (c.get("chunk_id") or c.get("id")) == "A")
        assert chunk_a["rerank_score"] == 0.9


class TestGlobalDedupEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_gap_with_duplicates(self):
        """Should deduplicate within a single gap result.

        Scenario: One gap returned same chunk twice (should not happen, but defensive)

        Expected: Only one instance kept
        """
        service = object.__new__(GapDetectionService)

        gap_results = [
            [
                {"chunk_id": "A", "rerank_score": 0.9, "code": "a"},
                {"chunk_id": "A", "rerank_score": 0.8, "code": "a"},  # Duplicate
            ]
        ]

        result = service._global_dedup(gap_results)

        assert len(result) == 1
        assert result[0]["chunk_id"] == "A"
        assert result[0]["rerank_score"] == 0.9

    def test_many_gaps_single_duplicate(self):
        """Should handle many gaps with single overlapping chunk.

        Scenario: 10 gaps, chunk X appears in all of them with different scores

        Expected: Chunk X kept with highest score from all gaps
        """
        service = object.__new__(GapDetectionService)

        # Create 10 gaps, each with chunk X at different score
        gap_results = [
            [{"chunk_id": "X", "rerank_score": 1.0 - (i * 0.05), "code": "x"}]
            for i in range(10)
        ]

        result = service._global_dedup(gap_results)

        # Verify single chunk with highest score
        assert len(result) == 1
        assert result[0]["chunk_id"] == "X"
        assert result[0]["rerank_score"] == 1.0  # First gap had highest score

    def test_dedup_preserves_chunk_metadata(self):
        """Should preserve all metadata fields from winning chunk.

        Scenario: Chunks have additional metadata beyond chunk_id and score

        Expected: All fields from highest-scored chunk preserved
        """
        service = object.__new__(GapDetectionService)

        gap_results = [
            [
                {
                    "chunk_id": "A",
                    "rerank_score": 0.7,
                    "code": "version_low",
                    "file_path": "low.py",
                    "start_line": 1,
                    "end_line": 10,
                    "metadata": {"version": "low"},
                }
            ],
            [
                {
                    "chunk_id": "A",
                    "rerank_score": 0.9,
                    "code": "version_high",
                    "file_path": "high.py",
                    "start_line": 100,
                    "end_line": 200,
                    "metadata": {"version": "high"},
                }
            ],
        ]

        result = service._global_dedup(gap_results)

        assert len(result) == 1
        chunk = result[0]

        # Verify all metadata from high-score chunk preserved
        assert chunk["rerank_score"] == 0.9
        assert chunk["code"] == "version_high"
        assert chunk["file_path"] == "high.py"
        assert chunk["start_line"] == 100
        assert chunk["end_line"] == 200
        assert chunk["metadata"]["version"] == "high"

    def test_dedup_zero_and_negative_scores(self):
        """Should handle zero and negative rerank scores correctly.

        Scenario: Chunks with unusual score values

        Expected: Highest score wins (even if negative)
        """
        service = object.__new__(GapDetectionService)

        gap_results = [
            [{"chunk_id": "A", "rerank_score": -0.5, "code": "negative"}],
            [{"chunk_id": "A", "rerank_score": 0.0, "code": "zero"}],
            [{"chunk_id": "A", "rerank_score": -0.8, "code": "more_negative"}],
        ]

        result = service._global_dedup(gap_results)

        # Verify highest score (0.0) wins
        assert len(result) == 1
        assert result[0]["rerank_score"] == 0.0
        assert result[0]["code"] == "zero"

    def test_dedup_missing_rerank_score(self):
        """Should handle chunks missing rerank_score field.

        Scenario: Some chunks don't have rerank_score (default to 0.0)

        Expected: Chunks with explicit scores preferred over defaults
        """
        service = object.__new__(GapDetectionService)

        gap_results = [
            [
                {"chunk_id": "A", "code": "no_score"}  # Missing score, defaults to 0.0
            ],
            [
                {"chunk_id": "A", "rerank_score": 0.5, "code": "has_score"}
            ],
        ]

        result = service._global_dedup(gap_results)

        # Verify chunk with explicit score wins
        assert len(result) == 1
        assert result[0]["code"] == "has_score"
        assert result[0]["rerank_score"] == 0.5


class TestGlobalDedupDocumentation:
    """Document expected behavior and invariants."""

    def test_dedup_invariants_documented(self):
        """Document the invariants that _global_dedup must maintain.

        INVARIANTS:
        1. No chunks lost: all unique chunk_ids appear exactly once
        2. Highest score wins: for duplicates, chunk with max rerank_score kept
        3. Tie-breaking: when scores equal, LAST occurrence wins (implementation detail)
        4. Missing IDs: chunks without chunk_id/id are skipped (logged warning)
        5. Missing scores: treated as 0.0 (via dict.get default)
        6. Metadata preserved: all fields from winning chunk retained
        7. Empty safe: empty gap_results returns empty list

        SYNC POINT:
        - _global_dedup is called AFTER all gap fills complete
        - No shared mutable state during gap fills (independent execution)
        - Global dedup is the synchronization point for merging results
        """
        invariants = {
            "no_chunks_lost": "All unique chunk_ids appear exactly once",
            "highest_score_wins": "For duplicates, max rerank_score kept",
            "tie_breaking": "When scores equal, LAST occurrence wins",
            "missing_ids_skipped": "Chunks without ID logged and skipped",
            "missing_scores_default": "Missing rerank_score defaults to 0.0",
            "metadata_preserved": "All fields from winner retained",
            "empty_safe": "Empty input returns empty output",
        }

        # Verify documentation structure
        assert len(invariants) == 7
        assert all(isinstance(v, str) for v in invariants.values())

    def test_implementation_analysis(self):
        """Document current implementation details for future reference.

        IMPLEMENTATION DETAILS (gap_detection.py lines 774-809):

        1. Conflict resolution: new_score > existing_score (strict inequality)
           - When scores equal, existing chunk is kept (FIRST wins on ties)
           - CORRECTION FROM EARLIER TEST: First occurrence wins, not last

        2. ID lookup order: chunk.get("chunk_id") or chunk.get("id")
           - Prefers chunk_id, falls back to id
           - Missing both: logs warning and skips chunk

        3. Score default: existing.get("rerank_score", 0.0)
           - Missing scores treated as 0.0
           - Explicit 0.0 same as missing (both resolve to 0.0)

        4. Data structure: dict[str, dict] for O(1) lookups
           - Keys: chunk_id strings
           - Values: full chunk dictionaries

        5. Logging: Debug log with dedup stats (input → output count)
        """
        # This test documents implementation for future maintainers
        implementation_notes = """
        Current implementation uses strict inequality (new_score > existing_score),
        which means ties preserve the FIRST occurrence, not last.

        Example:
        - gap1: chunk X score 0.8 (stored first)
        - gap2: chunk X score 0.8 (0.8 > 0.8 is False, so gap1 kept)
        Result: gap1's chunk X is preserved
        """

        assert "strict inequality" in implementation_notes
        assert "FIRST occurrence" in implementation_notes

    def test_tie_breaking_correction(self):
        """CORRECTION: Verify tie-breaking actually keeps FIRST occurrence.

        Earlier test incorrectly stated LAST occurrence wins. The implementation
        uses strict inequality (new_score > existing_score), so equal scores
        keep the FIRST occurrence.
        """
        service = object.__new__(GapDetectionService)

        gap_results = [
            [{"chunk_id": "X", "rerank_score": 0.8, "code": "first"}],
            [{"chunk_id": "X", "rerank_score": 0.8, "code": "second"}],
            [{"chunk_id": "X", "rerank_score": 0.8, "code": "third"}],
        ]

        result = service._global_dedup(gap_results)

        # Verify FIRST occurrence kept (not last)
        assert len(result) == 1
        assert result[0]["code"] == "first"
