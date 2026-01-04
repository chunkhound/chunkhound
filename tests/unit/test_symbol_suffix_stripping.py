"""Tests for symbol suffix stripping in search results.

This module tests the functionality that strips _partN suffixes from symbol names
in search results. These suffixes are added during chunk splitting and need to be
hidden from consumers like deep research to ensure regex searches work correctly.
"""


from chunkhound.services.search.result_enhancer import _strip_chunk_part_suffix


class TestStripChunkPartSuffix:
    """Tests for _strip_chunk_part_suffix utility function."""

    def test_strip_binary_split_part1(self):
        """Should strip _part1 suffix from binary split."""
        assert (
            _strip_chunk_part_suffix("DeepResearchService_part1")
            == "DeepResearchService"
        )

    def test_strip_binary_split_part2(self):
        """Should strip _part2 suffix from binary split."""
        assert (
            _strip_chunk_part_suffix("IndexingCoordinator_part2")
            == "IndexingCoordinator"
        )

    def test_strip_multi_part_split(self):
        """Should strip _partN suffix from multi-part emergency split."""
        assert _strip_chunk_part_suffix("MinifiedCode_part3") == "MinifiedCode"
        assert _strip_chunk_part_suffix("LargeClass_part10") == "LargeClass"
        assert _strip_chunk_part_suffix("VeryLargeClass_part33") == "VeryLargeClass"

    def test_strip_nested_split(self):
        """Should strip nested _part1_part2 suffixes from recursive splitting."""
        assert (
            _strip_chunk_part_suffix("VeryLargeService_part1_part1")
            == "VeryLargeService"
        )
        assert (
            _strip_chunk_part_suffix("VeryLargeService_part1_part2")
            == "VeryLargeService"
        )
        assert (
            _strip_chunk_part_suffix("VeryLargeService_part2_part1")
            == "VeryLargeService"
        )

    def test_strip_deeply_nested_split(self):
        """Should strip deeply nested _partN_partM_partK suffixes."""
        assert _strip_chunk_part_suffix("HugeClass_part1_part2_part1") == "HugeClass"
        assert _strip_chunk_part_suffix("HugeClass_part2_part3_part4") == "HugeClass"

    def test_preserve_symbol_with_part_in_name(self):
        """Should preserve symbols that legitimately contain 'part' in their name."""
        assert _strip_chunk_part_suffix("parse_parts") == "parse_parts"
        assert _strip_chunk_part_suffix("get_participant") == "get_participant"
        assert _strip_chunk_part_suffix("MyPartialClass") == "MyPartialClass"
        assert _strip_chunk_part_suffix("part_number") == "part_number"

    def test_preserve_normal_symbols(self):
        """Should not modify symbols without _partN suffixes."""
        assert _strip_chunk_part_suffix("normal_function") == "normal_function"
        assert _strip_chunk_part_suffix("MyClass") == "MyClass"
        assert _strip_chunk_part_suffix("_private_method") == "_private_method"
        assert _strip_chunk_part_suffix("__init__") == "__init__"

    def test_preserve_block_line_symbols(self):
        """Should preserve block_line_N symbols (only strip _partN suffix)."""
        assert _strip_chunk_part_suffix("block_line_767") == "block_line_767"
        assert _strip_chunk_part_suffix("block_line_767_part1") == "block_line_767"
        assert _strip_chunk_part_suffix("block_line_783_part2") == "block_line_783"

    def test_preserve_comment_line_symbols(self):
        """Should preserve comment_line_N symbols (only strip _partN suffix)."""
        assert _strip_chunk_part_suffix("comment_line_42") == "comment_line_42"
        assert _strip_chunk_part_suffix("comment_line_42_part1") == "comment_line_42"

    def test_handle_empty_string(self):
        """Should handle empty string gracefully."""
        assert _strip_chunk_part_suffix("") == ""

    def test_handle_symbol_ending_with_number(self):
        """Should not strip symbols that end with numbers but not _partN pattern."""
        assert _strip_chunk_part_suffix("function123") == "function123"
        assert _strip_chunk_part_suffix("test_case_2") == "test_case_2"
        assert _strip_chunk_part_suffix("v2") == "v2"

    def test_strip_with_underscores_in_name(self):
        """Should strip _partN even when symbol has many underscores."""
        assert (
            _strip_chunk_part_suffix("_extract_symbols_from_chunks_part1")
            == "_extract_symbols_from_chunks"
        )
        assert (
            _strip_chunk_part_suffix("__private__method_part2") == "__private__method"
        )

    def test_real_world_examples(self):
        """Test with real-world symbol names from ChunkHound codebase."""
        # From deep_research_service.py
        assert (
            _strip_chunk_part_suffix("DeepResearchService_part33")
            == "DeepResearchService"
        )
        assert (
            _strip_chunk_part_suffix("_extract_symbols_from_chunks_part2")
            == "_extract_symbols_from_chunks"
        )

        # From universal_parser.py
        assert _strip_chunk_part_suffix("UniversalParser_part1") == "UniversalParser"
        assert (
            _strip_chunk_part_suffix("_apply_cast_algorithm_part2")
            == "_apply_cast_algorithm"
        )

        # Malformed cases from database
        assert (
            _strip_chunk_part_suffix("ract_symbols_from_chunks(_part2")
            == "ract_symbols_from_chunks("
        )


class TestSearchResultEnhancement:
    """Integration tests for symbol cleaning in search results."""

    def test_enhance_result_cleans_symbol(self):
        """Should clean symbol and preserve original in metadata."""
        from unittest.mock import MagicMock

        from chunkhound.services.search_service import SearchService

        # Create mock service
        mock_db = MagicMock()
        service = SearchService(mock_db, embedding_provider=None)

        # Test result with suffixed symbol
        result = {
            "chunk_id": 123,
            "symbol": "MyClass_part2",
            "content": "class MyClass:\n    pass",
            "file_path": "test.py",
            "start_line": 1,
            "end_line": 10,
        }

        enhanced = service._result_enhancer.enhance_search_result(result)

        # Should have clean symbol
        assert enhanced["symbol"] == "MyClass"

        # Should preserve original in metadata
        assert "metadata" in enhanced
        assert enhanced["metadata"]["original_symbol"] == "MyClass_part2"

    def test_enhance_result_preserves_clean_symbol(self):
        """Should not modify symbols without suffixes."""
        from unittest.mock import MagicMock

        from chunkhound.services.search_service import SearchService

        mock_db = MagicMock()
        service = SearchService(mock_db, embedding_provider=None)

        result = {
            "chunk_id": 123,
            "symbol": "normal_function",
            "content": "def normal_function():\n    pass",
            "file_path": "test.py",
            "start_line": 1,
            "end_line": 2,
        }

        enhanced = service._result_enhancer.enhance_search_result(result)

        # Symbol should be unchanged
        assert enhanced["symbol"] == "normal_function"

        # Should NOT add original_symbol to metadata
        metadata = enhanced.get("metadata", {})
        assert "original_symbol" not in metadata

    def test_enhance_result_preserves_existing_metadata(self):
        """Should preserve existing metadata when adding original_symbol."""
        from unittest.mock import MagicMock

        from chunkhound.services.search_service import SearchService

        mock_db = MagicMock()
        service = SearchService(mock_db, embedding_provider=None)

        result = {
            "chunk_id": 123,
            "symbol": "MyClass_part1",
            "content": "class MyClass:\n    pass",
            "metadata": {
                "concept": "definition",
                "language": "python",
                "parameters": ["self"],
            },
        }

        enhanced = service._result_enhancer.enhance_search_result(result)

        # Should clean symbol
        assert enhanced["symbol"] == "MyClass"

        # Should preserve existing metadata AND add original_symbol
        assert enhanced["metadata"]["concept"] == "definition"
        assert enhanced["metadata"]["language"] == "python"
        assert enhanced["metadata"]["parameters"] == ["self"]
        assert enhanced["metadata"]["original_symbol"] == "MyClass_part1"

    def test_enhance_result_handles_nested_suffixes(self):
        """Should handle deeply nested _partN suffixes."""
        from unittest.mock import MagicMock

        from chunkhound.services.search_service import SearchService

        mock_db = MagicMock()
        service = SearchService(mock_db, embedding_provider=None)

        result = {
            "chunk_id": 123,
            "symbol": "VeryLargeClass_part1_part2_part1",
            "content": "class VeryLargeClass:\n    pass",
        }

        enhanced = service._result_enhancer.enhance_search_result(result)

        # Should strip all nested suffixes
        assert enhanced["symbol"] == "VeryLargeClass"
        assert (
            enhanced["metadata"]["original_symbol"]
            == "VeryLargeClass_part1_part2_part1"
        )

    def test_enhance_result_handles_none_symbol(self):
        """Should handle results without symbol field gracefully."""
        from unittest.mock import MagicMock

        from chunkhound.services.search_service import SearchService

        mock_db = MagicMock()
        service = SearchService(mock_db, embedding_provider=None)

        result = {
            "chunk_id": 123,
            "content": "some code",
            "file_path": "test.py",
        }

        enhanced = service._result_enhancer.enhance_search_result(result)

        # Should not crash, just not have symbol field
        assert "symbol" not in enhanced

    def test_enhance_result_handles_empty_symbol(self):
        """Should handle empty symbol field gracefully."""
        from unittest.mock import MagicMock

        from chunkhound.services.search_service import SearchService

        mock_db = MagicMock()
        service = SearchService(mock_db, embedding_provider=None)

        result = {
            "chunk_id": 123,
            "symbol": "",
            "content": "some code",
        }

        enhanced = service._result_enhancer.enhance_search_result(result)

        # Should preserve empty symbol
        assert enhanced["symbol"] == ""
        # Should not add metadata
        metadata = enhanced.get("metadata", {})
        assert "original_symbol" not in metadata
