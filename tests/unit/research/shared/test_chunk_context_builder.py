"""Unit tests for ChunkContextBuilder.

Tests the chunk context building utilities used across synthesis pipelines
to format chunk data for LLM consumption.

Test Categories:
1. Chunk summary building - navigation summaries with line numbers
2. Code context with imports - aggregated chunk content with import headers
3. Token budget enforcement - respecting max_tokens limits
4. Edge cases - empty chunks, missing fields, etc.
"""

from unittest.mock import MagicMock

import pytest

from chunkhound.services.research.shared.chunk_context_builder import (
    ChunkContextBuilder,
    get_chunk_text,
)


class TestGetChunkText:
    """Tests for get_chunk_text function."""

    def test_returns_content_field(self):
        """Should return content field when present."""
        chunk = {"content": "def main(): pass"}
        assert get_chunk_text(chunk) == "def main(): pass"

    def test_returns_code_field_when_no_content(self):
        """Should fall back to code field when content is missing."""
        chunk = {"code": "function test() {}"}
        assert get_chunk_text(chunk) == "function test() {}"

    def test_returns_text_field_when_no_content_or_code(self):
        """Should fall back to text field when content and code are missing."""
        chunk = {"text": "Some text content"}
        assert get_chunk_text(chunk) == "Some text content"

    def test_content_takes_precedence_over_code(self):
        """Content field should take precedence over code field."""
        chunk = {"content": "content value", "code": "code value"}
        assert get_chunk_text(chunk) == "content value"

    def test_returns_empty_string_when_no_fields(self):
        """Should return empty string when no text fields are present."""
        chunk = {"file_path": "/path/to/file.py", "start_line": 1}
        assert get_chunk_text(chunk) == ""

    def test_handles_empty_content_falls_back_to_code(self):
        """Should fall back to code when content is empty string."""
        chunk = {"content": "", "code": "actual code"}
        assert get_chunk_text(chunk) == "actual code"

    def test_handles_none_content_falls_back_to_code(self):
        """Should fall back to code when content is None."""
        chunk = {"content": None, "code": "actual code"}
        assert get_chunk_text(chunk) == "actual code"

    def test_handles_empty_dict(self):
        """Should handle empty dictionary gracefully."""
        chunk = {}
        assert get_chunk_text(chunk) == ""


class TestBuildChunkSummary:
    """Test chunk summary building for navigation context."""

    def test_single_chunk_with_all_fields(self):
        """Chunk with all fields should format correctly."""
        builder = ChunkContextBuilder()
        chunks = [
            {
                "start_line": 10,
                "end_line": 25,
                "symbol": "process_data",
                "content": "def process_data(items): return [x * 2 for x in items]",
            }
        ]

        result = builder.build_chunk_summary(chunks)

        assert "Lines 10-25" in result
        assert "(process_data)" in result
        assert "def process_data" in result
        assert result.startswith("- Lines")

    def test_chunk_without_symbol(self):
        """Chunk without symbol should omit parentheses."""
        builder = ChunkContextBuilder()
        chunks = [
            {
                "start_line": 1,
                "end_line": 5,
                "content": "# Comment block\n# More comments",
            }
        ]

        result = builder.build_chunk_summary(chunks)

        assert "Lines 1-5" in result
        assert "(" not in result or "()" not in result
        # Should have content preview
        assert "Comment" in result

    def test_empty_chunks_returns_no_chunks(self):
        """Empty chunk list should return '(no chunks)'."""
        builder = ChunkContextBuilder()

        result = builder.build_chunk_summary([])

        assert result == "(no chunks)"

    def test_multiple_chunks_formats_each(self):
        """Multiple chunks should each be formatted on separate lines."""
        builder = ChunkContextBuilder()
        chunks = [
            {"start_line": 1, "end_line": 10, "symbol": "func_a", "content": "def func_a(): pass"},
            {"start_line": 12, "end_line": 20, "symbol": "func_b", "content": "def func_b(): pass"},
        ]

        result = builder.build_chunk_summary(chunks)

        lines = result.split("\n")
        assert len(lines) == 2
        assert "(func_a)" in lines[0]
        assert "(func_b)" in lines[1]

    def test_max_chunks_limits_output(self):
        """Should respect max_chunks parameter."""
        builder = ChunkContextBuilder()
        chunks = [
            {"start_line": i, "end_line": i + 5, "symbol": f"func_{i}", "content": f"def func_{i}(): pass"}
            for i in range(10)
        ]

        result = builder.build_chunk_summary(chunks, max_chunks=3)

        lines = result.split("\n")
        assert len(lines) == 3
        assert "func_0" in lines[0]
        assert "func_2" in lines[2]

    def test_missing_line_numbers_shows_question_marks(self):
        """Missing line numbers should show '?' as placeholder."""
        builder = ChunkContextBuilder()
        chunks = [
            {"content": "some code here"}
        ]

        result = builder.build_chunk_summary(chunks)

        assert "Lines ?-?" in result

    def test_long_content_truncated_at_200_chars(self):
        """Content preview should be truncated at 200 characters."""
        builder = ChunkContextBuilder()
        long_content = "x" * 500
        chunks = [
            {"start_line": 1, "end_line": 10, "content": long_content}
        ]

        result = builder.build_chunk_summary(chunks)

        # Content portion should be at most 200 chars + "..."
        # The line format is "- Lines X-Y: <content>..."
        content_start = result.find(": ") + 2
        content_portion = result[content_start:]
        assert content_portion == "x" * 200 + "..."


class TestBuildCodeContextWithImports:
    """Test code context building with import headers."""

    def test_single_chunk_without_imports(self):
        """Single chunk without import service should format content only."""
        builder = ChunkContextBuilder()
        chunks = [
            {"file_path": "/src/main.py", "content": "def main(): pass"}
        ]

        result = builder.build_code_context_with_imports(chunks, max_tokens=1000)

        assert "File: /src/main.py" in result
        assert "def main(): pass" in result
        # No imports header when import_context_service is None
        assert "# Imports:" not in result

    def test_empty_chunks_returns_empty_string(self):
        """Empty chunk list should return empty string."""
        builder = ChunkContextBuilder()

        result = builder.build_code_context_with_imports([], max_tokens=1000)

        assert result == ""

    def test_multiple_files_grouped_correctly(self):
        """Chunks from multiple files should each have file header."""
        builder = ChunkContextBuilder()
        chunks = [
            {"file_path": "/src/a.py", "content": "code_a"},
            {"file_path": "/src/b.py", "content": "code_b"},
        ]

        result = builder.build_code_context_with_imports(chunks, max_tokens=2000)

        assert "File: /src/a.py" in result
        assert "File: /src/b.py" in result
        assert "code_a" in result
        assert "code_b" in result

    def test_with_import_context_service(self):
        """Should include import headers when import service is provided."""
        mock_import_service = MagicMock()
        mock_import_service.get_file_imports.return_value = [
            "import os",
            "from pathlib import Path",
        ]

        builder = ChunkContextBuilder(import_context_service=mock_import_service)
        chunks = [
            {"file_path": "/src/main.py", "content": "def main(): pass"}
        ]

        result = builder.build_code_context_with_imports(chunks, max_tokens=2000)

        assert "# Imports:" in result
        assert "import os" in result
        assert "from pathlib import Path" in result
        mock_import_service.get_file_imports.assert_called_once()

    def test_import_extraction_failure_graceful(self):
        """Should handle import extraction failures gracefully."""
        mock_import_service = MagicMock()
        mock_import_service.get_file_imports.side_effect = Exception("Parse error")

        builder = ChunkContextBuilder(import_context_service=mock_import_service)
        chunks = [
            {"file_path": "/src/main.py", "content": "def main(): pass"}
        ]

        # Should not raise
        result = builder.build_code_context_with_imports(chunks, max_tokens=2000)

        assert "File: /src/main.py" in result
        assert "def main(): pass" in result
        # No imports header due to failure
        assert "# Imports:" not in result


class TestTokenBudgetEnforcement:
    """Test token budget enforcement in context building."""

    def test_truncates_when_exceeding_budget(self):
        """Should truncate content when exceeding token budget."""
        builder = ChunkContextBuilder()
        # Large content that would exceed budget
        chunks = [
            {"file_path": f"/src/file_{i}.py", "content": "x" * 1000}
            for i in range(10)
        ]

        # Very small budget
        result = builder.build_code_context_with_imports(chunks, max_tokens=100)

        # Should have truncation marker
        assert "(truncated)" in result or len(result) < 1000

    def test_respects_budget_across_chunks(self):
        """Should respect budget across multiple chunks."""
        builder = ChunkContextBuilder()
        chunks = [
            {"file_path": f"/src/file_{i}.py", "content": "code " * 50}
            for i in range(5)
        ]

        # Budget that allows only some chunks
        result = builder.build_code_context_with_imports(chunks, max_tokens=200)

        # Not all files should be present
        file_count = result.count("File:")
        assert file_count < 5

    def test_with_llm_manager_token_estimation(self):
        """Should use LLM provider for token estimation when available."""
        mock_provider = MagicMock()
        mock_provider.estimate_tokens.return_value = 50  # 50 tokens per call

        mock_llm_manager = MagicMock()
        mock_llm_manager.get_utility_provider.return_value = mock_provider

        builder = ChunkContextBuilder(llm_manager=mock_llm_manager)
        chunks = [
            {"file_path": "/src/main.py", "content": "def main(): pass"}
        ]

        result = builder.build_code_context_with_imports(chunks, max_tokens=100)

        assert "File: /src/main.py" in result
        mock_provider.estimate_tokens.assert_called()

    def test_fallback_token_estimation_without_llm(self):
        """Should use character-based estimation when no LLM manager."""
        builder = ChunkContextBuilder()

        # Internal method test
        tokens = builder._estimate_tokens("x" * 400)  # ~100 tokens at 4 chars/token

        assert tokens == 100


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_chunk_with_missing_file_path(self):
        """Chunk without file_path should use 'unknown'."""
        builder = ChunkContextBuilder()
        chunks = [
            {"content": "orphan code"}
        ]

        result = builder.build_code_context_with_imports(chunks, max_tokens=1000)

        assert "File: unknown" in result
        assert "orphan code" in result

    def test_chunk_with_empty_content(self):
        """Chunk with empty content should still format correctly."""
        builder = ChunkContextBuilder()
        chunks = [
            {"file_path": "/src/empty.py", "content": ""}
        ]

        result = builder.build_code_context_with_imports(chunks, max_tokens=1000)

        assert "File: /src/empty.py" in result

    def test_none_fields_handled(self):
        """Should handle None values in chunk fields."""
        builder = ChunkContextBuilder()
        chunks = [
            {"file_path": None, "content": None}
        ]

        # Should not raise
        result = builder.build_chunk_summary(chunks)
        assert "(no chunks)" not in result  # Has one chunk, even if invalid

    def test_llm_manager_exception_falls_back(self):
        """Should fall back to char-based estimation on LLM errors."""
        mock_llm_manager = MagicMock()
        mock_llm_manager.get_utility_provider.side_effect = Exception("Not configured")

        builder = ChunkContextBuilder(llm_manager=mock_llm_manager)

        # Should not raise, falls back to char-based
        tokens = builder._estimate_tokens("test text")
        assert tokens == 2  # "test text" = 9 chars / 4 = 2 tokens


class TestBuildFileGroupedContext:
    """Test file-grouped context building."""

    def test_groups_chunks_by_file(self):
        """Should group chunks by file path."""
        builder = ChunkContextBuilder()
        chunks = [
            {"file_path": "/src/a.py", "content": "code_a1"},
            {"file_path": "/src/b.py", "content": "code_b"},
            {"file_path": "/src/a.py", "content": "code_a2"},
        ]

        result = builder.build_file_grouped_context(chunks, max_tokens=1000)

        assert len(result) == 2
        assert "/src/a.py" in result
        assert "/src/b.py" in result
        # File a should have both chunks
        assert "code_a1" in result["/src/a.py"]
        assert "code_a2" in result["/src/a.py"]

    def test_empty_chunks_returns_empty_dict(self):
        """Empty chunks should return empty dict."""
        builder = ChunkContextBuilder()

        result = builder.build_file_grouped_context([], max_tokens=1000)

        assert result == {}

    def test_respects_per_file_token_budget(self):
        """Should respect token budget per file."""
        builder = ChunkContextBuilder()
        chunks = [
            {"file_path": "/src/a.py", "content": "x" * 1000}
            for _ in range(5)
        ]

        # Small per-file budget
        result = builder.build_file_grouped_context(chunks, max_tokens=50)

        # Should have truncation
        assert "(truncated)" in result["/src/a.py"]

    def test_include_imports_false(self):
        """Should skip imports when include_imports is False."""
        mock_import_service = MagicMock()
        mock_import_service.get_file_imports.return_value = ["import os"]

        builder = ChunkContextBuilder(import_context_service=mock_import_service)
        chunks = [
            {"file_path": "/src/main.py", "content": "def main(): pass"}
        ]

        result = builder.build_file_grouped_context(
            chunks, max_tokens=1000, include_imports=False
        )

        assert "# Imports:" not in result["/src/main.py"]
        mock_import_service.get_file_imports.assert_not_called()
