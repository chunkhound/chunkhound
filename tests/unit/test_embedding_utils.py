"""Tests for embedding utilities."""

import pytest

from chunkhound.core.utils.embedding_utils import format_chunk_for_embedding


class TestFormatChunkForEmbedding:
    """Tests for format_chunk_for_embedding function."""

    def test_with_path_and_language(self):
        """Should prepend path and language header."""
        result = format_chunk_for_embedding(
            code="def foo(): pass",
            file_path="src/main.py",
            language="python",
        )
        assert result == "# src/main.py (python)\ndef foo(): pass"

    def test_with_path_only(self):
        """Should prepend path header without language."""
        result = format_chunk_for_embedding(
            code="def foo(): pass",
            file_path="src/main.py",
        )
        assert result == "# src/main.py\ndef foo(): pass"

    def test_with_language_only(self):
        """Should prepend language header without path."""
        result = format_chunk_for_embedding(
            code="def foo(): pass",
            language="python",
        )
        assert result == "# (python)\ndef foo(): pass"

    def test_with_neither(self):
        """Should return code unchanged when no metadata provided."""
        result = format_chunk_for_embedding(code="def foo(): pass")
        assert result == "def foo(): pass"

    def test_with_empty_strings(self):
        """Should treat empty strings as missing metadata."""
        result = format_chunk_for_embedding(
            code="def foo(): pass",
            file_path="",
            language="",
        )
        assert result == "def foo(): pass"

    def test_with_none_values(self):
        """Should handle None values explicitly."""
        result = format_chunk_for_embedding(
            code="def foo(): pass",
            file_path=None,
            language=None,
        )
        assert result == "def foo(): pass"

    def test_preserves_multiline_code(self):
        """Should preserve multiline code content."""
        code = """def foo():
    x = 1
    return x"""
        result = format_chunk_for_embedding(
            code=code,
            file_path="src/main.py",
            language="python",
        )
        expected = f"# src/main.py (python)\n{code}"
        assert result == expected

    def test_with_nested_path(self):
        """Should handle deeply nested paths."""
        result = format_chunk_for_embedding(
            code="class Handler:",
            file_path="src/services/auth/handlers/oauth.py",
            language="python",
        )
        assert result == "# src/services/auth/handlers/oauth.py (python)\nclass Handler:"

    def test_with_various_languages(self):
        """Should work with different language values."""
        languages = ["python", "typescript", "javascript", "go", "rust"]
        for lang in languages:
            result = format_chunk_for_embedding(
                code="// code",
                file_path="file.txt",
                language=lang,
            )
            assert f"({lang})" in result
