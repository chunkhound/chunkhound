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

    def test_with_single_constant(self):
        """Should append single constant to header."""
        result = format_chunk_for_embedding(
            code="MAX = 100",
            file_path="config.py",
            language="python",
            constants=[{"name": "MAX", "value": "100"}],
        )
        assert result == "# config.py (python) [MAX=100]\nMAX = 100"

    def test_with_multiple_constants(self):
        """Should append multiple constants to header."""
        result = format_chunk_for_embedding(
            code="A = 1\nB = 2\nC = 3",
            file_path="config.py",
            language="python",
            constants=[
                {"name": "A", "value": "1"},
                {"name": "B", "value": "2"},
                {"name": "C", "value": "3"},
            ],
        )
        assert result == "# config.py (python) [A=1, B=2, C=3]\nA = 1\nB = 2\nC = 3"

    def test_with_constants_truncation_at_5(self):
        """Should truncate constants after 5 items with overflow indicator."""
        constants = [{"name": f"CONST_{i}", "value": str(i)} for i in range(7)]
        result = format_chunk_for_embedding(
            code="code",
            file_path="config.py",
            language="python",
            constants=constants,
        )
        # Should include first 5 constants and overflow indicator
        assert "CONST_0=0" in result
        assert "CONST_4=4" in result
        assert "+2 more" in result
        assert "CONST_5" not in result  # Should be truncated
        assert "CONST_6" not in result  # Should be truncated

    def test_with_no_constants(self):
        """Should preserve original behavior when constants is None."""
        result = format_chunk_for_embedding(
            code="def foo(): pass",
            file_path="src/main.py",
            language="python",
            constants=None,
        )
        assert result == "# src/main.py (python)\ndef foo(): pass"

    def test_with_empty_constants_list(self):
        """Should preserve original behavior when constants is empty list."""
        result = format_chunk_for_embedding(
            code="def foo(): pass",
            file_path="src/main.py",
            language="python",
            constants=[],
        )
        assert result == "# src/main.py (python)\ndef foo(): pass"

    def test_with_constants_only(self):
        """Should create header with only constants when no file/language provided."""
        result = format_chunk_for_embedding(
            code="MAX = 100",
            constants=[{"name": "MAX", "value": "100"}],
        )
        assert result == "# [MAX=100]\nMAX = 100"

    def test_with_constants_and_path_only(self):
        """Should combine path and constants without language."""
        result = format_chunk_for_embedding(
            code="MAX = 100",
            file_path="config.py",
            constants=[{"name": "MAX", "value": "100"}],
        )
        assert result == "# config.py [MAX=100]\nMAX = 100"

    def test_with_constants_and_language_only(self):
        """Should combine language and constants without path."""
        result = format_chunk_for_embedding(
            code="MAX = 100",
            language="python",
            constants=[{"name": "MAX", "value": "100"}],
        )
        assert result == "# (python) [MAX=100]\nMAX = 100"

    def test_constants_with_complex_values(self):
        """Should handle constants with complex string values."""
        result = format_chunk_for_embedding(
            code='URL = "https://api.example.com"',
            file_path="config.py",
            language="python",
            constants=[{"name": "URL", "value": '"https://api.example.com"'}],
        )
        assert result == '# config.py (python) [URL="https://api.example.com"]\nURL = "https://api.example.com"'

    def test_constants_edge_case_exactly_5(self):
        """Should not add overflow indicator when exactly 5 constants."""
        constants = [{"name": f"C{i}", "value": str(i)} for i in range(5)]
        result = format_chunk_for_embedding(
            code="code",
            file_path="config.py",
            language="python",
            constants=constants,
        )
        assert "[C0=0, C1=1, C2=2, C3=3, C4=4]" in result
        assert "more" not in result

    def test_constants_edge_case_exactly_6(self):
        """Should add overflow indicator when exactly 6 constants."""
        constants = [{"name": f"C{i}", "value": str(i)} for i in range(6)]
        result = format_chunk_for_embedding(
            code="code",
            file_path="config.py",
            language="python",
            constants=constants,
        )
        assert "C0=0" in result
        assert "C4=4" in result
        assert "+1 more" in result
        assert "C5" not in result
