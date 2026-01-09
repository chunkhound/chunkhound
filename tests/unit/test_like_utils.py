"""Unit tests for SQL LIKE escaping utilities."""

from chunkhound.providers.database.like_utils import escape_like_pattern


def test_escape_like_pattern_parameterized_keeps_quotes() -> None:
    escaped = escape_like_pattern("scope_%[path]'\\name", escape_quotes=False)
    assert "\\%" in escaped
    assert "\\_" in escaped
    assert "\\[" in escaped
    assert "\\\\" in escaped
    assert "''" not in escaped
    assert "'" in escaped


def test_escape_like_pattern_interpolated_escapes_quotes() -> None:
    escaped = escape_like_pattern("scope_%[path]'\\name", escape_quotes=True)
    assert "\\%" in escaped
    assert "\\_" in escaped
    assert "\\[" in escaped
    assert "\\\\" in escaped
    assert "''" in escaped
