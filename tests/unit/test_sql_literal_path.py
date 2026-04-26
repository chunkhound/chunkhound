"""Tests for DuckDBProvider._sql_literal_path escaping and validation.

The helper accepts arbitrary Unicode (so users with accented usernames or
CJK locales can run compaction) while rejecting characters that can break
DuckDB SQL string literals or embed commands. Single quotes are escaped
by doubling, per the SQL literal convention.
"""

import sys
from pathlib import Path, PurePosixPath

import pytest

from chunkhound.core.exceptions.core import CompactionError
from chunkhound.providers.database.duckdb_provider import DuckDBProvider


class TestSqlLiteralPath:
    """SQL literal path helper accepts Unicode and rejects injection-dangerous chars."""

    @pytest.mark.parametrize(
        "path_str",
        [
            "/tmp/foo",
            "C:/Users/bar",
            "/path with spaces/db",
            "a.b+c,d=e",
            "/normal/path/to/db.duckdb",
            "relative/path",
            "-flag-like-path",
            # Unicode paths must be accepted
            "/Users/José/project",
            "/代码/db",
            "/home/müller/.chunkhound",
            "/tmp/\u00e9",
            # Colon is legal outside Windows drive prefix (e.g. in filenames)
            "/tmp/foo:bar",
            # Dollar sign — not SQL-dangerous
            "/tmp/$HOME",
        ],
    )
    def test_valid_paths_pass_unchanged(self, path_str: str) -> None:
        result = DuckDBProvider._sql_literal_path(Path(path_str))
        assert result == Path(path_str).as_posix()

    def test_single_quote_escaped_by_doubling(self) -> None:
        """Single quotes are escaped via SQL literal doubling, not rejected."""
        result = DuckDBProvider._sql_literal_path(Path("/tmp/it's"))
        assert result == "/tmp/it''s"

    def test_multiple_single_quotes_all_doubled(self) -> None:
        result = DuckDBProvider._sql_literal_path(Path("/tmp/a'b'c"))
        assert result == "/tmp/a''b''c"

    @pytest.mark.parametrize(
        "path_str",
        [
            "/tmp/foo; DROP TABLE x",
            "/tmp/foo`cmd`",
            "/tmp/foo\x00bar",
            '/tmp/foo"bar',
            "/tmp/foo\nbar",
            "/tmp/foo\rbar",
        ],
        ids=[
            "semicolon",
            "backtick",
            "null-byte",
            "double-quote",
            "newline",
            "carriage-return",
        ],
    )
    def test_rejected_paths_raise(self, path_str: str) -> None:
        with pytest.raises(CompactionError, match="forbidden"):
            DuckDBProvider._sql_literal_path(Path(path_str))

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Windows Path treats \\ as separator; backslash cannot reach as_posix()",
    )
    def test_backslash_rejected(self) -> None:
        """On Unix, backslash is a valid filename char and must be rejected."""
        with pytest.raises(CompactionError, match="forbidden"):
            DuckDBProvider._sql_literal_path(PurePosixPath("/tmp/foo\\bar"))

    def test_rejected_error_has_operation(self) -> None:
        with pytest.raises(CompactionError) as exc_info:
            DuckDBProvider._sql_literal_path(Path("/tmp/foo;bar"))
        assert exc_info.value.operation == "validation"
