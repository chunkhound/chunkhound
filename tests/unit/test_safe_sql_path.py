"""Tests for DuckDBProvider._safe_sql_path allowlist validation."""

import sys
from pathlib import Path, PurePosixPath

import pytest

from chunkhound.core.exceptions.core import CompactionError
from chunkhound.providers.database.duckdb_provider import DuckDBProvider


class TestSafeSqlPath:
    """SQL path allowlist must reject anything outside known-safe characters."""

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
        ],
    )
    def test_valid_paths_pass(self, path_str: str) -> None:
        result = DuckDBProvider._safe_sql_path(Path(path_str))
        assert result == Path(path_str).as_posix()

    @pytest.mark.parametrize(
        "path_str",
        [
            "/tmp/foo; DROP TABLE x",
            "/tmp/foo`cmd`",
            "/tmp/foo\x00bar",
            "/tmp/foo'bar",
            "/tmp/\u00e9",
            '/tmp/foo"bar',
            "/tmp/$HOME",
            "/tmp/foo:bar",
        ],
        ids=[
            "semicolon",
            "backtick",
            "null-byte",
            "single-quote",
            "unicode",
            "double-quote",
            "dollar-sign",
            "colon-non-drive-letter",
        ],
    )
    def test_rejected_paths_raise(self, path_str: str) -> None:
        with pytest.raises(CompactionError, match="not allowed in SQL"):
            DuckDBProvider._safe_sql_path(Path(path_str))

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Windows Path treats \\ as separator; backslash cannot reach as_posix()",
    )
    def test_backslash_rejected(self) -> None:
        """On Unix, backslash is a valid filename char and must be rejected."""
        with pytest.raises(CompactionError, match="not allowed in SQL"):
            DuckDBProvider._safe_sql_path(PurePosixPath("/tmp/foo\\bar"))

    def test_rejected_error_has_operation(self) -> None:
        with pytest.raises(CompactionError) as exc_info:
            DuckDBProvider._safe_sql_path(Path("/tmp/foo;bar"))
        assert exc_info.value.operation == "validation"
