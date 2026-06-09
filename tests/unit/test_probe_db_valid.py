"""Tests for DuckDBConnectionManager._probe_db_valid and _escape_path_for_sql.

Covers:
- _probe_db_valid succeeds on a real DB with data (exercises COUNT(*) queries)
- _probe_db_valid returns False when duckdb.connect raises
- _escape_path_for_sql security validation (new checks added in PR review)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from chunkhound.core.exceptions.core import CompactionError
from chunkhound.providers.database.duckdb.connection_manager import (
    DuckDBConnectionManager,
    _escape_path_for_sql,
)


class TestProbeDbValidCountQueries:
    """_probe_db_valid must run COUNT(*) queries against required tables."""

    def test_probe_db_valid_on_real_db_with_data(self, tmp_path: Path) -> None:
        """Probe succeeds on a real DB that has both required tables with rows."""
        db_path = tmp_path / "valid_with_data.duckdb"
        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE TABLE files (id INTEGER, path TEXT)")
        conn.execute("CREATE TABLE chunks (id INTEGER, code TEXT)")
        conn.execute("INSERT INTO files VALUES (1, '/tmp/foo.py')")
        conn.execute("INSERT INTO chunks VALUES (1, 'def foo(): pass')")
        conn.close()

        mgr = DuckDBConnectionManager(db_path)
        assert mgr._probe_db_valid(db_path) is True

    def test_probe_db_valid_returns_false_on_exception(self, tmp_path: Path) -> None:
        """Probe returns False when duckdb.connect raises (e.g. lock error)."""
        db_path = tmp_path / "unreachable.duckdb"

        mgr = DuckDBConnectionManager(db_path)
        with patch("duckdb.connect", side_effect=RuntimeError("simulated lock error")):
            result = mgr._probe_db_valid(db_path)

        assert result is False


class TestEscapePathForSql:
    """_escape_path_for_sql must escape safe chars and reject dangerous ones."""

    def test_escape_path_for_sql_doubles_single_quotes(self) -> None:
        """Single quotes in path are doubled per SQL literal convention."""
        path = Path("/tmp/it's a path")
        result = _escape_path_for_sql(path)
        assert result == "/tmp/it''s a path"

    def test_escape_path_for_sql_rejects_newline(self) -> None:
        """Path with newline raises CompactionError."""
        # Use str construction so the newline survives Path normalization on all OS
        path_str = "/tmp/foo\nbar"
        with pytest.raises(CompactionError, match="forbidden"):
            _escape_path_for_sql(Path(path_str))

    def test_escape_path_for_sql_rejects_semicolon(self) -> None:
        """Path with semicolon raises CompactionError (SQL injection guard)."""
        with pytest.raises(CompactionError, match="forbidden"):
            _escape_path_for_sql(Path("/tmp/foo;bar"))

    def test_escape_path_for_sql_rejects_null_byte(self) -> None:
        """Path with null byte raises CompactionError."""
        with pytest.raises(CompactionError, match="forbidden"):
            _escape_path_for_sql(Path("/tmp/foo\x00bar"))

    def test_escape_path_for_sql_accepts_unicode_path(self) -> None:
        """Path with unicode characters is accepted and returned as POSIX string."""
        path = Path("/home/müller/.chunkhound/db")
        result = _escape_path_for_sql(path)
        assert result == "/home/müller/.chunkhound/db"

    def test_escape_path_for_sql_accepts_cjk_path(self) -> None:
        """Path with CJK characters is accepted."""
        path = Path("/代码/db.duckdb")
        result = _escape_path_for_sql(path)
        assert result == "/代码/db.duckdb"

    def test_escape_path_for_sql_error_has_operation_preflight(self) -> None:
        """CompactionError from escape has operation='preflight'."""
        with pytest.raises(CompactionError) as exc_info:
            _escape_path_for_sql(Path("/tmp/bad;path"))
        assert exc_info.value.operation == "preflight"
