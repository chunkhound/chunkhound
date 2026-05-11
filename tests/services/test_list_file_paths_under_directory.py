"""Provider-level tests for list_file_paths_under_directory enumeration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chunkhound.core.models import File
from chunkhound.core.types.common import Language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider

try:
    from chunkhound.providers.database.lancedb_provider import LanceDBProvider

    LANCEDB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    LANCEDB_AVAILABLE = False


def _seed(provider: Any, paths: list[str]) -> None:
    for path in paths:
        provider.insert_file(
            File(
                path=path,
                mtime=0.0,
                language=Language.PYTHON,
                size_bytes=0,
            )
        )


def test_duckdb_list_file_paths_returns_chunkless_rows(tmp_path: Path) -> None:
    base = tmp_path / "repo"
    base.mkdir()
    db_path = tmp_path / "list.duckdb"
    provider = DuckDBProvider(db_path, base_directory=base)
    provider.connect()
    try:
        _seed(
            provider,
            [
                "sub/a.txt",
                "sub/b.bin",
                "sub/nested/c.py",
                "other/d.txt",
                "sub",
            ],
        )

        results = provider.list_file_paths_under_directory("sub")
        assert sorted(results) == sorted(
            ["sub", "sub/a.txt", "sub/b.bin", "sub/nested/c.py"]
        )
        assert "other/d.txt" not in results
    finally:
        provider.disconnect()


def test_duckdb_list_file_paths_handles_empty_prefix_match(tmp_path: Path) -> None:
    base = tmp_path / "repo"
    base.mkdir()
    db_path = tmp_path / "list_empty.duckdb"
    provider = DuckDBProvider(db_path, base_directory=base)
    provider.connect()
    try:
        _seed(provider, ["other/x.py"])
        results = provider.list_file_paths_under_directory("missing")
        assert results == []
    finally:
        provider.disconnect()


def test_duckdb_list_file_paths_does_not_match_partial_segment(tmp_path: Path) -> None:
    base = tmp_path / "repo"
    base.mkdir()
    db_path = tmp_path / "list_partial.duckdb"
    provider = DuckDBProvider(db_path, base_directory=base)
    provider.connect()
    try:
        _seed(provider, ["sub/a.py", "subway/b.py"])
        results = provider.list_file_paths_under_directory("sub")
        assert "sub/a.py" in results
        assert "subway/b.py" not in results
    finally:
        provider.disconnect()


def test_duckdb_list_file_paths_escapes_underscore_metacharacter(
    tmp_path: Path,
) -> None:
    """A literal ``_`` in the prefix must not match any single character."""
    base = tmp_path / "repo"
    base.mkdir()
    db_path = tmp_path / "list_under.duckdb"
    provider = DuckDBProvider(db_path, base_directory=base)
    provider.connect()
    try:
        _seed(
            provider,
            [
                "sub_dir/a.py",
                "sub_dir/inner/b.py",
                "subXdir/c.py",  # would overmatch under raw LIKE ``sub_dir/%``
                "sub_dirmate/e.py",
            ],
        )

        results = provider.list_file_paths_under_directory("sub_dir")
        assert sorted(results) == ["sub_dir/a.py", "sub_dir/inner/b.py"]
        assert "subXdir/c.py" not in results
        assert "sub_dirmate/e.py" not in results
    finally:
        provider.disconnect()


def test_duckdb_list_file_paths_escapes_percent_metacharacter(
    tmp_path: Path,
) -> None:
    """A literal ``%`` in the prefix must be treated as the character ``%``."""
    base = tmp_path / "repo"
    base.mkdir()
    db_path = tmp_path / "list_pct.duckdb"
    provider = DuckDBProvider(db_path, base_directory=base)
    provider.connect()
    try:
        _seed(
            provider,
            [
                "sub%dir/a.py",
                "subXdir/b.py",
                "sub%dir/nested/c.py",
            ],
        )

        results = provider.list_file_paths_under_directory("sub%dir")
        assert sorted(results) == ["sub%dir/a.py", "sub%dir/nested/c.py"]
        assert "subXdir/b.py" not in results
    finally:
        provider.disconnect()


@pytest.mark.skipif(not LANCEDB_AVAILABLE, reason="lancedb not installed")
def test_lancedb_list_file_paths_returns_chunkless_rows(tmp_path: Path) -> None:
    """LanceDB provider must implement the same enumeration contract."""
    base = tmp_path / "repo"
    base.mkdir()
    db_path = tmp_path / "list.lancedb"
    provider = LanceDBProvider(db_path, base)
    provider.connect()
    try:
        _seed(
            provider,
            [
                "sub/a.txt",
                "sub/b.bin",
                "sub/nested/c.py",
                "other/d.txt",
            ],
        )

        results = provider.list_file_paths_under_directory("sub")
        assert sorted(results) == sorted(
            ["sub/a.txt", "sub/b.bin", "sub/nested/c.py"]
        )
        assert "other/d.txt" not in results
    finally:
        provider.disconnect()


@pytest.mark.skipif(not LANCEDB_AVAILABLE, reason="lancedb not installed")
def test_lancedb_list_file_paths_escapes_metacharacters(tmp_path: Path) -> None:
    """LanceDB impl must not overmatch on ``_`` or ``%`` in directory names."""
    base = tmp_path / "repo"
    base.mkdir()
    db_path = tmp_path / "list_meta.lancedb"
    provider = LanceDBProvider(db_path, base)
    provider.connect()
    try:
        _seed(
            provider,
            [
                "sub_dir/a.py",
                "subXdir/b.py",
                "sub%dir/c.py",
                "subYdir/d.py",
            ],
        )

        under_score = provider.list_file_paths_under_directory("sub_dir")
        assert under_score == ["sub_dir/a.py"]
        assert "subXdir/b.py" not in under_score

        percent = provider.list_file_paths_under_directory("sub%dir")
        assert percent == ["sub%dir/c.py"]
        assert "subYdir/d.py" not in percent
    finally:
        provider.disconnect()
