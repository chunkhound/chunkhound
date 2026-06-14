"""Unit tests for DuckDB path normalization contract.

Tests the external invariant that paths are always relative unix paths at the
DuckDB boundary:
  - insert_file() preserves normalized forward-slash paths provided by callers
  - get_file_by_path() normalizes lookup input
  - Search results return whatever is stored

The storage contract is enforced by the indexing pipeline, which normalizes
all paths before insertion. Raw SQL inserts bypass this normalization.
"""

from pathlib import Path

import pytest

from chunkhound.core.models import Chunk, Embedding, File
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider

pytest.importorskip("duckdb")


# ---------------------------------------------------------------------------
# Storage contract: insert_file preserves normalized paths
# ---------------------------------------------------------------------------


class TestStorageContract:
    """Normalized paths passed to insert_file() stay normalized in storage."""

    def test_insert_file_stores_forward_slash(self, tmp_path: Path) -> None:
        """insert_file stores paths with forward slashes."""
        provider = DuckDBProvider(
            db_path=tmp_path / "db.duckdb", base_directory=tmp_path
        )
        provider.connect()
        try:
            provider.insert_file(
                File(
                    path="src/utils/helpers.py",
                    mtime=1.0,
                    size_bytes=100,
                    language=Language.PYTHON,
                )
            )
            result = provider.get_file_by_path("src/utils/helpers.py", as_model=False)
            assert result is not None
            assert "\\" not in result["path"]
            assert result["path"] == "src/utils/helpers.py"
        finally:
            provider.disconnect()

    def test_insert_file_with_embedding_returns_forward_slash(
        self, tmp_path: Path
    ) -> None:
        """File + chunk + embedding all use forward-slash paths."""
        provider = DuckDBProvider(
            db_path=tmp_path / "db.duckdb", base_directory=tmp_path
        )
        provider.connect()
        try:
            file_id = provider.insert_file(
                File(
                    path="lib/core/engine.py",
                    mtime=1.0,
                    size_bytes=200,
                    language=Language.PYTHON,
                )
            )
            chunk_id = provider.insert_chunk(
                Chunk(
                    file_id=file_id,
                    symbol="engine",
                    start_line=1,
                    end_line=20,
                    code="class Engine: pass",
                    chunk_type=ChunkType.CLASS,
                    language=Language.PYTHON,
                )
            )
            provider.insert_embedding(
                Embedding(
                    chunk_id=chunk_id,
                    provider="test",
                    model="fake",
                    dims=8,
                    vector=[0.1] * 8,
                )
            )

            # Verify path in search results
            results, _ = provider.search_regex(
                pattern="class", page_size=10, offset=0
            )
            assert len(results) > 0
            for r in results:
                assert "\\" not in r["file_path"]
                assert not r["file_path"].startswith("/")
        finally:
            provider.disconnect()

    def test_multiple_files_all_use_forward_slash(
        self, tmp_path: Path
    ) -> None:
        """Multiple files all stored with forward-slash paths."""
        provider = DuckDBProvider(
            db_path=tmp_path / "db.duckdb", base_directory=tmp_path
        )
        provider.connect()
        try:
            paths = ["src/a.py", "lib/core/b.py", "test/c.py"]
            for p in paths:
                provider.insert_file(
                    File(path=p, mtime=1.0, size_bytes=100, language=Language.PYTHON)
                )

            rows = provider.execute_query("SELECT path FROM files ORDER BY path")
            stored = [r["path"] for r in rows]
            assert stored == ["lib/core/b.py", "src/a.py", "test/c.py"]
            for p in stored:
                assert "\\" not in p
        finally:
            provider.disconnect()


# ---------------------------------------------------------------------------
# Lookup contract: get_file_by_path normalizes input
# ---------------------------------------------------------------------------


class TestLookupContract:
    """get_file_by_path normalizes input paths for lookup."""

    def test_lookup_with_forward_slash(self, tmp_path: Path) -> None:
        """Lookup with forward-slash path finds the file."""
        provider = DuckDBProvider(
            db_path=tmp_path / "db.duckdb", base_directory=tmp_path
        )
        provider.connect()
        try:
            provider.insert_file(
                File(
                    path="src/utils.py",
                    mtime=1.0,
                    size_bytes=100,
                    language=Language.PYTHON,
                )
            )
            result = provider.get_file_by_path("src/utils.py", as_model=False)
            assert result is not None
            assert result["path"] == "src/utils.py"
        finally:
            provider.disconnect()

    def test_lookup_nonexistent_returns_none(self, tmp_path: Path) -> None:
        """Lookup of nonexistent file returns None."""
        provider = DuckDBProvider(
            db_path=tmp_path / "db.duckdb", base_directory=tmp_path
        )
        provider.connect()
        try:
            result = provider.get_file_by_path("nonexistent.py", as_model=False)
            assert result is None
        finally:
            provider.disconnect()

    def test_lookup_with_absolute_native_path(self, tmp_path: Path) -> None:
        """Lookup with an absolute host-native path finds the stored file."""
        provider = DuckDBProvider(
            db_path=tmp_path / "db.duckdb", base_directory=tmp_path
        )
        provider.connect()
        try:
            provider.insert_file(
                File(
                    path="src/utils.py",
                    mtime=1.0,
                    size_bytes=100,
                    language=Language.PYTHON,
                )
            )
            absolute_path = tmp_path / "src" / "utils.py"
            result = provider.get_file_by_path(str(absolute_path), as_model=False)
            assert result is not None
            assert result["path"] == "src/utils.py"
        finally:
            provider.disconnect()


# ---------------------------------------------------------------------------
# Search contract: search results use stored paths
# ---------------------------------------------------------------------------


class TestSearchContract:
    """Search results return paths exactly as stored."""

    def test_regex_search_returns_relative_paths(
        self, tmp_path: Path
    ) -> None:
        """Regex search returns relative forward-slash paths."""
        provider = DuckDBProvider(
            db_path=tmp_path / "db.duckdb", base_directory=tmp_path
        )
        provider.connect()
        try:
            file_id = provider.insert_file(
                File(
                    path="src/engine.py",
                    mtime=1.0,
                    size_bytes=100,
                    language=Language.PYTHON,
                )
            )
            provider.insert_chunk(
                Chunk(
                    file_id=file_id,
                    symbol="engine",
                    start_line=1,
                    end_line=10,
                    code="def run(): pass",
                    chunk_type=ChunkType.FUNCTION,
                    language=Language.PYTHON,
                )
            )

            results, _ = provider.search_regex(
                pattern="def", page_size=10, offset=0
            )
            assert len(results) > 0
            for r in results:
                fp = r["file_path"]
                assert "\\" not in fp, f"Backslash in path: {fp}"
                assert not fp.startswith("/"), f"Absolute path: {fp}"
                assert ".." not in fp.split("/"), f"Traversal in path: {fp}"
        finally:
            provider.disconnect()


# ---------------------------------------------------------------------------
# Contract invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    """Invariants that must hold for all paths in DuckDB."""

    def test_all_stored_paths_are_relative_forward_slash(
        self, tmp_path: Path
    ) -> None:
        """All paths inserted via insert_file are relative with forward slashes."""
        provider = DuckDBProvider(
            db_path=tmp_path / "db.duckdb", base_directory=tmp_path
        )
        provider.connect()
        try:
            paths = [
                "src/a.py",
                "lib/core/b.py",
                "test/deep/nested/c.py",
                ".github/workflows/d.yml",
            ]
            for p in paths:
                provider.insert_file(
                    File(path=p, mtime=1.0, size_bytes=100, language=Language.PYTHON)
                )

            rows = provider.execute_query("SELECT path FROM files")
            for row in rows:
                p = row["path"]
                assert not Path(p).is_absolute(), f"Absolute path: {p}"
                assert "\\" not in p, f"Backslash in path: {p}"
                assert ".." not in p.split("/"), f"Traversal in path: {p}"
        finally:
            provider.disconnect()

    def test_search_results_are_relative_forward_slash(
        self, tmp_path: Path
    ) -> None:
        """All search results have relative forward-slash paths."""
        provider = DuckDBProvider(
            db_path=tmp_path / "db.duckdb", base_directory=tmp_path
        )
        provider.connect()
        try:
            file_id = provider.insert_file(
                File(
                    path="src/module.py",
                    mtime=1.0,
                    size_bytes=100,
                    language=Language.PYTHON,
                )
            )
            provider.insert_chunk(
                Chunk(
                    file_id=file_id,
                    symbol="module",
                    start_line=1,
                    end_line=5,
                    code="def func(): pass",
                    chunk_type=ChunkType.FUNCTION,
                    language=Language.PYTHON,
                )
            )

            results, _ = provider.search_regex(
                pattern="def", page_size=100, offset=0
            )
            for r in results:
                fp = r["file_path"]
                assert not Path(fp).is_absolute(), f"Absolute path: {fp}"
                assert "\\" not in fp, f"Backslash: {fp}"
        finally:
            provider.disconnect()
