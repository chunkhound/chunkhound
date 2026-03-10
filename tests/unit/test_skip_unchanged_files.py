import asyncio
import os
import sys
import types
from pathlib import Path

import pytest


def _install_parser_stubs():
    """Install lightweight stubs to avoid importing heavy tree-sitter modules in tests."""
    # Stub for chunkhound.parsers.universal_parser
    up = types.ModuleType("chunkhound.parsers.universal_parser")
    class _UniversalParser:  # pragma: no cover - only to satisfy type hints
        pass
    up.UniversalParser = _UniversalParser
    sys.modules["chunkhound.parsers.universal_parser"] = up

    # Stub for chunkhound.parsers.parser_factory
    pf = types.ModuleType("chunkhound.parsers.parser_factory")
    def create_parser_for_language(language):  # pragma: no cover - not used in this test
        class _DummyParser:
            def parse_file(self, file_path, file_id):
                return []
        return _DummyParser()
    pf.create_parser_for_language = create_parser_for_language
    sys.modules["chunkhound.parsers.parser_factory"] = pf


class _FakeDB:
    """Minimal DatabaseProvider stub that serves File records by path."""
    def __init__(self, records):
        # records: dict[str, FileModel]
        self._records = records

    # Methods used by IndexingCoordinator in this test path
    def get_file_by_path(self, path: str, as_model: bool = False):
        rec = self._records.get(path)
        if not rec:
            return None
        return rec if as_model else {
            "id": rec.id,
            "path": rec.path,
            "size": rec.size_bytes,
            "modified_time": rec.mtime,
            "language": rec.language.value,
            "content_hash": rec.content_hash,
        }

    def should_optimize(self, operation: str = "") -> bool:
        return False


class _Cfg:
    class _Indexing:
        cleanup = False
        force_reindex = False
        per_file_timeout_seconds = 0.0
        # Discovery config used by coordinator
        min_dirs_for_parallel = 4
        max_discovery_workers = 4
        parallel_discovery = False  # keep sequential for test stability

    indexing = _Indexing()


def test_process_directory_skips_unchanged_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CHUNKHOUND_DEBUG_SKIP", "1")
    # Install parser stubs before importing the coordinator to avoid heavy deps
    _install_parser_stubs()

    # Import after stubbing
    from chunkhound.core.models.file import File
    from chunkhound.core.types.common import Language
    from chunkhound.services.indexing_coordinator import IndexingCoordinator

    # Create some files
    paths = []
    for i in range(3):
        p = tmp_path / f"f{i}.txt"
        p.write_text("hello")
        paths.append(p)

    # Build DB records with same size+mtime (simulate previous index)
    records = {}
    for p in paths:
        st = p.stat()
        rel = p.relative_to(tmp_path).as_posix()
        records[rel] = File(
            path=rel,  # stored as relative
            mtime=float(st.st_mtime),
            language=Language.TEXT,
            size_bytes=int(st.st_size),
        )

    db = _FakeDB(records)
    coord = IndexingCoordinator(database_provider=db, base_directory=tmp_path, config=_Cfg())

    # Ensure we don't accidentally parse anything: capture files list used
    called_batches = []

    async def _fake_batches(files, config_file_size_threshold_kb=20, parse_task=None, on_batch=None):
        # Record what coordinator attempted to parse
        called_batches.append(list(files))
        return []

    monkeypatch.setattr(coord, "_process_files_in_batches", _fake_batches)

    # Run
    result = asyncio.run(
        coord.process_directory(
            tmp_path, patterns=["**/*.txt"], exclude_patterns=[], config_file_size_threshold_kb=20
        )
    )

    # Assert: no files processed, all skipped as unchanged
    assert result["files_processed"] == 0
    assert result.get("skipped_unchanged", 0) == 3
    # And we never attempted to parse any file
    assert called_batches and len(called_batches[0]) == 0
