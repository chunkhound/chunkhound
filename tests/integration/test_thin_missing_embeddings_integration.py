"""Integration-style tests for thin missing-embeddings mode at the service layer.

These tests exercise DirectoryIndexingService with a real IndexingCoordinator
instance while monkeypatching coordinator methods to control outcomes, avoiding
heavy parse/DB work but preserving the orchestration flow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Avoid importing chunkhound.services package (which imports heavy dependencies)
import importlib.util as _ilu
from pathlib import Path as _P

_CODE_PATH = _P(__file__).resolve().parents[2] / "chunkhound" / "services" / "directory_indexing_service.py"
_SPEC = _ilu.spec_from_file_location("_dir_indexing_service_int", str(_CODE_PATH))
assert _SPEC and _SPEC.loader
_MOD = _ilu.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)  # type: ignore[arg-type]
DirectoryIndexingService = _MOD.DirectoryIndexingService


class _DummyIndexing:
    def __init__(self) -> None:
        self.include: list[str] = ["**/*.py"]
        self.exclude: list[str] = []
        self.config_file_size_threshold_kb: int = 20


class _DummyConfig:
    def __init__(self) -> None:
        self.indexing = _DummyIndexing()


def test_service_thin_mode_skips_missing_when_no_new_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Coord:
        async def process_directory(self, *args, **kwargs):
            return {
                "status": "success",
                "files_processed": 0,
                "total_chunks": 0,
                "errors": [],
                "skipped": 0,
                "skipped_unchanged": 0,
                "skipped_filtered": 0,
                "cleanup": {"deleted_files": 0, "deleted_chunks": 0},
            }
        async def generate_missing_embeddings(self, *args, **kwargs):
            called["v"] += 1
            return {"status": "complete", "generated": 0}

    called = {"v": 0}
    coordinator = _Coord()
    svc = DirectoryIndexingService(indexing_coordinator=coordinator, config=_DummyConfig())

    # Thin mode + no new chunks: should not invoke missing-embeddings
    import asyncio as _aio
    stats = _aio.run(svc.process_directory(tmp_path, no_embeddings=False, thin_missing_embeddings=True))
    assert stats.embeddings_generated == 0
    assert called["v"] == 0

    # Default mode + no new chunks: should invoke missing-embeddings exactly once
    stats = _aio.run(svc.process_directory(tmp_path, no_embeddings=False, thin_missing_embeddings=False))
    assert stats.embeddings_generated == 0
    assert called["v"] == 1


def test_service_thin_mode_does_not_skip_when_new_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Coord:
        async def process_directory(self, *args, **kwargs):
            return {
                "status": "success",
                "files_processed": 1,
                "total_chunks": 3,
                "errors": [],
                "skipped": 0,
                "skipped_unchanged": 0,
                "skipped_filtered": 0,
                "cleanup": {"deleted_files": 0, "deleted_chunks": 0},
            }

        async def generate_missing_embeddings(self, *args, **kwargs):
            called["v"] += 1
            return {"status": "complete", "generated": 0}

    called = {"v": 0}
    coordinator = _Coord()
    svc = DirectoryIndexingService(indexing_coordinator=coordinator, config=_DummyConfig())

    # Thin mode + new chunks: should still invoke missing-embeddings
    import asyncio as _aio
    stats = _aio.run(svc.process_directory(tmp_path, no_embeddings=False, thin_missing_embeddings=True))
    assert stats.embeddings_generated == 0
    assert called["v"] == 1
