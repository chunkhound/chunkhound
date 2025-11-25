"""Unit tests for thin missing-embeddings mode.

These tests verify that DirectoryIndexingService skips the
generate_missing_embeddings phase when:
- The thin mode flag is enabled, and
- No new chunks were created in the current run.

They also verify that:
- Without the flag, the embeddings step is invoked, and
- With the flag but with new chunks created, the step is still invoked.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

# Avoid importing chunkhound.services package (which imports heavy dependencies)
import importlib.util as _ilu
import sys as _sys
from pathlib import Path as _P

_CODE_PATH = _P(__file__).resolve().parents[2] / "chunkhound" / "services" / "directory_indexing_service.py"
_SPEC = _ilu.spec_from_file_location("_dir_indexing_service", str(_CODE_PATH))
assert _SPEC and _SPEC.loader
_MOD = _ilu.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)  # type: ignore[arg-type]
DirectoryIndexingService = _MOD.DirectoryIndexingService


class _DummyIndexing:
    def __init__(self) -> None:
        # Minimal include/exclude required by DirectoryIndexingService
        self.include: list[str] = ["**/*.py"]
        self.exclude: list[str] = []
        # Service reads this threshold and forwards it to coordinator
        self.config_file_size_threshold_kb: int = 20


class _DummyConfig:
    def __init__(self) -> None:
        self.indexing = _DummyIndexing()


class _FakeCoordinator:
    """Minimal stub for the indexing coordinator used by DirectoryIndexingService."""

    def __init__(self, chunks_created: int = 0, files_processed: int = 0) -> None:
        self._chunks_created = int(chunks_created)
        self._files_processed = int(files_processed)
        self.missing_called = False

    async def process_directory(
        self,
        directory: Path,
        patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        config_file_size_threshold_kb: int | float | None = None,
    ) -> dict:
        # Return a minimal result dictionary consistent with coordinator contract
        return {
            "status": "success",
            "files_processed": self._files_processed,
            "total_chunks": self._chunks_created,
            "errors": [],
            "skipped": 0,
            "skipped_unchanged": 0,
            "skipped_filtered": 0,
            "cleanup": {"deleted_files": 0, "deleted_chunks": 0},
        }

    async def generate_missing_embeddings(
        self, exclude_patterns: list[str] | None = None, thin_check: bool = False
    ) -> dict:
        self.missing_called = True
        return {"status": "complete", "generated": 0}


def test_thin_mode_skips_embeddings_when_no_new_chunks(tmp_path: Path) -> None:
    coord = _FakeCoordinator(chunks_created=0, files_processed=0)
    svc = DirectoryIndexingService(indexing_coordinator=coord, config=_DummyConfig())

    import asyncio as _aio
    stats = _aio.run(svc.process_directory(
        tmp_path,
        no_embeddings=False,
        thin_missing_embeddings=True,
    ))

    # The coordinator's missing-embeddings step should not be called
    assert coord.missing_called is False
    # And reported embeddings_generated should be zero
    assert getattr(stats, "embeddings_generated", 0) == 0


def test_default_mode_invokes_embeddings_on_noop(tmp_path: Path) -> None:
    coord = _FakeCoordinator(chunks_created=0, files_processed=0)
    svc = DirectoryIndexingService(indexing_coordinator=coord, config=_DummyConfig())

    import asyncio as _aio
    stats = _aio.run(svc.process_directory(
        tmp_path,
        no_embeddings=False,
        thin_missing_embeddings=False,
    ))

    # Without thin flag, the coordinator should be invoked even on no-op runs
    assert coord.missing_called is True
    assert getattr(stats, "embeddings_generated", 0) == 0


def test_thin_mode_still_embeds_when_new_chunks_present(tmp_path: Path) -> None:
    coord = _FakeCoordinator(chunks_created=3, files_processed=1)
    svc = DirectoryIndexingService(indexing_coordinator=coord, config=_DummyConfig())

    import asyncio as _aio
    _ = _aio.run(svc.process_directory(
        tmp_path,
        no_embeddings=False,
        thin_missing_embeddings=True,
    ))

    # New chunks present -> even thin mode should invoke embeddings
    assert coord.missing_called is True
