"""Contract: orphan cleanup continues when some batch entries are absent from the DB."""

import asyncio
import pytest
from pathlib import Path

from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, FileId, Language, LineNumber
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator


def test_delete_files_batch_returns_partial_count_for_absent_files(
    tmp_path: Path,
) -> None:
    """delete_files_batch returns 1 when one of two paths is absent from files table."""
    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()

    fid = provider.insert_file(
        File(path="file_a.py", mtime=1.0, size_bytes=10, language=Language.PYTHON)
    )
    provider.insert_chunk(
        Chunk(
            file_id=FileId(fid),
            symbol="fn",
            start_line=LineNumber(1),
            end_line=LineNumber(2),
            code="def fn(): pass\n",
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
        )
    )

    deleted = provider.delete_files_batch(["file_a.py", "ghost_file.py"])
    assert deleted == 1, "Expected 1 successful delete (ghost absent → returns False)"


def test_process_directory_warns_on_partial_orphan_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IndexingCoordinator.process_directory must not raise RuntimeError
    when delete_files_batch returns fewer deletes than batch size.

    Setup: two files inserted in the DB but NOT on disk (orphans).
    Patch: delete_files_batch returns n-1, simulating one entry already absent.
    Pre-fix behaviour: RuntimeError (deleted_count=1 != batch_count=2).
    Post-fix behaviour: warning logged, indexing succeeds.
    """
    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()

    # Insert two orphaned entries — present in DB, absent on disk
    for name in ["orphan1.py", "orphan2.py"]:
        provider.insert_file(
            File(path=name, mtime=1.0, size_bytes=10, language=Language.PYTHON)
        )

    # One real file on disk so discovery has something to scan
    (tmp_path / "real.py").write_text("x = 1\n")

    delete_batch_calls: list[list[str]] = []
    original_delete_batch = provider.delete_files_batch

    def patched_delete_batch(paths: list[str]) -> int:
        delete_batch_calls.append(list(paths))
        result = original_delete_batch(paths)
        # Simulate one entry in the batch that was already absent from the DB
        return max(0, result - 1) if paths else 0

    monkeypatch.setattr(provider, "delete_files_batch", patched_delete_batch)

    coordinator = IndexingCoordinator(provider, tmp_path)

    result = asyncio.run(
        coordinator.process_directory(
            tmp_path, patterns=["**/*.py"], exclude_patterns=[], force_reindex=False
        )
    )

    # Cleanup path was actually exercised (patch was called)
    assert delete_batch_calls, "delete_files_batch was never called — orphans were not detected"
    assert result.get("status") in ("success",), f"Unexpected status: {result}"
