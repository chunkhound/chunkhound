"""LanceDB parity and state-transition tests for skipped-file recording (issue #297).

Covers:
- LanceDB records skipped files the same way DuckDB does
- indexed → skipped: existing file id is preserved (chunks not orphaned)
- skipped → indexed: skip_reason is cleared on re-index
"""
import asyncio
from pathlib import Path

import pytest


# Use explicit extensions rather than **/* — LanceDB stores its database under
# tmp_path/lancedb.lancedb/ which would otherwise be scanned as unknown-type files.
_PY_PATTERNS = ["**/*.py"]
_ALL_PATTERNS = ["**/*.py", "**/*.xyzunk"]


@pytest.fixture
def coordinator(lancedb_provider, tmp_path):
    from chunkhound.core.types.common import Language
    from chunkhound.parsers.parser_factory import create_parser_for_language
    from chunkhound.services.indexing_coordinator import IndexingCoordinator

    parser = create_parser_for_language(Language.PYTHON)
    return IndexingCoordinator(
        lancedb_provider, tmp_path, None, {Language.PYTHON: parser}
    )


# ---------------------------------------------------------------------------
# LanceDB parity: basic skip recording
# ---------------------------------------------------------------------------


def test_lancedb_skipped_file_recorded(coordinator, tmp_path):
    """Skipped file gets a DB record with skip_reason in LanceDB (parity with DuckDB)."""
    (tmp_path / "data.xyzunk").write_text("binary\n")

    asyncio.run(coordinator.process_directory(tmp_path, patterns=_ALL_PATTERNS))

    rows = coordinator._db.execute_query(
        "SELECT path, skip_reason FROM files WHERE skip_reason IS NOT NULL"
    )
    assert rows, "Skipped file should be recorded in the LanceDB files table"
    assert rows[0]["skip_reason"] == "Unknown file type"


def test_lancedb_skipped_file_not_reparsed_on_second_run(coordinator, tmp_path):
    """Previously-skipped file must not be re-queued on second run (LanceDB parity)."""
    (tmp_path / "main.py").write_text("def hello(): pass\n")
    (tmp_path / "data.xyzunk").write_text("binary\n")

    result1 = asyncio.run(coordinator.process_directory(tmp_path, patterns=_ALL_PATTERNS))
    assert result1["status"] == "success"
    assert result1.get("skipped_filtered", 0) >= 1

    result2 = asyncio.run(coordinator.process_directory(tmp_path, patterns=_ALL_PATTERNS))
    assert result2["status"] == "success"
    assert result2.get("skipped_filtered", 0) == 0, (
        "Skipped file must not be re-parsed on run 2 in LanceDB"
    )
    assert result2.get("skipped_unchanged", 0) >= 1


# ---------------------------------------------------------------------------
# State transition: indexed → skipped preserves file id
# ---------------------------------------------------------------------------


def test_indexed_to_skipped_preserves_file_id(lancedb_provider, tmp_path):
    """record_skipped_file on a previously-indexed path must keep the same file id.

    If the id changes, existing chunks.file_id rows become orphaned.
    """
    from chunkhound.core.types.common import Language
    from chunkhound.parsers.parser_factory import create_parser_for_language
    from chunkhound.services.indexing_coordinator import IndexingCoordinator

    pyfile = tmp_path / "code.py"
    pyfile.write_text("x = 1\n")

    parser = create_parser_for_language(Language.PYTHON)
    coord = IndexingCoordinator(
        lancedb_provider, tmp_path, None, {Language.PYTHON: parser}
    )
    asyncio.run(coord.process_directory(tmp_path, patterns=_PY_PATTERNS))

    rel = "code.py"
    file_record_before = lancedb_provider.get_file_by_path(rel, as_model=False)
    assert file_record_before is not None
    original_id = file_record_before["id"]

    chunks_before = lancedb_provider.get_chunks_by_file_id(original_id)
    assert len(chunks_before) > 0, "Should have chunks after indexing"

    # Simulate the file becoming unrecognised/binary — record it as skipped
    lancedb_provider.record_skipped_file(
        rel, pyfile.name, pyfile.suffix,
        pyfile.stat().st_size, pyfile.stat().st_mtime,
        None, None, "test skip reason",
    )

    file_record_after = lancedb_provider.get_file_by_path(rel, as_model=False)
    assert file_record_after is not None
    assert file_record_after["id"] == original_id, (
        "File id must not change when transitioning indexed → skipped; "
        "changing it orphans existing chunks"
    )
    assert file_record_after["skip_reason"] == "test skip reason"

    chunks_after = lancedb_provider.get_chunks_by_file_id(original_id)
    assert len(chunks_after) == len(chunks_before), (
        "Chunks must not be orphaned after indexed → skipped transition"
    )


# ---------------------------------------------------------------------------
# State transition: skipped → indexed clears skip_reason
# ---------------------------------------------------------------------------


def test_skipped_to_indexed_clears_skip_reason(lancedb_provider, tmp_path):
    """Re-indexing a file whose skip record has a stale mtime must clear skip_reason.

    We record a skip with mtime=0 (deliberately stale) so change-detection
    re-queues the file on the next process_directory call.
    """
    from chunkhound.core.types.common import Language
    from chunkhound.parsers.parser_factory import create_parser_for_language
    from chunkhound.services.indexing_coordinator import IndexingCoordinator

    pyfile = tmp_path / "code.py"
    pyfile.write_text("x = 1\n")

    rel = "code.py"
    # Record skip with stale mtime=0 so change detection will re-queue the file
    lancedb_provider.record_skipped_file(
        rel, pyfile.name, pyfile.suffix,
        pyfile.stat().st_size, 0.0,  # mtime=0 forces mtime mismatch
        None, None, "old skip reason",
    )

    pre = lancedb_provider.get_file_by_path(rel, as_model=False)
    assert pre is not None
    assert pre["skip_reason"] == "old skip reason"

    parser = create_parser_for_language(Language.PYTHON)
    coord = IndexingCoordinator(
        lancedb_provider, tmp_path, None, {Language.PYTHON: parser}
    )
    asyncio.run(coord.process_directory(tmp_path, patterns=_PY_PATTERNS))

    post = lancedb_provider.get_file_by_path(rel, as_model=False)
    assert post is not None
    assert post["skip_reason"] is None, (
        "skip_reason must be cleared after successful re-indexing"
    )
