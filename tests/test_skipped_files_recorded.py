"""Tests for issue #297: skipped files are recorded in DB to prevent re-scanning.

Contract: a file that is skipped during parsing (binary, unknown type, etc.)
must be persisted in the `files` table with a `skip_reason`.  On the next run,
change-detection finds the record, sees the metadata is unchanged, and does NOT
re-queue the file for parsing — eliminating the repeated re-scan penalty.
"""
import pytest
from pathlib import Path

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator

_PATTERNS = ["**/*"]  # discover all files; binary/unknown filtering happens in batch_processor


@pytest.fixture
def coordinator(tmp_path):
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()
    parser = create_parser_for_language(Language.PYTHON)
    return IndexingCoordinator(db, tmp_path, None, {Language.PYTHON: parser})


@pytest.mark.asyncio
async def test_skipped_file_is_recorded_in_db(coordinator, tmp_path):
    """A file skipped at parse time (unknown extension) gets a DB record with skip_reason."""
    # .xyzunk has no registered parser → Language.UNKNOWN → skip with "Unknown file type"
    unknown = tmp_path / "data.xyzunk"
    unknown.write_text("some content\n")

    await coordinator.process_directory(tmp_path, patterns=_PATTERNS)

    rows = coordinator._db.execute_query(
        "SELECT path, skip_reason FROM files WHERE skip_reason IS NOT NULL"
    )
    assert rows, "Skipped file should be recorded in the files table"
    assert rows[0]["skip_reason"] == "Unknown file type"


@pytest.mark.asyncio
async def test_skipped_file_not_reparsed_on_second_run(coordinator, tmp_path):
    """A previously-skipped file whose content hasn't changed must not be
    re-queued for parsing on subsequent runs (fix for issue #297)."""
    (tmp_path / "main.py").write_text("def hello(): pass\n")
    unknown = tmp_path / "data.xyzunk"
    unknown.write_text("some content\n")

    # Run 1: unknown-extension file is parsed-and-skipped, then recorded in DB.
    result1 = await coordinator.process_directory(tmp_path, patterns=_PATTERNS)
    assert result1["status"] == "success"
    assert result1.get("skipped_filtered", 0) >= 1, (
        "Unknown-type file should be counted as a parse-time skip on run 1"
    )

    # Run 2: no file changes — binary should be caught by change detection,
    # NOT re-sent to the parser.  Parse-time skip count must be 0.
    result2 = await coordinator.process_directory(tmp_path, patterns=_PATTERNS)
    assert result2["status"] == "success"
    assert result2.get("skipped_filtered", 0) == 0, (
        "Binary file must not be re-parsed on run 2 (should be skipped by "
        "change detection, not by the parser)"
    )
    assert result2.get("skipped_unchanged", 0) >= 1, (
        "Binary file should appear in skipped_unchanged (caught by change detection)"
    )


@pytest.mark.asyncio
async def test_successfully_indexed_file_has_no_skip_reason(coordinator, tmp_path):
    """A file that is successfully indexed must not have skip_reason set."""
    pyfile = tmp_path / "code.py"
    pyfile.write_text("x = 1\n")

    await coordinator.process_directory(tmp_path, patterns=_PATTERNS)

    rows = coordinator._db.execute_query(
        "SELECT path, skip_reason FROM files WHERE path LIKE '%code.py'"
    )
    assert rows, "Indexed file should have a DB record"
    assert rows[0]["skip_reason"] is None, (
        "Successfully indexed file must not have skip_reason"
    )
