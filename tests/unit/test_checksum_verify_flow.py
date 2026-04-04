import asyncio
from pathlib import Path

import pytest

from tests.unit.helpers import _Cfg, _FakeDB


def test_checksum_verify_populate_and_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test new checksum behavior: skip on mtime+size match, verify on mtime/size change."""
    from chunkhound.core.types.common import Language
    from chunkhound.services.indexing_coordinator import IndexingCoordinator
    from chunkhound.services.batch_processor import ParsedFileResult

    # Create a file (not in DB yet)
    p = tmp_path / "a.txt"
    p.write_text("hello world")
    st = p.stat()
    rel = p.relative_to(tmp_path).as_posix()

    # Start with empty DB
    db = _FakeDB({})
    coord = IndexingCoordinator(database_provider=db, base_directory=tmp_path, config=_Cfg())

    async def _fake_parse(files, config_file_size_threshold_kb=20, parse_task=None, on_batch=None):
        # Simulate one ParsedFileResult success for each file
        results = []
        for item in files:
            # Handle both Path and (Path, hash) tuple formats
            if isinstance(item, tuple):
                f, precomputed_hash = item
            else:
                f = item
                precomputed_hash = None
            st = f.stat()
            results.append(
                ParsedFileResult(
                    file_path=f,
                    chunks=[],
                    language=Language.TEXT,
                    file_size=st.st_size,
                    file_mtime=st.st_mtime,
                    content_hash=precomputed_hash,
                    status="success",
                )
            )
        if on_batch:
            await on_batch(results)
        return results

    monkeypatch.setattr(coord, "_process_files_in_batches", _fake_parse)

    # First run: file not in DB -> process and populate hash
    res1 = asyncio.run(
        coord.process_directory(tmp_path, patterns=["**/*.txt"], exclude_patterns=[])
    )
    assert res1["files_processed"] == 1

    # Verify file was inserted with hash
    assert rel in db._records
    assert db._records[rel]["content_hash"] is not None

    # Second run: mtime+size match -> skip immediately (fast path)
    res2 = asyncio.run(
        coord.process_directory(tmp_path, patterns=["**/*.txt"], exclude_patterns=[])
    )
    assert res2.get("skipped_unchanged", 0) == 1

    # Third run: change size (different content)
    p.write_text("hello world!")  # Different size
    st2 = p.stat()

    res3 = asyncio.run(
        coord.process_directory(tmp_path, patterns=["**/*.txt"], exclude_patterns=[])
    )
    # Size changed -> check hash -> hash differs -> process
    assert res3["files_processed"] == 1
