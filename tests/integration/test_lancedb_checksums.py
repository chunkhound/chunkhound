import asyncio
from pathlib import Path

import pytest


lancedb = pytest.importorskip("lancedb")


def test_lancedb_checksums_saved_on_first_index(tmp_path: Path):
    """Verify checksums are computed and saved during first indexing pass with LanceDB."""
    from chunkhound.providers.database.lancedb_provider import LanceDBProvider
    from chunkhound.services.indexing_coordinator import IndexingCoordinator
    from chunkhound.utils.hashing import compute_file_hash

    db_path = tmp_path / "ldb"
    provider = LanceDBProvider(db_path, base_directory=tmp_path)
    provider.connect()

    # Create test files
    test_files = []
    for i in range(3):
        p = tmp_path / f"test{i}.py"
        p.write_text(f"# Test file {i}\nprint('hello')\n")
        test_files.append(p)

    coord = IndexingCoordinator(database_provider=provider, base_directory=tmp_path)

    # First indexing pass - files are new
    result1 = asyncio.run(
        coord.process_directory(tmp_path, patterns=["**/*.py"], exclude_patterns=[])
    )

    assert result1["files_processed"] == 3, "Should process 3 new files"
    assert result1.get("skipped_unchanged", 0) == 0, "No files should be skipped on first pass"

    # Verify checksums were saved in database
    for p in test_files:
        rel_path = p.relative_to(tmp_path).as_posix()
        db_file = provider.get_file_by_path(rel_path, as_model=False)
        assert db_file is not None, f"File {rel_path} should exist in DB"

        # Verify checksum was saved
        db_hash = db_file.get("content_hash")
        assert db_hash is not None, f"File {rel_path} should have checksum saved"
        assert db_hash != "", f"File {rel_path} checksum should not be empty"

        # Verify checksum matches actual file content
        expected_hash = compute_file_hash(p)
        assert db_hash == expected_hash, f"Checksum for {rel_path} should match actual content"

    # Second indexing pass - files unchanged
    result2 = asyncio.run(
        coord.process_directory(tmp_path, patterns=["**/*.py"], exclude_patterns=[])
    )

    # Files should be skipped on second pass
    assert result2["files_processed"] == 0, "No files should be processed on second pass"
    assert result2.get("skipped_unchanged", 0) == 3, "All 3 files should be skipped on second pass"


def test_lancedb_checksum_skip_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify LanceDB skips files with matching checksums."""
    from chunkhound.core.models.file import File
    from chunkhound.core.types.common import Language
    from chunkhound.providers.database.lancedb_provider import LanceDBProvider
    from chunkhound.services.indexing_coordinator import IndexingCoordinator
    from chunkhound.utils.hashing import compute_file_hash

    db_path = tmp_path / "ldb"
    provider = LanceDBProvider(db_path, base_directory=tmp_path)
    provider.connect()

    # Create files and insert prior state with content hash
    files = []
    for i in range(2):
        p = tmp_path / f"f{i}.txt"
        p.write_text("hello")
        st = p.stat()
        # Compute content hash to simulate a previous index with checksum
        content_hash = compute_file_hash(p)
        f = File(
            path=(p.relative_to(tmp_path)).as_posix(),
            mtime=float(st.st_mtime),
            language=Language.TEXT,
            size_bytes=int(st.st_size),
            content_hash=content_hash,
        )
        provider.insert_file(f)
        files.append(p)

    coord = IndexingCoordinator(database_provider=provider, base_directory=tmp_path)

    # Avoid parsing: record what would be parsed
    called = []

    async def _fake(files, config_file_size_threshold_kb=20, parse_task=None, on_batch=None):
        called.append(list(files))
        return []

    monkeypatch.setattr(coord, "_process_files_in_batches", _fake)

    result = asyncio.run(
        coord.process_directory(
            tmp_path, patterns=["**/*.txt"], exclude_patterns=[], config_file_size_threshold_kb=20
        )
    )

    assert result["files_processed"] == 0
    assert result.get("skipped_unchanged", 0) == 2
    assert called and len(called[0]) == 0
