import asyncio
from pathlib import Path

import pytest


class _FakeDB:
    def __init__(self, records):
        self._records = records  # rel_path -> dict
        self.updated = []

    def get_file_by_path(self, path: str, as_model: bool = False):
        return self._records.get(path)

    def update_file(self, file_id: int, **kwargs):
        # Persist content_hash into our record if present
        for rec in self._records.values():
            if rec["id"] == file_id:
                if "content_hash" in kwargs:
                    rec["content_hash"] = kwargs["content_hash"]
                self.updated.append((file_id, kwargs))
                return

    # Methods used by store path
    def begin_transaction(self):
        return None

    def commit_transaction(self):
        return None

    def rollback_transaction(self):
        return None

    def get_chunks_by_file_id(self, file_id: int, as_model: bool = True):
        return []

    def insert_chunks_batch(self, chunks):
        return []


class _Cfg:
    class _Indexing:
        cleanup = False
        force_reindex = False
        per_file_timeout_seconds = 0.0
        min_dirs_for_parallel = 4
        max_discovery_workers = 4
        parallel_discovery = False

    indexing = _Indexing()


def test_checksum_verify_populate_and_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from chunkhound.core.types.common import Language
    from chunkhound.services.indexing_coordinator import IndexingCoordinator
    from chunkhound.services.batch_processor import ParsedFileResult

    # Create a file and initial DB record without content_hash
    p = tmp_path / "a.txt"
    p.write_text("hello world")
    st = p.stat()
    rel = p.relative_to(tmp_path).as_posix()
    rec = {
        "id": 1,
        "path": rel,
        "size": int(st.st_size),
        "modified_time": float(st.st_mtime),
        "content_hash": None,  # provider supports content_hash, but not populated yet
    }
    db = _FakeDB({rel: rec})
    coord = IndexingCoordinator(database_provider=db, base_directory=tmp_path, config=_Cfg())

    # First run: no hash -> should process once and populate hash via update_file
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
    res1 = asyncio.run(
        coord.process_directory(tmp_path, patterns=["**/*.txt"], exclude_patterns=[])
    )
    # Should have processed 1 file
    assert res1["files_processed"] == 1
    assert any("content_hash" in kwargs for _, kwargs in db.updated)

    # Second run: now hash exists and content unchanged -> skip unchanged
    res2 = asyncio.run(
        coord.process_directory(tmp_path, patterns=["**/*.txt"], exclude_patterns=[])
    )
    assert res2.get("skipped_unchanged", 0) == 1

    # Third run: change content without touching mtime much (best effort)
    p.write_text("hello world!")
    st2 = p.stat()
    # keep size near; checksum will differ
    res3 = asyncio.run(
        coord.process_directory(tmp_path, patterns=["**/*.txt"], exclude_patterns=[])
    )
    # Should process again due to checksum difference
    assert res3["files_processed"] == 1
