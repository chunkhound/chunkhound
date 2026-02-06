import asyncio
from pathlib import Path

import pytest


class _FakeDB:
    def __init__(self, records):
        self._records = records  # rel_path -> dict
        self.updated = []
        self._next_id = max([rec.get("id", 0) for rec in records.values()], default=0) + 1

    def get_file_by_path(self, path: str, as_model: bool = False):
        return self._records.get(path)

    def insert_file(self, file_model):
        """Insert new file and return file_id."""
        file_id = self._next_id
        self._next_id += 1
        rec = {
            "id": file_id,
            "path": file_model.path,
            "size": file_model.size_bytes,
            "modified_time": file_model.mtime,
            "content_hash": file_model.content_hash,
        }
        self._records[file_model.path] = rec
        return file_id

    def update_file(self, file_id: int, **kwargs):
        # Persist content_hash into our record if present
        for rec in self._records.values():
            if rec["id"] == file_id:
                if "content_hash" in kwargs:
                    rec["content_hash"] = kwargs["content_hash"]
                if "size_bytes" in kwargs:
                    rec["size"] = kwargs["size_bytes"]
                if "mtime" in kwargs:
                    rec["modified_time"] = kwargs["mtime"]
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

    def should_optimize(self, operation: str = "") -> bool:
        return False


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
