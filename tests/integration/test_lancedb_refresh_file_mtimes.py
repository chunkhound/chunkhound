"""LanceDB: storing the mtime of a hash-confirmed file changes that mtime only.

Another process can index the same LanceDB database between the moment a
file's hash is confirmed and the moment its mtime is stored; nothing locks
the files table across connections. The refresh must then leave that
writer's row as it is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chunkhound.core.models import File
from chunkhound.core.types.common import Language

lancedb = pytest.importorskip("lancedb")

# What a second indexing process writes just before the refresh writes:
# new content for a.py, and the same content for c.py recorded as skipped.
_CONCURRENT_WRITES = {
    "a.py": {
        "content_hash": "hash-new",
        "size": 222,
        "modified_time": 2000.0,
        "indexed_time": 2000.0,
    },
    "c.py": {
        "skip_reason": "too_large",
        "modified_time": 2100.0,
        "indexed_time": 2100.0,
    },
}


def _file(path: str, content_hash: str, size: int, mtime: float) -> File:
    return File(
        path=path,
        mtime=mtime,
        language=Language.PYTHON,
        size_bytes=size,
        content_hash=content_hash,
    )


def _rows(db_path: Path) -> dict[str, dict[str, Any]]:
    """The files table as a fresh connection sees it."""
    table = lancedb.connect(str(db_path)).open_table("files")
    return {row["path"]: row for row in table.search().limit(100).to_list()}


class _WriterBeforeFirstWrite:
    """The provider's files table, with another connection writing first.

    Just before the provider's first write to the table, a second connection
    applies _CONCURRENT_WRITES, as a concurrent indexing process would.
    """

    def __init__(self, table: Any, db_path: Path) -> None:
        self._table = table
        self._db_path = db_path
        self.wrote = False

    def _concurrent_write(self) -> None:
        if self.wrote:
            return
        self.wrote = True
        other = lancedb.connect(str(self._db_path)).open_table("files")
        for path, values in _CONCURRENT_WRITES.items():
            other.update(where=f"path = '{path}'", values=values)

    def __getattr__(self, name: str) -> Any:
        if name in ("update", "merge_insert", "add", "delete"):
            self._concurrent_write()
        return getattr(self._table, name)

    def __bool__(self) -> bool:
        return True


def test_refresh_never_rolls_back_a_concurrent_writer(lancedb_provider, tmp_path):
    from chunkhound.core.config.database_config import DatabaseConfig

    db_path = DatabaseConfig(path=tmp_path, provider="lancedb").get_db_path()
    lancedb_provider.insert_file(_file("a.py", "hash-a", 111, 1000.0))
    lancedb_provider.insert_file(_file("b.py", "hash-b", 333, 1000.0))
    lancedb_provider.insert_file(_file("c.py", "hash-c", 444, 1000.0))
    before = _rows(db_path)

    proxy = _WriterBeforeFirstWrite(lancedb_provider._files_table, db_path)
    lancedb_provider._files_table = proxy
    try:
        refreshed = lancedb_provider.refresh_file_mtimes(
            [
                ("a.py", 1500.0, "hash-a"),
                ("b.py", 1600.0, "hash-b"),
                ("c.py", 1700.0, "hash-c"),
            ]
        )
    finally:
        lancedb_provider._files_table = proxy._table
    after = _rows(db_path)

    assert proxy.wrote
    # The rows the other process rewrote keep what it wrote, mtime included.
    for path, values in _CONCURRENT_WRITES.items():
        assert after[path] == {**before[path], **values}, path
    # b.py is still the row that was confirmed: only its mtime changes.
    assert after["b.py"] == {**before["b.py"], "modified_time": 1600.0}
    assert refreshed == 1
