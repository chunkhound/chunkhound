"""Independent committed cursors; not a replacement for native/engine locks."""

import os
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb

_lock = threading.RLock()
_live: dict[tuple[str, bool], weakref.WeakSet[Any]] = {}


def _key(path: Path | str, read_only: bool) -> tuple[str, bool]:
    return os.path.normcase(str(Path(path).resolve())), read_only


def register_duckdb_connection(path: Path | str, read_only: bool, conn: Any) -> None:
    if str(path) == ":memory:":
        return
    with _lock:
        _live.setdefault(_key(path, read_only), weakref.WeakSet()).add(conn)


def close_duckdb_connection(conn: Any) -> None:
    with _lock:
        # Close before unregistering; a failed close remains represented.
        conn.close()
        for key, connections in list(_live.items()):
            connections.discard(conn)
            if not connections:
                del _live[key]


@contextmanager
def compatible_duckdb_cursor(
    path: Path | str, *, read_only: bool
) -> Iterator[Any | None]:
    cursor = None
    with _lock:
        if str(path) != ":memory:":
            for conn in list(_live.get(_key(path, read_only), ())):
                try:
                    cursor = conn.cursor()
                    break
                except duckdb.ConnectionException:
                    # Public compatibility callers may have closed their handle.
                    _live[_key(path, read_only)].discard(conn)
    try:
        yield cursor
    finally:
        if cursor is not None:
            cursor.close()
