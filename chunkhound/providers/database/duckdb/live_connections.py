"""Independent committed cursors; not a replacement for native/engine locks."""

import os
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb


class _Admission:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.connections: dict[bool, weakref.WeakSet[Any]] = {
            False: weakref.WeakSet(),
            True: weakref.WeakSet(),
        }


_lock = threading.RLock()
_admissions: weakref.WeakValueDictionary[str, _Admission] = (
    weakref.WeakValueDictionary()
)
_connection_admissions: weakref.WeakKeyDictionary[Any, _Admission] = (
    weakref.WeakKeyDictionary()
)


def _admission(path: Path | str) -> _Admission:
    # Resolve before taking the bookkeeping mutex: filesystem I/O must not
    # stop lifecycle progress on unrelated databases.
    target = os.path.normcase(str(Path(path).resolve()))
    with _lock:
        entry = _admissions.get(target)
        if entry is None:
            entry = _Admission()
            _admissions[target] = entry
        # The return value strongly retains the entry BEFORE the caller waits.
        return entry


def register_duckdb_connection(path: Path | str, read_only: bool, conn: Any) -> None:
    if str(path) == ":memory:":
        return
    entry = _admission(path)
    with entry.lock:
        with _lock:
            entry.connections[read_only].add(conn)
            _connection_admissions[conn] = entry


def close_duckdb_connection(conn: Any) -> None:
    with _lock:
        entry = _connection_admissions.get(conn)
    if entry is None:
        conn.close()
        return
    with entry.lock:
        # Backend I/O is outside _lock; failed close retains both registries.
        conn.close()
        with _lock:
            for connections in entry.connections.values():
                connections.discard(conn)
            _connection_admissions.pop(conn, None)


def close_retained_duckdb_connections(connections: list[Any]) -> None:
    """Attempt every retained handle; failed closes stay owned for public retry."""
    failure: BaseException | None = None
    for conn in list(connections):
        try:
            close_duckdb_connection(conn)
        except BaseException as error:
            if failure is None:
                failure = error
        else:
            connections.remove(conn)
    if failure is not None:
        raise failure


@contextmanager
def compatible_duckdb_cursor(
    path: Path | str, *, read_only: bool, retained_connections: list[Any]
) -> Iterator[Any | None]:
    if str(path) == ":memory:":
        yield None
        return
    entry = _admission(path)
    cursor = None
    with entry.lock:
        with _lock:
            candidates = list(entry.connections[read_only])
        for conn in candidates:
            try:
                cursor = conn.cursor()
                break
            except duckdb.ConnectionException:
                # Public compatibility callers may close a handle directly.
                with _lock:
                    entry.connections[read_only].discard(conn)
                    _connection_admissions.pop(conn, None)
        if cursor is not None:
            retained_connections.append(cursor)
            register_duckdb_connection(path, read_only, cursor)
        try:
            # Admission spans preflight, backend open, registration and close.
            yield cursor
        finally:
            if cursor is not None:
                close_duckdb_connection(cursor)
                retained_connections.remove(cursor)


@contextmanager
def exclusive_duckdb_compaction(
    path: Path | str, owned_connections: tuple[Any, ...]
) -> Iterator[None]:
    """Exclude same-path opens through swap and reconnect, or refuse live users."""
    if str(path) == ":memory:":
        yield
        return
    entry = _admission(path)
    with entry.lock:
        with _lock:
            connections = [
                conn
                for group in entry.connections.values()
                for conn in group
            ]
        if any(
            all(conn is not owned for owned in owned_connections)
            for conn in connections
        ):
            raise RuntimeError(
                "Cannot compact database while another provider connection is live"
            )
        yield
