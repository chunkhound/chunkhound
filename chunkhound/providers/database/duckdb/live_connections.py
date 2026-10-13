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
    # Admission spans preflight, backend open and registration. Compaction must
    # not see an empty registry while an open already pins the old database.
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
        if cursor is not None:
            # The borrowing manager/provider owns cursor cleanup, not the source
            # connection or an exception traceback. Failed exit keeps both this
            # strong reference and registry admission until a successful retry.
            retained_connections.append(cursor)
            register_duckdb_connection(path, read_only, cursor)
        try:
            yield cursor
        finally:
            if cursor is not None:
                close_duckdb_connection(cursor)
                retained_connections.remove(cursor)


@contextmanager
def exclusive_duckdb_compaction(
    path: Path | str, owned_connections: tuple[Any, ...]
) -> Iterator[None]:
    """Exclude opens until swap/recovery and reconnect finish, or refuse live users.

    This same-process admission gate complements native installation exclusion;
    it never closes or changes another provider's handles or transactions.
    """
    with _lock:
        target = _key(path, False)[0]
        for (database, _), connections in _live.items():
            if database == target and any(
                all(conn is not owned for owned in owned_connections)
                for conn in connections
            ):
                raise RuntimeError(
                    "Cannot compact database while another provider connection is live"
                )
        yield
