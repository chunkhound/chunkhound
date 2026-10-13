"""Scoped ownership of the native database coordination handles."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Any

from chunkhound_native import DatabaseLease, InstallLease, acquire_database_lease

# Failed reader shutdown must survive dropped/cleared exception frames and even
# an abandoned caller. Explicit close retries release these owners; otherwise
# they conservatively exclude installation until process exit.
_failed_reader_owners: set[DatabaseOwnership] = set()


class DatabaseOwnership:
    """Own either native lease; borrowed scopes never close their caller's lease."""

    def __init__(self, handle: DatabaseLease | InstallLease) -> None:
        self._handle = handle
        self._failed_readers: list[Any] = []
        self._close_lock = RLock()

    def assert_covers(self, path: Path | str) -> None:
        self._handle.assert_covers(str(path))

    def retain_failed_reader(self, reader: Any) -> None:
        """Keep a reader and its lease alive independently of exception frames."""
        with self._close_lock:
            if not any(held is reader for held in self._failed_readers):
                self._failed_readers.append(reader)
            _failed_reader_owners.add(self)

    def close(self) -> None:
        """Retry failed reader shutdown before releasing either kind of lease."""
        with self._close_lock:
            failure: BaseException | None = None
            for reader in list(self._failed_readers):
                try:
                    reader.close()
                except BaseException as error:
                    if failure is None:
                        failure = error
                else:
                    self._failed_readers.remove(reader)
            if failure is not None:
                raise DatabaseReaderCloseError(self) from failure
            self._handle.close()
            _failed_reader_owners.discard(self)

    def __enter__(self) -> DatabaseOwnership:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class DatabaseReaderCloseError(RuntimeError):
    """Reader shutdown is unproven; ownership.close() retries without unlocking."""

    def __init__(self, ownership: DatabaseOwnership) -> None:
        super().__init__(
            "Cannot close database inspection reader; ownership is retained. "
            "Retry this error's ownership.close() before installation."
        )
        self.ownership = ownership


def inspection_parent_missing(
    path: Path | str, ownership: DatabaseOwnership | None = None
) -> bool:
    """Permit only a terminal absence result, never an unowned backend read."""
    if ownership is not None or str(path) == ":memory:":
        # Borrowed ownership must reach ownership_scope's identity/open check.
        return False
    try:
        # Resolve aliases before checking: missing/../existing still coordinates
        # with the existing target, including its interruption marker.
        Path(path).resolve().parent.stat()
    except FileNotFoundError:
        return True
    return False


@contextmanager
def ownership_scope(
    path: Path | str, ownership: DatabaseOwnership | None = None
) -> Iterator[DatabaseOwnership | None]:
    if ownership is not None:
        ownership.assert_covers(path)
        yield ownership
    elif str(path) == ":memory:":
        yield None
    else:
        with DatabaseOwnership(acquire_database_lease(str(path))) as acquired:
            yield acquired
