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

    @property
    def readers_closed(self) -> bool:
        """Whether all retained inspection readers have successfully closed."""
        with self._close_lock:
            return not self._failed_readers

    def close_readers(self) -> None:
        """Drain failed inspection readers without releasing the enclosing lease."""
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
            _failed_reader_owners.discard(self)

    def close(self) -> None:
        """Release the lease; the lifecycle owner must close its other handles first."""
        with self._close_lock:
            try:
                self.close_readers()
            except DatabaseReaderCloseError as error:
                # This lifecycle has requested full closure (e.g. an acquired
                # inspection scope unwinding), unlike a borrowed reader failure.
                error._release_ownership = True
                raise
            self._handle.close()

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
    """Retry reader shutdown safely, without closing a borrowed lifecycle's lease."""

    def __init__(self, ownership: DatabaseOwnership) -> None:
        super().__init__(
            "Cannot close database inspection reader; ownership is retained. "
            "Call this error's retry_close() to retry reader shutdown. "
            "Borrowed ownership stays held until its lifecycle owner fully closes."
        )
        self.ownership = ownership
        self._release_ownership = False

    def retry_close(self) -> None:
        """Close retained readers; finish lease release only for a closing owner.

        Borrowed provider inspection errors never request lease release: use the
        provider's public close() to finish its complete cleanup. An acquired
        inspection scope already requested full closure on exit, so its retry
        also releases the otherwise-abandoned standalone lease.
        """
        if self._release_ownership:
            self.ownership.close()
        else:
            self.ownership.close_readers()


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
