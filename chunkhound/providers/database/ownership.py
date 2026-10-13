"""Scoped ownership of the native database coordination handles."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType

from chunkhound_native import DatabaseLease, InstallLease, acquire_database_lease


class DatabaseOwnership:
    """Own either native lease; borrowed scopes never close their caller's lease."""

    def __init__(self, handle: DatabaseLease | InstallLease) -> None:
        self._handle = handle

    def assert_covers(self, path: Path | str) -> None:
        self._handle.assert_covers(str(path))

    def close(self) -> None:
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
