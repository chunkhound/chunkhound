"""Hub failure categories and safe diagnostic labels."""

import json
from collections.abc import Callable

from chunkhound.providers.database.ownership import (
    DatabaseOwnership,
    DatabaseReaderCloseError,
)

HUB_PULL_FAILURE_GUIDANCE = (
    "The attempted hub pull did not complete. To index locally without another "
    "hub attempt, run chunkhound pull . --forget-trust or disable hub.url, then "
    "rerun chunkhound index ."
)


class HubUnavailable(RuntimeError):  # noqa: N818 - public hub failure category
    """Hub cannot supply a usable candidate; auto indexing may continue."""


class HubDatabaseMismatch(RuntimeError):  # noqa: N818 - public hub failure category
    """Downloaded database contradicts the selected manifest; never fall back."""


class HubInstallationInterrupted(RuntimeError):  # noqa: N818 - public hub category
    """Installation may be incomplete; ordinary opens and automatic fallback stop."""


class HubLocalStateError(RuntimeError):
    """Local installation state is unsafe or ambiguous; do not fall back."""


class HubAuthorizationError(RuntimeError):
    """Hub authorization state could not be verified safely; do not fall back."""


class HubReaderCloseError(RuntimeError):
    """Fatal inspection shutdown failure retaining the original safe retry."""

    def __init__(
        self,
        reader_error: DatabaseReaderCloseError,
        *,
        cleanup: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            "Cannot close hub database inspection reader; database ownership "
            "is retained. Stop this database user and retry; local indexing "
            "was not started"
        )
        self.reader_error = reader_error
        self._cleanup = cleanup

    @property
    def ownership(self) -> DatabaseOwnership:
        return self.reader_error.ownership

    def retry_close(self) -> None:
        self.reader_error.retry_close()
        if self._cleanup is not None:
            self._cleanup()


FATAL_HUB_ERRORS: tuple[type[Exception], ...] = (
    HubAuthorizationError,
    HubDatabaseMismatch,
    HubInstallationInterrupted,
    HubReaderCloseError,
    HubLocalStateError,
)


def safe_label(value: object) -> str:
    text = json.dumps(value, ensure_ascii=True, default=str)
    return (
        "[redacted metadata]" if any(s in text for s in ("://", "?")) else text[:2048]
    )
