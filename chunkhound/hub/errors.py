"""Hub failure categories and safe diagnostic labels."""

import json

from chunkhound.providers.database.ownership import (
    DatabaseOwnership,
    DatabaseReaderCloseError,
)


class HubUnavailable(RuntimeError):  # noqa: N818 - public hub failure category
    """Hub cannot supply a usable candidate; auto indexing may continue."""


class HubDatabaseMismatch(RuntimeError):  # noqa: N818 - public hub failure category
    """Downloaded database contradicts the selected manifest; never fall back."""


class HubInstallationInterrupted(RuntimeError):  # noqa: N818 - public hub category
    """Installation may be incomplete; ordinary opens and automatic fallback stop."""


class HubReaderCloseError(RuntimeError):
    """Fatal inspection shutdown failure retaining the original safe retry."""

    def __init__(self, reader_error: DatabaseReaderCloseError) -> None:
        super().__init__(
            "Cannot close hub database inspection reader; database ownership "
            "is retained. Stop this database user and retry; local indexing "
            "was not started"
        )
        self.reader_error = reader_error

    @property
    def ownership(self) -> DatabaseOwnership:
        return self.reader_error.ownership

    def retry_close(self) -> None:
        self.reader_error.retry_close()


def safe_label(value: object) -> str:
    text = json.dumps(value, ensure_ascii=True, default=str)
    return (
        "[redacted metadata]" if any(s in text for s in ("://", "?")) else text[:2048]
    )
