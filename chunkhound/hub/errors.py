"""Hub failure categories and safe diagnostic labels."""

import json
from collections.abc import Callable
from urllib.parse import urlsplit

from chunkhound.providers.database.ownership import (
    DatabaseOwnership,
    DatabaseReaderCloseError,
)

HUB_PULL_CANCELLATION_DIAGNOSTIC = (
    "The attempted hub pull was cancelled before completion."
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
        """Database ownership held by the underlying reader error."""
        return self.reader_error.ownership

    def retry_close(self) -> None:
        """Retry closing the retained readers, then run any extra cleanup."""
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
    """Render a value for logs, redacting anything that looks like a URL or query."""
    text = json.dumps(value, ensure_ascii=True, default=str)
    return (
        "[redacted metadata]" if any(s in text for s in ("://", "?")) else text[:2048]
    )


def hub_origin(manifest_url: str) -> str:
    """Return only scheme, normalized host, and a non-default port."""
    parts = urlsplit(manifest_url)
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    default_port = 80 if parts.scheme == "http" else 443
    suffix = f":{port}" if port not in (None, default_port) else ""
    return f"{parts.scheme}://{host}{suffix}"
