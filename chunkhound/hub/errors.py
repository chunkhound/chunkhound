"""Hub failure categories and safe diagnostic labels."""

import json


class HubUnavailable(RuntimeError):  # noqa: N818 - public hub failure category
    """Hub cannot supply a usable candidate; auto indexing may continue."""


class HubDatabaseMismatch(RuntimeError):  # noqa: N818 - public hub failure category
    """Downloaded database contradicts the selected manifest; never fall back."""


class HubInstallationInterrupted(RuntimeError):  # noqa: N818 - public hub category
    """Installation may be incomplete; ordinary opens and automatic fallback stop."""


def safe_label(value: object) -> str:
    text = json.dumps(value, ensure_ascii=True, default=str)
    return (
        "[redacted metadata]" if any(s in text for s in ("://", "?")) else text[:2048]
    )
