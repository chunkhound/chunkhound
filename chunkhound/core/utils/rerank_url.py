"""Reranker URL rules shared by configuration and HTTP providers."""

import re
from urllib.parse import urljoin, urlsplit

RERANK_BASE_URL_REQUIRED = (
    "rerank_model or rerank_format requires base_url or explicit rerank_url"
)
RERANK_URL_SCHEME_INVALID = (
    "rerank_url must use a complete HTTP(S) URL or a relative path"
)


def normalize_rerank_url(value: str) -> str:
    """Normalize only the scheme; preserve endpoint path, query, and fragment."""
    if value.startswith("//") or any(char.isspace() for char in value):
        raise ValueError(RERANK_URL_SCHEME_INVALID)
    match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):", value)
    if not match:
        return value
    scheme = match[1].lower()
    if scheme not in ("http", "https") or not value[len(match[0]) :].startswith("//"):
        raise ValueError(RERANK_URL_SCHEME_INVALID)
    normalized = scheme + value[len(match[1]) :]
    _validate_http_authority(normalized)
    return normalized


def _validate_http_authority(value: str) -> None:
    """Reject incomplete hosts and malformed ports before the HTTP client runs."""
    try:
        parsed = urlsplit(value)
        if not parsed.hostname:
            raise ValueError(RERANK_URL_SCHEME_INVALID)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(RERANK_URL_SCHEME_INVALID) from exc


def resolve_rerank_url(value: str, base_url: str | None, *, join: bool = False) -> str:
    """Keep each provider's existing relative-path semantics (Voyage uses urljoin)."""
    value = normalize_rerank_url(value)
    if value.startswith(("http://", "https://")):
        return value
    if not base_url:
        raise ValueError(RERANK_BASE_URL_REQUIRED)
    base = base_url.rstrip("/") + "/"
    relative = value.lstrip("/")
    return urljoin(base, relative) if join else base + relative
