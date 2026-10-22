"""Reranker URL rules shared by configuration and HTTP providers."""

import re
from urllib.parse import urljoin, urlsplit

RERANK_BASE_URL_REQUIRED = (
    "rerank_model or rerank_format requires base_url or explicit rerank_url"
)
RERANK_URL_SCHEME_INVALID = (
    "rerank_url must use a complete HTTP(S) URL or a relative path"
)


def rerank_url_for_log(value: str) -> str:
    """Redact before escaping: rejected URLs cannot safely use a URL parser."""
    endpoint = re.split(r"[?#]", value, maxsplit=1)[0]
    # A delimiter can hide userinfo from endpoint. If the raw URL has @, do
    # not echo the pre-delimiter text because it can hold credentials.
    if "@" in value and "@" not in endpoint:
        return ascii("<invalid URL>")
    authority = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*:)?//([^/]*)(.*)$", endpoint, re.S)
    if authority and "@" not in authority[3]:
        scheme, host, path = authority.groups()
        endpoint = f"{scheme or ''}//{host.rsplit('@', 1)[-1]}{path}"
    elif "@" in endpoint:
        # Malformed authorities cannot distinguish credentials from a path.
        endpoint = "<invalid URL>"
    return ascii(endpoint)


def drop_invalid_rerank_url_message(provider: str, url: str, exc: ValueError) -> str:
    """One sentence shape for every layer that drops an unusable rerank_url."""
    return (
        f"Dropping invalid rerank_url={rerank_url_for_log(url)} "
        f"for {provider} provider: {exc}"
    )


def drop_relative_rerank_url_message(provider: str, url: str) -> str:
    """One sentence shape for dropping a relative rerank_url without base_url."""
    return (
        f"Dropping relative rerank_url={rerank_url_for_log(url)} "
        f"for {provider} provider: base_url is required"
    )


def normalize_rerank_url(value: str) -> str:
    """Normalize only the scheme; preserve endpoint path, query, and fragment."""
    # A literal backslash has no URL meaning and makes the effective host
    # parser-dependent (WHATWG treats it as a path separator, httpx as an
    # authority delimiter). Reject it so both layers agree on the target.
    if value.startswith("//") or "\\" in value or any(c.isspace() for c in value):
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
