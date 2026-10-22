"""HTTP fetch for the remote-config pipeline.

Contract:
- 10-second wall-clock deadline enforced by ``asyncio.wait_for``. The per-phase
  ``httpx.Timeout`` on the client is a finer-grained safeguard; ``wait_for``
  is what guarantees the total budget even across many phases (connect +
  read + retry) so a slow-drip server can't stall startup indefinitely.
- ``${VAR}`` interpolation on ``auth_header`` runs against ``os.environ``
  immediately before the request. Any unset reference drops the entire
  header — sending a partially-interpolated string would leak the literal
  ``${VAR}`` placeholder to the wire and could authenticate as a different
  principal than the operator intended.
- URL scheme must be ``https`` OR the host must be loopback
  (``localhost``, ``127.0.0.0/8``, ``::1``). Loopback is a narrow escape
  hatch for local development and mock servers; loopback traffic does not
  leave the machine, so cleartext HTTP there is not a MITM vector. The
  rule never DNS-resolves hostnames — a resolvable name could point at
  loopback at check time and elsewhere at fetch time.
- Redirects are followed manually with per-hop scheme + loopback
  validation (``_MAX_REDIRECTS`` caps the chain — tighter than httpx's
  default of 20). The target of every 3xx is re-validated with the same
  rule before the next request is issued, so an HTTPS→HTTP downgrade
  never dispatches a cleartext request. ``Authorization`` is stripped
  on cross-origin redirects (differing scheme, host, or port) so a
  bearer token issued for the configured endpoint does not travel to
  a redirect target.
- Any recoverable failure (timeout, transport error, non-2xx, disallowed
  scheme, redirect loop) logs a WARNING and returns ``None``; the caller
  then aborts the pipeline for this invocation.
- Failure logs drop userinfo, query, and fragment from every URL so a
  credential or pre-signed object does not copy secrets into the WARNING
  line.
- ``httpx.AsyncClient`` is constructed with ``trust_env=False`` so
  proxy and TLS-trust env vars cannot silently reroute the credentialed
  fetch. Real threats: an attacker-injected ``HTTPS_PROXY`` redirecting
  the request to an endpoint they control, and a MITM proxy holding a
  system-trusted CA terminating the tunnel and reading the
  ``Authorization`` header. Side effect: ``SSL_CERT_FILE`` /
  ``SSL_CERT_DIR`` env-based CA bundles are also ignored — operators
  needing a private CA must extend the system trust store.
- The raw templated string is never mutated upstream. Persistence keeps it
  verbatim so subsequent runs re-interpolate against the *current*
  environment rather than freezing a stale secret to disk.
"""

import asyncio
import ipaddress
import os
import re
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from chunkhound.utils.logging_guard import log_if_not_mcp

_TIMEOUT_SECONDS: float = 10.0
_MAX_REDIRECTS: int = 5
_REDIRECT_STATUSES: frozenset[int] = frozenset({301, 302, 303, 307, 308})
_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


class _FetchAbortedError(Exception):
    """Internal: fetch refused (disallowed scheme / redirect loop).

    WARNING already logged at the point of raise.
    """


def _url_for_log(raw: str) -> str:
    """Scheme + host + port + path only.

    Drops userinfo, query, and fragment so credentials and pre-signed
    tokens don't copy into WARNING lines.
    """
    parts = urlsplit(raw)
    _, _, netloc = parts.netloc.rpartition("@")
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _is_loopback_host(hostname: str | None) -> bool:
    """True iff ``hostname`` is a loopback IP literal or the string ``localhost``.

    Never DNS-resolves: a resolvable name could answer loopback at check
    time and elsewhere at fetch time.
    """
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname.strip("[]")).is_loopback
    except ValueError:
        return False


def url_scheme_ok(url: str) -> bool:
    """True iff URL is ``https`` or points at a loopback host over ``http``."""
    parts = urlsplit(url)
    if parts.scheme == "https":
        return True
    return parts.scheme == "http" and _is_loopback_host(parts.hostname)


_DEFAULT_PORTS: dict[str, int] = {"http": 80, "https": 443}


def _origin(url: str) -> tuple[str, str | None, int | None]:
    """(scheme, host, port) — the tuple used to decide same-origin for auth carry.

    Missing ports are normalized to the scheme's default so
    ``https://h/`` and ``https://h:443/`` compare equal — a CDN that
    emits an explicit ``:443`` in ``Location`` must not look cross-origin.
    """
    parts = urlsplit(url)
    port = parts.port if parts.port is not None else _DEFAULT_PORTS.get(parts.scheme)
    return (parts.scheme, parts.hostname, port)


def _interpolate_env(template: str) -> str | None:
    """Substitute ``${VAR}`` refs in ``template`` from ``os.environ``.

    Returns the interpolated string, or ``None`` if any single referenced
    variable is unset or empty — signal to drop the header entirely.
    """
    missing: list[str] = []

    def sub(match: re.Match[str]) -> str:
        var = match.group(1)
        value = os.environ.get(var)
        if not value:
            missing.append(var)
            return ""
        return value

    result = _VAR_PATTERN.sub(sub, template)
    if missing:
        log_if_not_mcp(
            "WARNING",
            "Remote-config auth header dropped — unset or empty env var(s): {}",
            ", ".join(missing),
        )
        return None
    return result


async def _fetch_with_redirects(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
) -> tuple[httpx.Response, str]:
    """Drive a manual redirect loop with per-hop scheme validation.

    Returns the terminal (non-redirect) response plus the URL that
    produced it so the caller can log the final hop. Raises
    ``_FetchAbortedError`` on disallowed scheme or exhausted hop budget
    (WARNING already emitted).
    """
    origin_url = _url_for_log(url)
    current_url = url
    request_headers = dict(headers)
    for _ in range(_MAX_REDIRECTS + 1):
        response = await client.get(current_url, headers=request_headers)
        if response.status_code not in _REDIRECT_STATUSES:
            return response, current_url
        location = response.headers.get("Location")
        if not location:
            return response, current_url
        next_url = urljoin(current_url, location)
        if not url_scheme_ok(next_url):
            log_if_not_mcp(
                "WARNING",
                "Remote-config redirect refused: URL scheme must be https "
                "(or http to a loopback host): {} (via {})",
                _url_for_log(next_url),
                origin_url,
            )
            raise _FetchAbortedError
        if _origin(next_url) != _origin(current_url):
            request_headers.pop("Authorization", None)
        current_url = next_url
    log_if_not_mcp(
        "WARNING",
        "Remote-config fetch aborted after {} redirects: {} (via {})",
        _MAX_REDIRECTS,
        _url_for_log(current_url),
        origin_url,
    )
    raise _FetchAbortedError


async def fetch(url: str, auth_header: str | None) -> Any | None:
    """Fetch the remote-config envelope; return parsed JSON or ``None``.

    Returns:
        Parsed JSON payload (typically a dict) on success. ``None`` on any
        recoverable failure (timeout, transport error, non-2xx, JSON parse
        failure of the response body, disallowed scheme, redirect loop).
    """
    origin_url = _url_for_log(url)

    if not url_scheme_ok(url):
        log_if_not_mcp(
            "WARNING",
            "Remote-config fetch refused: URL scheme must be https "
            "(or http to a loopback host): {}",
            origin_url,
        )
        return None

    headers: dict[str, str] = {}
    if auth_header is not None:
        interpolated = _interpolate_env(auth_header)
        if interpolated is not None:
            headers["Authorization"] = interpolated
        # else: header dropped, WARNING already logged; continue without auth

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_TIMEOUT_SECONDS),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response, final_url = await asyncio.wait_for(
                _fetch_with_redirects(client, url, headers),
                timeout=_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError:
        log_if_not_mcp(
            "WARNING",
            "Remote-config fetch timed out after {}s: {}",
            _TIMEOUT_SECONDS,
            origin_url,
        )
        return None
    except httpx.TimeoutException:
        log_if_not_mcp(
            "WARNING",
            "Remote-config fetch timeout: {}",
            origin_url,
        )
        return None
    except _FetchAbortedError:
        return None
    except Exception as exc:  # httpx transport / connection / SSL / etc.
        log_if_not_mcp(
            "WARNING",
            "Remote-config fetch failed ({}): {}",
            type(exc).__name__,
            origin_url,
        )
        return None

    final_url_for_log = _url_for_log(final_url)
    logged_url = (
        f"{final_url_for_log} (via {origin_url})"
        if final_url_for_log != origin_url
        else origin_url
    )

    if response.status_code >= 400:
        log_if_not_mcp(
            "WARNING",
            "Remote-config fetch returned HTTP {}: {}",
            response.status_code,
            logged_url,
        )
        return None

    try:
        return response.json()
    except Exception as exc:
        log_if_not_mcp(
            "WARNING",
            "Remote-config envelope_parse_error: {} — {}",
            logged_url,
            exc,
        )
        return None
