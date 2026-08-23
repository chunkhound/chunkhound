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
- Any recoverable failure (timeout, transport error, non-2xx) logs a
  WARNING and returns ``None``; the caller then aborts the pipeline for
  this invocation.
- Redirects are followed (``follow_redirects=True``) so operators can put
  the endpoint behind a CDN or path rewriter without every fetch failing
  envelope_parse_error on the 301's non-JSON body (3xx responses satisfy
  status_code < 400 and would slip past the check). The 10-second
  wall-clock budget still bounds the total including any redirect chain. Note: httpx strips the
  ``Authorization`` header on cross-origin redirects but forwards it on
  same-origin hops (same scheme + host + port, with an HTTP→HTTPS
  upgrade exception on the same host); a URL that redirects within its
  own origin will still receive the interpolated token, so the
  configured endpoint's origin is the trust boundary. Failure logs drop
  userinfo, query, and fragment from both the configured URL and any
  redirected final URL so a credential or pre-signed object (as the
  endpoint or as a CDN hop) does not copy secrets into the WARNING line.
- The raw templated string is never mutated upstream. Persistence keeps it
  verbatim so subsequent runs re-interpolate against the *current*
  environment rather than freezing a stale secret to disk.
"""

import asyncio
import os
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from chunkhound.utils.logging_guard import log_if_not_mcp

_TIMEOUT_SECONDS: float = 10.0
_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _url_for_log(raw: str) -> str:
    """Scheme + host + port + path; drop userinfo/query/fragment so secrets stay out of logs."""
    parts = urlsplit(raw)
    _, _, netloc = parts.netloc.rpartition("@")
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


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


async def fetch(url: str, auth_header: str | None) -> Any | None:
    """Fetch the remote-config envelope; return parsed JSON or ``None``.

    Returns:
        Parsed JSON payload (typically a dict) on success. ``None`` on any
        recoverable failure (timeout, transport error, non-2xx, JSON parse
        failure of the response body).
    """
    headers: dict[str, str] = {}
    if auth_header is not None:
        interpolated = _interpolate_env(auth_header)
        if interpolated is not None:
            headers["Authorization"] = interpolated
        # else: header dropped, WARNING already logged; continue without auth

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_TIMEOUT_SECONDS),
            follow_redirects=True,
        ) as client:
            response = await asyncio.wait_for(
                client.get(url, headers=headers),
                timeout=_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError:
        log_if_not_mcp(
            "WARNING",
            "Remote-config fetch timed out after {}s: {}",
            _TIMEOUT_SECONDS,
            _url_for_log(url),
        )
        return None
    except httpx.TimeoutException:
        log_if_not_mcp(
            "WARNING",
            "Remote-config fetch timeout: {}",
            _url_for_log(url),
        )
        return None
    except Exception as exc:  # httpx transport / connection / SSL / etc.
        log_if_not_mcp(
            "WARNING",
            "Remote-config fetch failed ({}): {}",
            type(exc).__name__,
            _url_for_log(url),
        )
        return None

    origin_url = _url_for_log(url)
    if response.history:
        logged_url = f"{_url_for_log(str(response.url))} (via {origin_url})"
    else:
        logged_url = origin_url

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
