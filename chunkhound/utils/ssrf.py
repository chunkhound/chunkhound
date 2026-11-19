"""Shared SSRF URL validation for fetchurl and websearch.

This module is deliberately a leaf: it imports only the standard library,
so both ``chunkhound.utils.fetchurl`` and ``chunkhound.utils.websearch_core``
can import it without creating a cycle (fetchurl already imports
websearch_core, which is why the two validators were duplicated before).

Websearch raises ``SsrfBlockedError`` directly from this module.
Fetchurl pre-validates with it and lets it propagate unchanged; the CLI
and MCP boundaries catch it alongside fetchurl's own ``FetchUrlError``
(which is a plain ``ValueError`` covering non-SSRF failures too). The
blocking logic itself lives here, once.

Validation rule: only http(s) URLs whose hostname resolves entirely to
globally routable addresses are allowed. ``not ip.is_global`` (rather than
``ip.is_private``) is the check, because it also covers CGNAT
100.64.0.0/10, which ``is_private`` does not flag on some Python versions.
IPv4-mapped IPv6 addresses (``::ffff:127.0.0.1``) are unwrapped before the
check, so a private address cannot hide behind an IPv6 mapping on older
Pythons. IP-literal hostnames are validated directly, with no DNS lookup.

Out of scope, as before: full DNS-rebind defence. A resolver that returns
a public IP here could still return a private IP at fetch time; closing
that gap requires pinning the resolved IP through the fetch, which the
fetch stack does not currently support.
"""

from __future__ import annotations

import asyncio
import ipaddress
import queue
import socket
import threading
from urllib.parse import urlparse

__all__ = [
    "SsrfBlockedError",
    "validate_url",
    "validate_url_sync",
]

# Bound on DNS resolution. getaddrinfo has no timeout parameter, so the
# lookup runs on a daemon thread and the caller fails closed on timeout
# (the orphaned thread cannot block interpreter exit, and the URL is
# rejected rather than fetched unvalidated).
_DEFAULT_DNS_TIMEOUT = 10.0


class SsrfBlockedError(ValueError):
    """Raised when a fetch URL fails the SSRF pre-check."""


def _check_ip(ip_str: str, hostname: str) -> None:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        # Fail closed: an unparseable address must never pass a security
        # check silently (unreachable in practice — callers pass either
        # a validated IP literal or a getaddrinfo result — but the check
        # is the wrong place to be lenient).
        raise SsrfBlockedError(
            f"Unparseable address {ip_str!r} for {hostname!r}"
        ) from None
    # Unwrap IPv4-mapped IPv6 so ::ffff:127.0.0.1 is judged as 127.0.0.1.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    # Multicast is the exception to is_global: 224.0.0.0/4 reports
    # is_global True (verified on Python 3.12.3: 224.0.0.1 and
    # 233.252.175.1 both report is_global True and is_multicast True),
    # so it needs its own check.
    if not ip.is_global or ip.is_multicast:
        raise SsrfBlockedError(
            f"Blocked non-public address {ip_str!r} for {hostname!r}"
        )


def _resolve(hostname: str, dns_timeout: float | None) -> list:
    """Resolve ``hostname`` to getaddrinfo results, failing closed on timeout."""
    if dns_timeout is None:
        return socket.getaddrinfo(hostname, None)
    results: queue.Queue[tuple[str, object]] = queue.Queue()

    def _lookup() -> None:
        try:
            results.put(("ok", socket.getaddrinfo(hostname, None)))
        except BaseException as e:  # re-raised by the caller as-is
            results.put(("err", e))

    thread = threading.Thread(target=_lookup, daemon=True)
    thread.start()
    try:
        kind, value = results.get(timeout=dns_timeout)
    except queue.Empty:
        raise SsrfBlockedError(
            f"DNS resolution timed out for {hostname!r} (>{dns_timeout:g}s)"
        ) from None
    if kind == "err":
        raise value  # type: ignore[misc]
    return value  # type: ignore[return-value]


def validate_url_sync(
    url: str, *, dns_timeout: float | None = _DEFAULT_DNS_TIMEOUT
) -> None:
    """SSRF pre-check for a fetch URL. Raises ``SsrfBlockedError`` on rejection.

    Rejects non-http(s) schemes, missing hostnames, unresolvable hosts,
    DNS timeouts, and any resolved address that is not globally routable
    (loopback, private, link-local, reserved, multicast, unspecified,
    CGNAT, and IPv4-mapped private addresses — including cloud-metadata
    addresses such as 169.254.169.254).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SsrfBlockedError(f"Unsupported URL scheme: {parsed.scheme!r}")
    hostname = parsed.hostname
    if hostname is None:
        raise SsrfBlockedError(f"URL has no hostname: {url!r}")

    # IP literals need no DNS round-trip (and no timeout risk).
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        _check_ip(hostname, hostname)
        return

    try:
        addrs = _resolve(hostname, dns_timeout)
    except SsrfBlockedError:
        raise
    except socket.gaierror as e:
        raise SsrfBlockedError(f"DNS resolution failed for {hostname!r}: {e}") from e

    _check_resolved(hostname, addrs)


def _check_resolved(hostname: str, addrs: list) -> None:
    """Reject ``hostname`` unless every resolved address is globally routable."""
    for _family, _type, _proto, _canon, sockaddr in addrs:
        _check_ip(sockaddr[0], hostname)


async def validate_url(
    url: str, *, dns_timeout: float | None = _DEFAULT_DNS_TIMEOUT
) -> None:
    """Async SSRF pre-check; same rejections as ``validate_url_sync``.

    DNS resolution uses the event loop's ``getaddrinfo`` (backed by the
    loop's executor) bounded by ``asyncio.wait_for``, so a slow lookup
    fails closed without spawning a dedicated thread per call. The
    synchronous path keeps its own thread in ``_resolve`` because
    blocking ``getaddrinfo`` has no cancellable form there; that thread
    is a daemon and cannot block interpreter exit.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SsrfBlockedError(f"Unsupported URL scheme: {parsed.scheme!r}")
    hostname = parsed.hostname
    if hostname is None:
        raise SsrfBlockedError(f"URL has no hostname: {url!r}")

    # IP literals need no DNS round-trip (and no timeout risk).
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        _check_ip(hostname, hostname)
        return

    loop = asyncio.get_running_loop()
    try:
        if dns_timeout is None:
            addrs = await loop.getaddrinfo(hostname, None)
        else:
            addrs = await asyncio.wait_for(
                loop.getaddrinfo(hostname, None), timeout=dns_timeout
            )
    except asyncio.TimeoutError:
        raise SsrfBlockedError(
            f"DNS resolution timed out for {hostname!r} (>{dns_timeout:g}s)"
        ) from None
    except socket.gaierror as e:
        raise SsrfBlockedError(f"DNS resolution failed for {hostname!r}: {e}") from e

    _check_resolved(hostname, addrs)
