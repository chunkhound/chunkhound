"""SSRF gate for fetchurl — non-http(s) scheme, missing host, private IPs."""

from __future__ import annotations

import socket

import pytest

from chunkhound.utils.fetchurl import FetchUrlError, _validate_url_and_resolve


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",  # non-http(s) scheme
        "http:///no-host/path",  # missing hostname
        "http://127.0.0.1/",  # loopback
        "http://[::1]/",  # IPv6 loopback
        "http://10.0.0.1/",  # private (RFC 1918)
        "http://169.254.169.254/latest/meta-data/",  # link-local (cloud metadata)
    ],
)
@pytest.mark.asyncio
async def test_fetchurl_ssrf_rejects_unsafe_urls(url: str) -> None:
    with pytest.raises(FetchUrlError):
        await _validate_url_and_resolve(url)


@pytest.mark.asyncio
async def test_hostname_resolving_to_private_ip_is_rejected(monkeypatch) -> None:
    """Addresses returned by DNS must pass the same SSRF classification."""

    def fake_getaddrinfo(*args: object, **kwargs: object) -> list[tuple]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.5", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(FetchUrlError):
        await _validate_url_and_resolve("http://internal.example/")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",  # IPv4 loopback literal
        "http://10.0.0.1/",  # IPv4 private literal
        "http://0.0.0.0/",  # IPv4 unspecified literal
        "http://[::]/",  # IPv6 unspecified literal
        "http://[::1]/",  # IPv6 loopback literal
        "http://[::ffff:127.0.0.1]/",  # IPv4-mapped IPv6 loopback
    ],
)
@pytest.mark.asyncio
async def test_literal_ip_never_calls_resolver(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    """Literal IPs classify without DNS; resolver must stay unused."""

    def _fail(*args: object, **kwargs: object) -> list[tuple]:
        raise AssertionError("resolver must not run for literal IP")

    monkeypatch.setattr(socket, "getaddrinfo", _fail)
    with pytest.raises(FetchUrlError):
        await _validate_url_and_resolve(url)
