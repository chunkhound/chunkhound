"""SSRF gate for fetchurl — non-http(s) scheme, missing host, private IPs.

These tests exercise the shared validator directly (raising
``SsrfBlockedError``); fetchurl's boundary lets that error propagate
unchanged, and the CLI/MCP boundaries catch it alongside
``FetchUrlError`` (covered by the ``_fetch_with_retry`` test below).
"""

from __future__ import annotations

import pytest

from chunkhound.utils.ssrf import SsrfBlockedError, validate_url


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",                          # non-http(s) scheme
        "http:///no-host/path",                        # missing hostname
        "http://127.0.0.1/",                           # loopback
        "http://[::1]/",                               # IPv6 loopback
        "http://10.0.0.1/",                            # private (RFC 1918)
        "http://169.254.169.254/latest/meta-data/",    # link-local (cloud metadata)
    ],
)
@pytest.mark.asyncio
async def test_fetchurl_ssrf_rejects_unsafe_urls(url: str) -> None:
    with pytest.raises(SsrfBlockedError):
        await validate_url(url)


@pytest.mark.asyncio
async def test_fetch_with_retry_propagates_ssrf_blocked_error(tmp_path) -> None:
    """fetchurl's boundary lets SsrfBlockedError propagate unchanged.

    ``FetchUrlError`` is a plain ``ValueError`` for fetchurl-local
    failures only; SSRF rejections keep the shared type and are caught
    alongside ``FetchUrlError`` at the CLI/MCP boundaries.
    """
    from chunkhound.core.config.config import Config
    from chunkhound.utils.fetchurl import _fetch_with_retry
    from chunkhound.utils.ssrf import SsrfBlockedError

    with pytest.raises(SsrfBlockedError):
        await _fetch_with_retry(
            "http://127.0.0.1:9/secret", Config(target_dir=tmp_path)
        )
