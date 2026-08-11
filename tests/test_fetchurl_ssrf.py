"""SSRF gate for fetchurl — non-http(s) scheme, missing host, private IPs."""

from __future__ import annotations

import pytest

from chunkhound.utils.fetchurl import FetchUrlError, _validate_url_and_resolve


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
    with pytest.raises(FetchUrlError):
        await _validate_url_and_resolve(url)
