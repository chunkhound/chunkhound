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


@pytest.mark.asyncio
async def test_fetch_with_retry_blocks_public_to_private_redirect(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A redirect from a public address to loopback is not followed (#478).

    The initial URL passes the SSRF pre-check; only its redirect target is
    private. ``_managed_browser`` is pinned to the urllib branch so the
    test does not depend on a local Chrome. The blocked hop must also not
    be retried: an SSRF rejection is terminal.
    """
    import http.server
    from contextlib import asynccontextmanager

    from chunkhound.core.config.config import Config
    from chunkhound.utils import fetchurl
    from tests.test_websearch_ssrf import (
        _PUBLIC_TEST_HOST,
        _patch_public_test_dns,
        _start_server,
        _stop_server,
    )

    _patch_public_test_dns(monkeypatch)

    @asynccontextmanager
    async def _no_browser(warning_callback=None):
        yield None

    monkeypatch.setattr(fetchurl, "_managed_browser", _no_browser)

    secret_hits: list[str] = []
    start_hits: list[str] = []

    class _SecretHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — stdlib API
            secret_hits.append(self.path)
            body = b"<html><body>SECRET</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    secret_server, secret_thread = _start_server(_SecretHandler)
    target = f"http://127.0.0.1:{secret_server.server_address[1]}/secret"

    class _RedirectHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — stdlib API
            start_hits.append(self.path)
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    try:
        redirect_server, redirect_thread = _start_server(_RedirectHandler)
        try:
            port = redirect_server.server_address[1]
            with pytest.raises(SsrfBlockedError):
                await fetchurl._fetch_with_retry(
                    f"http://{_PUBLIC_TEST_HOST}:{port}/start",
                    Config(target_dir=tmp_path),
                )
        finally:
            _stop_server(redirect_server, redirect_thread)
    finally:
        _stop_server(secret_server, secret_thread)

    # The private hop was validated, never fetched, and nothing was retried.
    assert secret_hits == []
    assert start_hits == ["/start"]
