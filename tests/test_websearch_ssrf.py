"""SSRF gate for websearch — direct private URLs, unsafe schemes, and
public-to-private redirects must not be fetched (issue #463).

The websearch pipeline (``fetch_and_save`` → ``_fetch_one``) is the
boundary: result URLs come from a public search engine, so they are
attacker-influenced. Low-level helpers keep ``validate_ssrf=False`` by
default so existing callers (fetchurl, which validates separately, and
tests using local servers) are unaffected.
"""

from __future__ import annotations

import asyncio
import http.server
import threading
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.utils.websearch_core import (
    UnsafeFetchURLError,
    _CaptureRedirect,
    _fetch_one,
    _fetch_url,
    _RedirectTargetError,
    _validate_url_sync,
    fetch_url_to_content,
)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",  # non-http(s) scheme
        "ftp://example.com/file",  # non-http(s) scheme
        "http:///no-host/path",  # missing hostname
        "http://127.0.0.1/",  # loopback
        "http://[::1]/",  # IPv6 loopback
        "http://10.0.0.1/",  # private (RFC 1918)
        "http://192.168.1.1/",  # private (RFC 1918)
        "http://172.16.0.1/",  # private (RFC 1918)
        "http://169.254.169.254/latest/meta-data/",  # link-local (cloud metadata)
        "http://0.0.0.0/",  # unspecified
        "http://224.0.0.1/",  # multicast
    ],
)
def test_validate_url_sync_rejects_unsafe_urls(url: str) -> None:
    with pytest.raises(UnsafeFetchURLError):
        _validate_url_sync(url)


def test_validate_url_sync_accepts_public_ip_literal() -> None:
    # IP literals resolve without DNS, so this needs no network access.
    _validate_url_sync("http://93.184.216.34/")


def test_fetch_url_blocks_private_url_before_connecting(tmp_path: Path) -> None:
    """A loopback server serving 'SECRET' must never be contacted."""
    received: list[str] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — stdlib API
            received.append(self.path)
            body = b"SECRET"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with pytest.raises(UnsafeFetchURLError):
            _fetch_url(f"http://127.0.0.1:{port}/secret", validate_ssrf=True)
        assert received == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


class _FakeHeaders:
    def __init__(self, content_type: str) -> None:
        self._content_type = content_type

    def get(self, name: str, default: str | None = None) -> str | None:
        if name.lower() == "content-type":
            return self._content_type
        return default

    def get_content_charset(self) -> str:
        return "utf-8"


class _FakeResponse:
    def __init__(self, body: bytes, content_type: str = "text/html") -> None:
        self._body = body
        self.headers = _FakeHeaders(content_type)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


class _FakeOpener:
    """Simulates urllib's opener: first open() redirects, second returns."""

    def __init__(self, redirect_target: str) -> None:
        self.redirect_target = redirect_target
        self.calls: list[str] = []

    def open(self, req: urllib.request.Request, timeout: int = 30) -> _FakeResponse:
        self.calls.append(req.full_url)
        if len(self.calls) == 1:
            raise _RedirectTargetError(self.redirect_target)
        return _FakeResponse(b"<html><body>public</body></html>")


def test_fetch_url_blocks_public_to_private_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = _FakeOpener("http://169.254.169.254/latest/meta-data/")
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: opener)
    with pytest.raises(UnsafeFetchURLError):
        _fetch_url("http://93.184.216.34/start", validate_ssrf=True)
    # The private hop was validated, never fetched.
    assert opener.calls == ["http://93.184.216.34/start"]


def test_fetch_url_follows_public_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    opener = _FakeOpener("http://93.184.216.34/landing")
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: opener)
    ct, body, _charset = _fetch_url("http://93.184.216.34/start", validate_ssrf=True)
    assert ct == "text/html"
    assert body == b"<html><body>public</body></html>"
    assert opener.calls == [
        "http://93.184.216.34/start",
        "http://93.184.216.34/landing",
    ]


def test_capture_redirect_handler_reports_destination() -> None:
    handler = _CaptureRedirect()
    req = urllib.request.Request("http://93.184.216.34/start")
    with pytest.raises(_RedirectTargetError) as excinfo:
        handler.redirect_request(
            req, BytesIO(), 302, "Found", SimpleNamespace(), "http://10.0.0.1/x"
        )
    assert excinfo.value.url == "http://10.0.0.1/x"


@pytest.mark.asyncio
async def test_fetch_one_treats_blocked_url_as_per_page_failure(
    tmp_path: Path,
) -> None:
    """A blocked URL warns and saves nothing; other results continue."""
    warnings: list[str] = []
    mapping: dict[str, str] = {}
    await _fetch_one(
        "http://127.0.0.1:9/secret",
        tmp_path,
        None,
        None,
        warnings.append,
        asyncio.Semaphore(1),
        mapping,
    )
    assert len(warnings) == 1
    assert "UnsafeFetchURLError" in warnings[0]
    assert list(tmp_path.iterdir()) == []
    assert mapping == {}


@pytest.mark.asyncio
async def test_fetch_url_to_content_without_flag_keeps_old_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default (validate_ssrf=False) still delegates to _fetch_url as before."""
    monkeypatch.setattr(
        "chunkhound.utils.websearch_core._fetch_url",
        lambda url, *a, **k: ("text/html", b"hi", "utf-8"),
    )
    kind, content, _meta = await fetch_url_to_content(
        "http://127.0.0.1:9/x", browser=None
    )
    assert kind == ".md"
    assert content == "hi"
