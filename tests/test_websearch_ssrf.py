"""SSRF gate for websearch — direct private URLs, unsafe schemes, and
public-to-private redirects must not be fetched (issue #463).

The public pipeline (``fetch_url_to_content`` → ``_fetch_one``) is the
boundary: result URLs come from a public search engine, so they are
attacker-influenced. ``validate_ssrf`` defaults to True on
``_fetch_url``, ``_fetch_page`` and ``fetch_url_to_content`` — secure
by default. Callers that must opt out (fetchurl, which pre-validates,
and tests using local servers) do so explicitly with
``validate_ssrf=False``.

Redirect tests use a real local HTTP server plus a patched ``getaddrinfo``
at the validation boundary only: the hostname ``public.test`` resolves to
a public IP for validation (port ``None`` in the getaddrinfo call) and to
127.0.0.1 for the actual connection (port set). That is what makes a
"public-looking address that redirects to loopback" testable without
touching the internet.
"""

from __future__ import annotations

import asyncio
import http.server
import socket
import threading
from types import SimpleNamespace
from typing import Any

import pytest

try:
    import zendriver  # noqa: F401 — needed by _fetch_page's CDP import
except ImportError:  # pragma: no cover - CI installs zendriver
    _HAS_ZENDRIVER = False
else:
    _HAS_ZENDRIVER = True

from chunkhound.utils.ssrf import SsrfBlockedError, validate_url_sync
from chunkhound.utils.websearch_core import (
    _fetch_one,
    _fetch_page,
    _fetch_url,
    fetch_url_to_content,
)

_PUBLIC_TEST_HOST = "public.test"
_PUBLIC_TEST_IP = "93.184.216.34"


def _start_server(handler_cls: type[http.server.BaseHTTPRequestHandler]):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_server(server, thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=1)


def _patch_public_test_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """The single DNS fixture for this file: route ``public.test`` to
    loopback for connections, public for checks. All getaddrinfo
    patching in this file goes through this helper (at the DNS
    boundary only), so the fake is defined once and documented here.

    Validation calls ``getaddrinfo(hostname, None)``; urllib connections
    call it with the port set. Distinguishing on that argument keeps the
    fake at the validation boundary: real DNS is used for everything else.

    Also adds ``public.test`` to no_proxy: urllib's ProxyHandler honours
    the environment, and in proxied environments the test
    fetch would otherwise be sent to the egress proxy instead of the
    local server.
    """
    import os

    for var in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(var, "")
        hosts = [h for h in current.split(",") if h] if current else []
        if _PUBLIC_TEST_HOST not in hosts:
            hosts.append(_PUBLIC_TEST_HOST)
        monkeypatch.setenv(var, ",".join(hosts))
    orig_getaddrinfo = socket.getaddrinfo

    def _fake_getaddrinfo(host, port, *args, **kwargs):
        if host == _PUBLIC_TEST_HOST:
            ip = _PUBLIC_TEST_IP if port is None else "127.0.0.1"
            sockaddr = (ip, port or 0)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", sockaddr)]
        return orig_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)


def _redirect_handler(target: str):
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — stdlib API
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    return _Handler


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
        "http://100.64.0.1/",  # CGNAT (not globally routable)
        "http://[::ffff:127.0.0.1]/",  # IPv4-mapped loopback
        "http://169.254.169.254/latest/meta-data/",  # link-local (cloud metadata)
        "http://0.0.0.0/",  # unspecified
        "http://224.0.0.1/",  # multicast
    ],
)
async def test_websearch_never_fetches_private_urls(url: str) -> None:
    """The public pipeline rejects private URLs before any connection."""
    with pytest.raises(SsrfBlockedError):
        await fetch_url_to_content(url, browser=None, validate_ssrf=True)


def test_ssrf_validator_accepts_public_ip_literal() -> None:
    # Shared-module public API (not a websearch private helper):
    # IP literals resolve without DNS, so this needs no network access.
    validate_url_sync("http://93.184.216.34/")


def test_ssrf_validator_fails_closed_on_dns_timeout(monkeypatch) -> None:
    """A DNS lookup that never returns rejects the URL, never fetches it."""
    import time

    def _slow_getaddrinfo(host, port, *args, **kwargs):
        time.sleep(5)
        sockaddr = ("93.184.216.34", port or 0)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", sockaddr)]

    monkeypatch.setattr(socket, "getaddrinfo", _slow_getaddrinfo)
    with pytest.raises(SsrfBlockedError, match="timed out"):
        validate_url_sync("http://example.test/", dns_timeout=0.05)


@pytest.mark.asyncio
async def test_ssrf_validator_async_fails_closed_on_dns_timeout(monkeypatch) -> None:
    """The async validator also fails closed on a slow DNS lookup."""
    import time

    from chunkhound.utils.ssrf import validate_url

    def _slow_getaddrinfo(host, port, *args, **kwargs):
        time.sleep(0.5)
        sockaddr = ("93.184.216.34", port or 0)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", sockaddr)]

    monkeypatch.setattr(socket, "getaddrinfo", _slow_getaddrinfo)
    with pytest.raises(SsrfBlockedError, match="timed out"):
        await validate_url("http://example.test/", dns_timeout=0.05)


def test_fetch_url_blocks_private_url_before_connecting() -> None:
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

    server, thread = _start_server(_Handler)
    try:
        port = server.server_address[1]
        with pytest.raises(SsrfBlockedError):
            _fetch_url(f"http://127.0.0.1:{port}/secret", validate_ssrf=True)
        assert received == []
    finally:
        _stop_server(server, thread)


def test_fetch_url_blocks_public_to_private_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real 302 from a public-looking address to loopback is not followed."""
    _patch_public_test_dns(monkeypatch)
    received: list[str] = []

    class _SecretHandler(http.server.BaseHTTPRequestHandler):
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

    secret_server, secret_thread = _start_server(_SecretHandler)
    try:
        secret_port = secret_server.server_address[1]
        target = f"http://127.0.0.1:{secret_port}/secret"
        redirect_server, redirect_thread = _start_server(_redirect_handler(target))
        try:
            port = redirect_server.server_address[1]
            with pytest.raises(SsrfBlockedError):
                _fetch_url(
                    f"http://{_PUBLIC_TEST_HOST}:{port}/start",
                    validate_ssrf=True,
                )
        finally:
            _stop_server(redirect_server, redirect_thread)
    finally:
        _stop_server(secret_server, secret_thread)
    # The private hop was validated, never fetched.
    assert received == []


def test_fetch_url_follows_public_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real 302 between public-looking addresses is followed normally."""
    _patch_public_test_dns(monkeypatch)

    class _LandingHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — stdlib API
            body = b"<html><body>public</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    landing_server, landing_thread = _start_server(_LandingHandler)
    try:
        landing_port = landing_server.server_address[1]
        target = f"http://{_PUBLIC_TEST_HOST}:{landing_port}/landing"
        redirect_server, redirect_thread = _start_server(_redirect_handler(target))
        try:
            port = redirect_server.server_address[1]
            ct, body, _charset = _fetch_url(
                f"http://{_PUBLIC_TEST_HOST}:{port}/start",
                validate_ssrf=True,
            )
            assert ct == "text/html"
            assert body == b"<html><body>public</body></html>"
        finally:
            _stop_server(redirect_server, redirect_thread)
    finally:
        _stop_server(landing_server, landing_thread)


def test_fetch_url_rejects_redirect_loop_after_ten_hops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An endlessly redirecting URL stops after the hop limit, not forever."""
    _patch_public_test_dns(monkeypatch)
    requests: list[str] = []

    class _LoopHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — stdlib API
            requests.append(self.path)
            self.send_response(302)
            self.send_header("Location", self.path)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server, thread = _start_server(_LoopHandler)
    try:
        port = server.server_address[1]
        with pytest.raises(ValueError, match="Too many redirects"):
            _fetch_url(
                f"http://{_PUBLIC_TEST_HOST}:{port}/loop",
                validate_ssrf=True,
            )
    finally:
        _stop_server(server, thread)
    # Initial fetch + 10 redirect hops, then the limit raises.
    assert len(requests) == 11


def _cookie_recording_handler(received: dict[str, list[str | None]], target: str):
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — stdlib API
            received.setdefault(self.path, []).append(self.headers.get("Cookie"))
            if self.path == "/start":
                self.send_response(302)
                self.send_header("Location", target)
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                body = b"ok"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    return _Handler


def test_fetch_url_drops_cookie_on_cross_origin_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Cookie header must not leak to a different origin via redirect."""
    _patch_public_test_dns(monkeypatch)
    received: dict[str, list[str | None]] = {}

    landing_server, landing_thread = _start_server(
        _cookie_recording_handler(received, "")
    )
    try:
        landing_port = landing_server.server_address[1]
        # The landing server only serves /landing; the redirect server
        # serves /start and points at the landing origin (different port).
        target = f"http://{_PUBLIC_TEST_HOST}:{landing_port}/landing"
        redirect_server, redirect_thread = _start_server(
            _cookie_recording_handler(received, target)
        )
        try:
            port = redirect_server.server_address[1]
            _fetch_url(
                f"http://{_PUBLIC_TEST_HOST}:{port}/start",
                {"Cookie": "session=abc"},
                validate_ssrf=True,
            )
        finally:
            _stop_server(redirect_server, redirect_thread)
    finally:
        _stop_server(landing_server, landing_thread)
    assert received["/start"] == ["session=abc"]
    assert received["/landing"] == [None]


def test_fetch_url_keeps_cookie_on_same_origin_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-origin redirects keep the Cookie header (only cross-origin drops)."""
    _patch_public_test_dns(monkeypatch)
    received: dict[str, list[str | None]] = {}

    server, thread = _start_server(_cookie_recording_handler(received, "/landing"))
    try:
        port = server.server_address[1]
        _fetch_url(
            f"http://{_PUBLIC_TEST_HOST}:{port}/start",
            {"Cookie": "session=abc"},
            validate_ssrf=True,
        )
    finally:
        _stop_server(server, thread)
    assert received["/start"] == ["session=abc"]
    assert received["/landing"] == ["session=abc"]


async def test_fetch_one_treats_blocked_url_as_per_page_failure(
    tmp_path,
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
    # Outcome, not error-type name: the blocked URL warned and saved
    # nothing; other results continue (see the mapping assertion below).
    assert "Failed to fetch http://127.0.0.1:9/secret" in warnings[0]
    assert list(tmp_path.iterdir()) == []
    assert mapping == {}


# --- Chrome path (validation logic only, via a fake tab) -------------------
# These tests exercise _fetch_page's SSRF checks, not real CDP behaviour:
# the fake tab records whether navigation was attempted and, for the
# final-URL case, reports a private final URL through the response handler.


class _FakeTab:
    """Minimal tab double: enable, then (optionally) navigate.

    Dispatch is by protocol state, not by send count: the first send
    enables the network domain, and any later send is the navigate —
    the same enable-before-navigate order ``_fetch_page`` itself uses.
    """

    def __init__(self, final_url: str | None) -> None:
        self.final_url = final_url
        self.handler: Any = None
        self.enabled = False
        self.navigated: list[str] = []
        self.closed = False

    def add_handler(self, _event_type: object, handler: Any) -> None:
        self.handler = handler

    async def send(self, _cmd: object) -> Any:
        if not self.enabled:
            self.enabled = True
            return None  # cdp.network.enable()
        # cdp.page.navigate(): report the final URL through the handler
        # (buffered, since nav_loader_id is not set yet) and return it.
        assert self.final_url is not None
        event = SimpleNamespace(
            loader_id="loader1",
            response=SimpleNamespace(
                url=self.final_url,
                headers={"content-type": "text/html"},
            ),
        )
        await self.handler(event)
        self.navigated.append(self.final_url)
        return ("frame1", "loader1", None)

    async def wait(self) -> None:
        return None

    async def get_content(self) -> str:
        return "<html></html>"

    async def close(self) -> None:
        self.closed = True


class _FakeBrowser:
    def __init__(self, tab: _FakeTab) -> None:
        self.tab = tab

    async def get(self, _url: str, new_tab: bool = True) -> _FakeTab:
        return self.tab


@pytest.mark.skipif(not _HAS_ZENDRIVER, reason="zendriver not installed")
async def test_fetch_page_rejects_private_initial_url_before_navigation() -> None:
    tab = _FakeTab(final_url=None)
    with pytest.raises(SsrfBlockedError):
        await _fetch_page(  # type: ignore[arg-type]
            _FakeBrowser(tab), "http://127.0.0.1:9/secret", validate_ssrf=True
        )
    # No navigation was ever attempted.
    assert tab.navigated == []


@pytest.mark.skipif(not _HAS_ZENDRIVER, reason="zendriver not installed")
async def test_fetch_page_rejects_private_final_url_after_redirect() -> None:
    tab = _FakeTab(final_url="http://127.0.0.1:9/secret")
    with pytest.raises(SsrfBlockedError):
        await _fetch_page(  # type: ignore[arg-type]
            _FakeBrowser(tab),
            "http://93.184.216.34/start",
            validate_ssrf=True,
        )
    # Outcome only: the private final URL was rejected before any
    # content was returned. (Navigation itself happened - Chrome
    # follows redirects itself - but this test must not lock in the
    # redirect-gap behaviour; see issue #476 for the interception fix.)
