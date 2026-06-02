"""Integration tests for ``_fetch_page`` — the CDP-driven page fetch.

Launches a real headless Chrome via zendriver and exercises the production
HTML and PDF branches end-to-end. Skipped when no Chrome binary is present
at any of the standard ``_CHROME_PATHS`` locations.

Each test gets its own browser (function-scoped fixture) so the PDF
branch's known one-tab leak cannot bleed into the HTML branch's target
count, regardless of test execution order.

The test resolves Chrome via ``_resolve_chrome_path`` so the same
explicit-path probe + version check that ``fetch_and_save`` relies on
runs here too — a Chrome <124 raises out of the resolver, matching
production's fail-loud contract for the silent ``Response.charset``
parse-failure loop on older Chrome.
"""

from __future__ import annotations

import asyncio
import http.server
import os
import threading
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from chunkhound.utils.websearch_core import (
    _CHROME_PATHS,
    _fetch_page,
    _resolve_chrome_path,
)

if TYPE_CHECKING:
    import zendriver as zd

# A small public PDF used by the spike; same URL keeps test behavior
# aligned with what the migration was verified against.
PDF_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
HTML_URL = "https://example.com/"


pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not any(os.path.exists(p) for p in _CHROME_PATHS),
        reason="No Chrome binary found at standard _CHROME_PATHS locations",
    ),
]


def _page_target_count(browser: zd.Browser) -> int:
    """Count page-type targets currently held by the browser.

    Mirrors the spike's filtering (``getattr(t, "type_", None) == "page"``)
    so service workers and other non-page targets do not skew the baseline.
    """
    return sum(1 for t in browser.targets if getattr(t, "type_", None) == "page")


async def _wait_for_target_count(
    browser: zd.Browser, expected: int, timeout: float
) -> int:
    """Poll until ``browser.targets`` reports ``expected`` page targets.

    Returns the final count whether or not it matched. ``browser.targets``
    only updates when the ``Target.targetDestroyed`` event lands, which
    can lag tab.close() — especially on the PDF viewer path where Chrome
    holds the tab until the viewer releases.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        n = _page_target_count(browser)
        if n == expected or loop.time() >= deadline:
            return n
        await asyncio.sleep(0.1)


@pytest_asyncio.fixture
async def chrome_browser() -> AsyncIterator[zd.Browser]:
    """Per-test Chrome. Function-scoped so target counts start clean.

    Uses the production ``_resolve_chrome_path`` so the explicit-path
    probe + version check that ``fetch_and_save`` relies on is exercised
    here — a Chrome <124 or otherwise unverifiable binary returns ``None``
    and skips the test instead of letting it hit the silent-event-drop
    loop in zendriver's listener.
    """
    import zendriver as zd

    chrome_path = _resolve_chrome_path()
    if chrome_path is None:
        pytest.skip("No Chrome resolvable via _resolve_chrome_path")

    browser = await zd.start(
        headless=True,
        browser_args=["--headless=new"],
        browser_executable_path=chrome_path,
    )
    try:
        yield browser
    finally:
        try:
            await asyncio.wait_for(browser.stop(), timeout=10)
        except asyncio.TimeoutError:
            pass


async def test_fetch_page_html_branch(chrome_browser: zd.Browser) -> None:
    """HTML branch returns rendered DOM and tears down its tab.

    Poll (rather than snapshot) because ``browser.targets`` updates only
    when ``Target.targetDestroyed`` lands, which lags ``tab.close()`` by
    a fraction of a second. ``_fetch_page`` has already awaited the
    production close cap by the time we get here, so this budget is for
    event-propagation lag only.
    """
    baseline = _page_target_count(chrome_browser)

    ct, body, charset = await _fetch_page(chrome_browser, HTML_URL)
    assert ct == "text/html"
    assert isinstance(body, bytes) and len(body) > 0
    assert charset == "utf-8"
    assert b"<html" in body.lower() or b"<!doctype" in body.lower()

    final = await _wait_for_target_count(chrome_browser, baseline, timeout=5.0)
    assert final == baseline


async def test_fetch_page_pdf_branch(chrome_browser: zd.Browser) -> None:
    """PDF branch returns raw bytes; tab close is best-effort.

    ``tab.close()`` on the PDF viewer path waits on zendriver's internal
    10s ack (the PDF viewer is still ingesting the in-flight download),
    which exceeds ``_close_tab_quietly``'s 5s cap — so the PDF tab
    typically remains in ``browser.targets`` until ``browser.stop()``
    reaps it. Allow either outcome (clean close OR one leaked tab) but
    no more, so we detect both regressions (multi-tab leak) and
    improvements (future zendriver/Chrome fix that lets the close succeed).
    """
    baseline = _page_target_count(chrome_browser)

    ct, body, charset = await _fetch_page(chrome_browser, PDF_URL)
    assert ct == "application/pdf"
    assert isinstance(body, bytes) and body.startswith(b"%PDF-")
    assert charset == "utf-8"

    final = await _wait_for_target_count(chrome_browser, baseline, timeout=6.0)
    assert final in (baseline, baseline + 1), (
        f"PDF branch leaked unexpectedly many tabs: {final - baseline}"
    )


# Minimal valid PDF prefix — the test only asserts the %PDF- magic bytes,
# so the body doesn't need to be a structurally complete PDF.
_MINIMAL_PDF = b"%PDF-1.4\n%minimal\n"


@pytest.fixture
def cookie_pdf_server() -> Iterator[tuple[str, list[str]]]:
    """Background HTTP server serving cookie-gated PDF endpoints.

    Uses ``ThreadingHTTPServer`` (not ``HTTPServer``) so HTTP/1.1
    keep-alive connections from Chrome cannot starve the urllib refetch:
    a single-threaded server holding Chrome's keep-alive socket would
    block accept() of the subsequent urllib connection and deadlock.

    The handler is defined inside the fixture so it closes over the
    per-invocation ``received`` list — no class-level state, no
    monkey-patched server attribute.
    """
    received: list[str] = []

    class _CookieGatedPDFHandler(http.server.BaseHTTPRequestHandler):
        """Local server: /pdf gates a PDF on the ``session=valid`` cookie.

        Returns the PDF body only when ``session=valid`` is present;
        otherwise 302s to ``/login`` (so the test fails loudly if
        redirect-block is reverted).
        """

        def do_GET(self) -> None:  # noqa: N802 — stdlib API
            if self.path == "/pdf":
                cookie = self.headers.get("Cookie", "")
                received.append(cookie)
                if "session=valid" in cookie:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/pdf")
                    self.send_header("Content-Length", str(len(_MINIMAL_PDF)))
                    self.end_headers()
                    self.wfile.write(_MINIMAL_PDF)
                    return
                self.send_response(302)
                self.send_header("Location", "/login")
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), _CookieGatedPDFHandler
    )
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


async def test_fetch_page_pdf_forwards_browser_cookies(
    chrome_browser: zd.Browser,
    cookie_pdf_server: tuple[str, list[str]],
) -> None:
    """PDF refetch carries cookies present in Chrome's jar.

    The cookie is injected directly into Chrome via CDP rather than
    obtained through a real /auth round-trip — we are testing our
    contract ("if a cookie is in Chrome's jar, our refetch forwards it"),
    not Chrome's Set-Cookie handling. Locks in both halves of the
    contract:
      - Cookies extracted from Chrome reach the urllib refetch (the tail
        recorded Cookie header contains ``session=valid``).
      - Redirects in the cookie-bearing refetch are blocked: without the
        defense, the cookie-less fallback path (or any reverted refetch)
        would follow 302 → /login and return non-PDF content, failing
        the magic-byte assertion.
    """
    from zendriver import cdp

    base, received_cookie_headers = cookie_pdf_server

    await chrome_browser.cookies.set_all(
        [cdp.network.CookieParam(name="session", value="valid", url=f"{base}/")]
    )

    ct, body, _ = await _fetch_page(chrome_browser, f"{base}/pdf")

    assert ct == "application/pdf"
    assert body.startswith(b"%PDF-")
    # The urllib refetch is the LAST request hitting /pdf. Asserting on
    # the tail entry directly proves the cookie traveled via the urllib
    # path — not just that Chrome's own navigation succeeded.
    assert received_cookie_headers, "no /pdf requests recorded"
    assert "session=valid" in received_cookie_headers[-1]
