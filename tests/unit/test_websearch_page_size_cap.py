"""Verify fetched pages are size-capped instead of buffered without bound.

A websearch target URL can be anything -- a mis-served multi-GB file, an
infinite stream -- and there's no way to know its size in advance. See
_MAX_PAGE_BYTES / _read_bounded / _check_page_size in
chunkhound/utils/websearch_core.py.
"""

from __future__ import annotations

import http.server
import threading
from collections.abc import Iterator

import pytest

from chunkhound.utils.websearch_core import _MAX_PAGE_BYTES, _fetch_url


@pytest.fixture
def oversized_page_server() -> Iterator[str]:
    """Background HTTP server whose /huge route exceeds _MAX_PAGE_BYTES."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 -- stdlib API
            if self.path == "/huge":
                body = b"x" * (_MAX_PAGE_BYTES + 1024)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/small":
                body = b"small page body"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_fetch_url_rejects_response_over_the_size_cap(
    oversized_page_server: str,
) -> None:
    with pytest.raises(ValueError, match="exceeds maximum page size"):
        _fetch_url(f"{oversized_page_server}/huge")


def test_fetch_url_allows_response_under_the_size_cap(
    oversized_page_server: str,
) -> None:
    _ct, body, _charset = _fetch_url(f"{oversized_page_server}/small")
    assert body == b"small page body"
