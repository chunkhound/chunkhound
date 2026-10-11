"""Local HTTP publisher used only at the transport's external boundary."""

import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest


@pytest.fixture
def hub_http() -> Iterator[SimpleNamespace]:
    state = SimpleNamespace(
        resources={},
        requests=[],
        encodings=[],
        ignore_range=False,
        bad_range=False,
        redirects={},
        status=200,
        statuses=[],
        interrupts=[],
        encoding=None,
        content_range=None,
        omit_length=False,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP handler API
            state.requests.append((self.path, self.headers.get("Range")))
            state.encodings.append(self.headers.get("Accept-Encoding"))
            if self.path in state.redirects:
                self.send_response(302)
                location = state.redirects[self.path]
                if location is not None:
                    self.send_header("Location", location)
                self.end_headers()
                return
            body = state.resources.get(self.path)
            status = state.statuses.pop(0) if state.statuses else state.status
            if body is None or status != 200:
                self.send_response(404 if body is None else status)
                self.end_headers()
                self.wfile.write(b"response-body-secret")
                return
            start = 0
            ranged = self.headers.get("Range") and not state.ignore_range
            if ranged:
                start = int(
                    self.headers["Range"].removeprefix("bytes=").removesuffix("-")
                )
                if start >= len(body):
                    self.send_response(416)
                    self.end_headers()
                    return
            self.send_response(206 if ranged else 200)
            if not state.omit_length:
                self.send_header("Content-Length", str(len(body) - start))
            if state.encoding is not None:
                self.send_header("Content-Encoding", state.encoding)
            if ranged:
                actual = start + 1 if state.bad_range else start
                content_range = state.content_range
                if content_range is None:
                    content_range = f"bytes {actual}-{len(body) - 1}/{len(body)}"
                self.send_header("Content-Range", content_range)
            self.end_headers()
            payload = body[start:]
            if state.interrupts:
                payload = payload[: state.interrupts.pop(0)]
            try:
                self.wfile.write(payload)
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_WR)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
