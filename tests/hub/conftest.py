"""Local HTTP(S) publisher used only at the transport's external boundary."""

import json
import socket
import ssl
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest


@contextmanager
def hub_server(context: ssl.SSLContext | None = None) -> Iterator[SimpleNamespace]:
    state = SimpleNamespace(
        resources={},
        requests=[],
        encodings=[],
        ignore_range=False,
        bad_range=False,
        redirects={},
        redirect_status=302,
        status=200,
        statuses=[],
        interrupts=[],
        encoding=None,
        content_range=None,
        omit_length=False,
        before_get=None,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP handler API
            state.requests.append((self.path, self.headers.get("Range")))
            if state.before_get is not None:
                state.before_get(self.path)
            state.encodings.append(self.headers.get("Accept-Encoding"))
            if self.path in state.redirects:
                self.send_response(state.redirect_status)
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
    if context is not None:
        server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    scheme = "https" if context is not None else "http"
    state.url = f"{scheme}://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def hub_http() -> Iterator[SimpleNamespace]:
    with hub_server() as server:
        yield server


@pytest.fixture
def embedding_http() -> Iterator[SimpleNamespace]:
    state = SimpleNamespace(requests=[], url="")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP handler API
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append(body)
            if self.path != "/v1/embeddings":
                self.send_error(404)
                return
            texts = body["input"]
            if isinstance(texts, str):
                texts = [texts]
            dims = body.get("dimensions", 8)
            payload = {
                "object": "list",
                "model": body["model"],
                "data": [
                    {"object": "embedding", "index": i, "embedding": [0.25] * dims}
                    for i in range(len(texts))
                ],
                "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
            }
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture(scope="session")
def hub_certificate(tmp_path_factory):
    directory = tmp_path_factory.mktemp("hub-tls")
    certificate, key = directory / "cert.pem", directory / "key.pem"
    # The repository's existing TLS tests also use the OpenSSL executable.
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
            "-days",
            "2",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    return certificate, key


@pytest.fixture
def hub_https(hub_certificate) -> Iterator[SimpleNamespace]:
    certificate, key = hub_certificate
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    with hub_server(context) as server:
        yield server


@pytest.fixture
def hub_trusted_certificate(hub_certificate, monkeypatch):
    certificate, _ = hub_certificate
    create_default_context = ssl.create_default_context

    def with_test_ca(*args, **kwargs):
        context = create_default_context(*args, **kwargs)
        # Extend only the test trust store; keep real certificate/hostname checks.
        context.load_verify_locations(cafile=str(certificate))
        return context

    monkeypatch.setattr(ssl, "create_default_context", with_test_ca)


@pytest.fixture
def hub_network(monkeypatch):
    getaddrinfo = socket.getaddrinfo
    connect = socket.socket.connect

    def resolve_test_host(host, port, *args, **kwargs):
        if host in {"hub.internal", b"hub.internal"}:
            host = "127.0.0.1"
        return getaddrinfo(host, port, *args, **kwargs)

    def connect_test_address(sock, address):
        # Numeric hosts bypass DNS in anyio/asyncio, so route at the socket edge.
        if address[0] == "192.0.2.1":
            address = ("127.0.0.1", address[1])
        return connect(sock, address)

    # Exercise real requests for non-loopback URLs without an external network.
    monkeypatch.setattr(socket, "getaddrinfo", resolve_test_host)
    monkeypatch.setattr(socket.socket, "connect", connect_test_address)
