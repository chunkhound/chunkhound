"""Regression tests for explicit SSL verification handling."""

import http.server
import json
import ssl
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest

from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider
from chunkhound.providers.llm.openai_llm_provider import OpenAILLMProvider


def create_self_signed_cert() -> tuple[Path, Path]:
    """
    Create a self-signed certificate for testing.

    Returns:
        Tuple of (cert_file_path, key_file_path)
    """
    cert_dir = Path(tempfile.mkdtemp())
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    # Generate self-signed certificate using openssl
    # This simulates what corporate/internal servers often use
    result = subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key_file),
            "-out",
            str(cert_file),
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=localhost/O=Test/C=US",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.skip(f"OpenSSL not available: {result.stderr}")

    return cert_file, key_file


class MockOpenAIEmbeddingServer(http.server.BaseHTTPRequestHandler):
    """
    Mock OpenAI-compatible server that responds to embedding requests.
    This simulates servers like Ollama, LocalAI, or corporate OpenAI proxies.
    """
    def do_POST(self):
        """Handle POST requests to /v1/embeddings endpoint."""
        if self.path == "/v1/embeddings":
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            request_body = self.rfile.read(content_length)

            try:
                request_data = json.loads(request_body.decode())
                input_texts = request_data.get("input", [])
                if isinstance(input_texts, str):
                    input_texts = [input_texts]

                # Mock embedding response (same format as OpenAI)
                embeddings_data = []
                for i, _text in enumerate(input_texts):
                    embeddings_data.append({
                        "object": "embedding",
                        "index": i,
                        "embedding": [0.1] * 1536,
                    })

                response = {
                    "object": "list",
                    "data": embeddings_data,
                    "model": request_data.get("model", "text-embedding-3-small"),
                    "usage": {
                        "prompt_tokens": sum(len(text.split()) for text in input_texts),
                        "total_tokens": sum(len(text.split()) for text in input_texts)
                    }
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "Invalid JSON"}')
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Not found"}')

    def log_message(self, format, *args):
        """Suppress server logs to avoid cluttering test output."""
        pass


class MockOpenAILLMServer(http.server.BaseHTTPRequestHandler):
    """Mock OpenAI-compatible LLM server for chat completions."""

    def do_POST(self):
        """Handle POST requests to /v1/chat/completions."""
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Not found"}')
            return

        content_length = int(self.headers.get("Content-Length", 0))
        request_body = self.rfile.read(content_length)

        try:
            request_data = json.loads(request_body.decode())
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid JSON"}')
            return

        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": request_data.get("model", "gpt-test"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Local HTTPS LLM endpoint is working.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 6,
                "total_tokens": 10,
            },
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        """Suppress server logs to avoid cluttering test output."""
        pass


class HTTPSTestServer:
    """Helper class to manage HTTPS test server lifecycle."""

    def __init__(self, cert_file: Path, key_file: Path, handler_cls):
        self.cert_file = cert_file
        self.key_file = key_file
        self.handler_cls = handler_cls
        self.server = None
        self.server_thread = None
        self.port = None

    def start(self) -> str:
        """Start the HTTPS server and return the base URL."""
        # Create HTTP server
        self.server = http.server.HTTPServer(("localhost", 0), self.handler_cls)

        # Create SSL context with self-signed certificate
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(self.cert_file, self.key_file)

        # Wrap server socket with SSL
        self.server.socket = ssl_context.wrap_socket(
            self.server.socket, server_side=True
        )

        self.port = self.server.server_address[1]
        base_url = f"https://localhost:{self.port}/v1"

        # Start server in background thread
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

        # Give server time to start
        time.sleep(0.1)

        return base_url

    def stop(self):
        """Stop the HTTPS server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join(timeout=1)


class HTTPTestServer:
    """Helper to manage plain HTTP test servers."""

    def __init__(self, handler_cls):
        self.handler_cls = handler_cls
        self.server = None
        self.server_thread = None
        self.port = None

    def start(self) -> str:
        """Start the HTTP server and return the base URL."""
        self.server = http.server.HTTPServer(("localhost", 0), self.handler_cls)
        self.port = self.server.server_address[1]
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        time.sleep(0.1)
        return f"http://localhost:{self.port}/v1"

    def stop(self):
        """Stop the HTTP server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join(timeout=1)


@pytest.mark.asyncio
async def test_self_signed_https_requires_explicit_ssl_disable():
    """Custom HTTPS endpoints should fail by default with self-signed certs."""
    cert_file, key_file = create_self_signed_cert()
    server = HTTPSTestServer(cert_file, key_file, MockOpenAIEmbeddingServer)

    try:
        base_url = server.start()
        provider = OpenAIEmbeddingProvider(
            base_url=base_url,
            api_key="sk-test-key-like-user-has",
            model="bge-en-icl",
        )

        with pytest.raises(Exception):
            await provider.embed(["test text for embedding"])
    finally:
        server.stop()
        cert_file.unlink(missing_ok=True)
        key_file.unlink(missing_ok=True)
        cert_file.parent.rmdir()


@pytest.mark.asyncio
async def test_self_signed_https_succeeds_when_ssl_verify_disabled():
    """ssl_verify=false should allow trusted self-signed local endpoints."""
    cert_file, key_file = create_self_signed_cert()
    server = HTTPSTestServer(cert_file, key_file, MockOpenAIEmbeddingServer)

    try:
        base_url = server.start()
        provider = OpenAIEmbeddingProvider(
            base_url=base_url,
            api_key="test-key",
            model="text-embedding-3-small",
            ssl_verify=False,
        )

        embeddings = await provider.embed(["test text"])

        assert len(embeddings) == 1
    finally:
        server.stop()
        cert_file.unlink(missing_ok=True)
        key_file.unlink(missing_ok=True)
        cert_file.parent.rmdir()


@pytest.mark.asyncio
async def test_regular_http_works_fine():
    """Control test: regular HTTP should continue to work unchanged."""
    # Create regular HTTP server (no SSL)
    server = http.server.HTTPServer(("localhost", 0), MockOpenAIEmbeddingServer)
    port = server.server_address[1]
    base_url = f"http://localhost:{port}/v1"

    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    time.sleep(0.1)  # Give server time to start

    try:
        provider = OpenAIEmbeddingProvider(
            base_url=base_url,
            api_key="test-key",
            model="text-embedding-3-small",
        )

        embeddings = await provider.embed(["test text"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1)


@pytest.mark.asyncio
async def test_llm_self_signed_https_requires_explicit_ssl_disable():
    """LLM requests should fail closed by default for self-signed custom endpoints."""
    cert_file, key_file = create_self_signed_cert()
    server = HTTPSTestServer(cert_file, key_file, MockOpenAILLMServer)

    try:
        base_url = server.start()
        provider = OpenAILLMProvider(
            base_url=base_url,
            model="llama3.2",
            api_key=None,
        )

        with pytest.raises(RuntimeError):
            await provider.complete("Explain the auth flow")
    finally:
        server.stop()
        cert_file.unlink(missing_ok=True)
        key_file.unlink(missing_ok=True)
        cert_file.parent.rmdir()


@pytest.mark.asyncio
async def test_llm_self_signed_https_succeeds_when_ssl_verify_disabled():
    """Explicit ssl_verify=false should allow local HTTPS LLM endpoints."""
    cert_file, key_file = create_self_signed_cert()
    server = HTTPSTestServer(cert_file, key_file, MockOpenAILLMServer)

    try:
        base_url = server.start()
        provider = OpenAILLMProvider(
            base_url=base_url,
            model="llama3.2",
            api_key=None,
            ssl_verify=False,
        )

        response = await provider.complete("Explain the auth flow")

        assert response.content == "Local HTTPS LLM endpoint is working."
    finally:
        server.stop()
        cert_file.unlink(missing_ok=True)
        key_file.unlink(missing_ok=True)
        cert_file.parent.rmdir()


@pytest.mark.asyncio
async def test_llm_regular_http_works_fine():
    """Control test: plain HTTP LLM endpoints should continue to work."""
    server = HTTPTestServer(MockOpenAILLMServer)
    base_url = server.start()

    try:
        provider = OpenAILLMProvider(
            base_url=base_url,
            model="llama3.2",
            api_key=None,
        )

        response = await provider.complete("Explain the auth flow")

        assert response.content == "Local HTTPS LLM endpoint is working."
        assert response.tokens_used == 10
    finally:
        server.stop()
