"""
Tests for SSL certificate handling with OpenAI-compatible endpoints.

Covers custom endpoints (Ollama, LocalAI, corporate proxies) that use
self-signed certificates, verifiable via verify_ssl=False.
"""

import asyncio
import http.server
import json
import ssl
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from chunkhound.api.cli.setup_wizard import _fetch_models_with_ssl_fallback
from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider


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
                for i, text in enumerate(input_texts):
                    embeddings_data.append(
                        {
                            "object": "embedding",
                            "index": i,
                            "embedding": [0.1] * 1536,  # Mock 1536-dim embedding
                        }
                    )

                response = {
                    "object": "list",
                    "data": embeddings_data,
                    "model": request_data.get("model", "text-embedding-3-small"),
                    "usage": {
                        "prompt_tokens": sum(len(text.split()) for text in input_texts),
                        "total_tokens": sum(len(text.split()) for text in input_texts),
                    },
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


class HTTPSTestServer:
    """Helper class to manage HTTPS test server lifecycle."""

    def __init__(self, cert_file: Path, key_file: Path):
        self.cert_file = cert_file
        self.key_file = key_file
        self.server = None
        self.server_thread = None
        self.port = None

    def start(self) -> str:
        """Start the HTTPS server and return the base URL."""
        # Create HTTP server
        self.server = http.server.HTTPServer(
            ("localhost", 0), MockOpenAIEmbeddingServer
        )

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


@pytest.mark.asyncio
async def test_self_signed_cert_works_with_verify_ssl_false():
    """
    Self-signed HTTPS endpoints must be reachable when verify_ssl=False.

    Starts a local HTTPS server with a self-signed certificate and confirms
    OpenAIEmbeddingProvider can generate embeddings through it.
    """
    # Create self-signed certificate (like corporate servers often use)
    cert_file, key_file = create_self_signed_cert()

    # Start mock HTTPS server with self-signed certificate
    server = HTTPSTestServer(cert_file, key_file)

    try:
        base_url = server.start()

        # Create OpenAI provider with SSL verification disabled for self-signed cert
        provider = OpenAIEmbeddingProvider(
            base_url=base_url,
            api_key="sk-test-key-like-user-has",
            model="bge-en-icl",
            verify_ssl=False,  # bypass self-signed cert verification
        )

        embeddings = await provider.embed(["test text for embedding"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536  # Mock embedding dimension

    finally:
        server.stop()
        # Cleanup certificate files
        cert_file.unlink(missing_ok=True)
        key_file.unlink(missing_ok=True)
        cert_file.parent.rmdir()


@pytest.mark.asyncio
async def test_regular_http_works_fine():
    """
    Control: plain HTTP endpoints work without any SSL configuration.
    """
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
            base_url=base_url, api_key="test-key", model="text-embedding-3-small"
        )

        # This should work fine with HTTP
        embeddings = await provider.embed(["test text"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536  # Mock embedding dimension

    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1)


if __name__ == "__main__":
    # Allow running this test directly for debugging
    asyncio.run(test_self_signed_cert_works_with_verify_ssl_false())


# ---------------------------------------------------------------------------
# Unit tests for _fetch_models_with_ssl_fallback
# ---------------------------------------------------------------------------

MODULE = "chunkhound.api.cli.setup_wizard"


@pytest.mark.asyncio
async def test_ssl_fallback_phase1_succeeds():
    """Phase 1 success → return immediately, no probe, verify_ssl=True."""
    with patch(
        f"{MODULE}._fetch_available_models", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.return_value = (["model-a"], False)
        models, needs_auth, verify_ssl = await _fetch_models_with_ssl_fallback(
            "https://custom.local", api_key=None
        )
    mock_fetch.assert_called_once()
    assert models == ["model-a"]
    assert verify_ssl is True


@pytest.mark.asyncio
async def test_ssl_fallback_user_accepts():
    """SSL error detected, user accepts → re-fetch with SSL disabled."""
    probe_models = ["model-a"]
    with (
        patch(
            f"{MODULE}._fetch_available_models", new_callable=AsyncMock
        ) as mock_fetch,
        patch(f"{MODULE}.rich_confirm", new_callable=AsyncMock, return_value=True),
    ):
        # Phase 1 fails, Phase 2 probe succeeds → result reused directly
        mock_fetch.side_effect = [
            (None, True),
            (probe_models, False),
        ]
        models, needs_auth, verify_ssl = await _fetch_models_with_ssl_fallback(
            "https://custom.local", api_key=None
        )
    assert mock_fetch.call_count == 2
    assert models == probe_models
    assert verify_ssl is False


@pytest.mark.asyncio
async def test_ssl_fallback_user_declines():
    """SSL error detected, user declines → keep SSL enabled, return original failure."""
    with (
        patch(
            f"{MODULE}._fetch_available_models", new_callable=AsyncMock
        ) as mock_fetch,
        patch(f"{MODULE}.rich_confirm", new_callable=AsyncMock, return_value=False),
    ):
        mock_fetch.side_effect = [
            (None, True),  # Phase 1: fails
            (["m"], False),  # Phase 2 probe: SSL was the issue
        ]
        models, needs_auth, verify_ssl = await _fetch_models_with_ssl_fallback(
            "https://custom.local", api_key=None
        )
    assert verify_ssl is True
    assert models is None


@pytest.mark.asyncio
async def test_ssl_fallback_not_ssl_issue():
    """Probe also fails → SSL was not the cause, return original failure."""
    with patch(
        f"{MODULE}._fetch_available_models", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.side_effect = [
            (None, True),  # Phase 1: fails
            (None, True),  # Phase 2 probe: also fails → not SSL
        ]
        models, needs_auth, verify_ssl = await _fetch_models_with_ssl_fallback(
            "https://custom.local", api_key=None
        )
    assert models is None
    assert verify_ssl is True


@pytest.mark.asyncio
async def test_ssl_fallback_skipped_for_http():
    """HTTP endpoints skip the SSL fallback entirely."""
    with patch(
        f"{MODULE}._fetch_available_models", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.return_value = (None, True)
        models, needs_auth, verify_ssl = await _fetch_models_with_ssl_fallback(
            "http://custom.local", api_key=None
        )
    mock_fetch.assert_called_once()  # Only Phase 1, no probe
    assert verify_ssl is True


@pytest.mark.asyncio
async def test_ssl_fallback_skipped_for_official_openai():
    """Official OpenAI endpoints skip the SSL fallback entirely."""
    with patch(
        f"{MODULE}._fetch_available_models", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.return_value = (None, True)
        models, needs_auth, verify_ssl = await _fetch_models_with_ssl_fallback(
            "https://api.openai.com/v1", api_key=None
        )
    mock_fetch.assert_called_once()
    assert verify_ssl is True


@pytest.mark.asyncio
async def test_ssl_fallback_with_api_key_still_probes():
    """Probe runs even with api_key set; Phase 2 uses api_key=None (no leak)."""
    probe_models = ["model-a"]
    with (
        patch(
            f"{MODULE}._fetch_available_models", new_callable=AsyncMock
        ) as mock_fetch,
        patch(f"{MODULE}.rich_confirm", new_callable=AsyncMock, return_value=True),
    ):
        mock_fetch.side_effect = [
            (None, True),  # Phase 1: fails (with api_key + SSL)
            (probe_models, False),  # Phase 2: probe succeeds (api_key=None, no SSL)
        ]
        models, needs_auth, verify_ssl = await _fetch_models_with_ssl_fallback(
            "https://custom.local", api_key="sk-secret"
        )
    assert mock_fetch.call_count == 2
    # Phase 2 probe must NOT send credentials over the unverified connection
    assert mock_fetch.call_args_list[1].kwargs["api_key"] is None
    assert models == probe_models
    assert verify_ssl is False
