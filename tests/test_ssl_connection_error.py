"""
Test to reproduce SSL connection error with OpenAI-compatible endpoints.

This test reproduces the exact error users experience when using custom
OpenAI-compatible endpoints with self-signed certificates, like:
https://pdc-llm-srv1/llm/v1

The test should INITIALLY FAIL, demonstrating the bug exists.
"""

import asyncio
import http.server
import json
import os
import ssl
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Tuple

import pytest

from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider


def create_self_signed_cert() -> Tuple[Path, Path]:
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
    result = subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", str(key_file), "-out", str(cert_file),
        "-days", "1", "-nodes", 
        "-subj", "/CN=localhost/O=Test/C=US"
    ], capture_output=True, text=True)
    
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
            content_length = int(self.headers.get('Content-Length', 0))
            request_body = self.rfile.read(content_length)
            
            try:
                request_data = json.loads(request_body.decode())
                input_texts = request_data.get("input", [])
                if isinstance(input_texts, str):
                    input_texts = [input_texts]
                
                # Mock embedding response (same format as OpenAI)
                embeddings_data = []
                for i, text in enumerate(input_texts):
                    embeddings_data.append({
                        "object": "embedding",
                        "index": i,
                        "embedding": [0.1] * 1536  # Mock 1536-dim embedding
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
        self.server = http.server.HTTPServer(('localhost', 0), MockOpenAIEmbeddingServer)
        
        # Create SSL context with self-signed certificate
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(self.cert_file, self.key_file)
        
        # Wrap server socket with SSL
        self.server.socket = ssl_context.wrap_socket(
            self.server.socket,
            server_side=True
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
async def test_ssl_connection_error_reproduces_user_issue():
    """
    Test that reproduces the SSL connection error with self-signed certificates.
    
    This test demonstrates the exact issue users face when connecting to
    OpenAI-compatible endpoints with self-signed certificates.
    
    Expected behavior: 
    - SHOULD FAIL initially (demonstrating the bug exists)
    - SHOULD PASS when SSL configuration fix is implemented
    
    The test attempts to generate embeddings from an HTTPS endpoint with
    self-signed certificate. Currently this fails with connection error.
    When fixed, it should successfully return embeddings.
    """
    # Create self-signed certificate (like corporate servers often use)
    cert_file, key_file = create_self_signed_cert()
    
    # Start mock HTTPS server with self-signed certificate
    server = HTTPSTestServer(cert_file, key_file)
    
    try:
        base_url = server.start()
        
        # Create OpenAI provider configured exactly like the user's setup
        provider = OpenAIEmbeddingProvider(
            base_url=base_url,
            api_key="sk-test-key-like-user-has",  # API key format like user's
            model="bge-en-icl"  # Exact model from user's config
        )
        
        # This call should:
        # - FAIL initially with "APIConnectionError: Connection error"
        # - PASS when SSL configuration is properly implemented
        embeddings = await provider.embed(["test text for embedding"])
        
        # If we get here, the SSL issue is fixed!
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536  # Mock embedding dimension
        print("✓ SSL connection issue is FIXED - embeddings generated successfully!")
        
    finally:
        server.stop()
        # Cleanup certificate files
        cert_file.unlink(missing_ok=True)
        key_file.unlink(missing_ok=True)
        cert_file.parent.rmdir()


@pytest.mark.asyncio 
async def test_httpx_ssl_verify_env_var_doesnt_work():
    """
    Test that the HTTPX_SSL_VERIFY=0 environment variable workaround
    suggested in the code comments doesn't actually work.
    
    Expected behavior:
    - SHOULD FAIL initially (proving env var workaround is broken)  
    - SHOULD PASS when proper SSL configuration is implemented
    
    This test sets HTTPX_SSL_VERIFY=0 but still expects connection to fail,
    proving the suggested workaround is ineffective.
    """
    cert_file, key_file = create_self_signed_cert()
    server = HTTPSTestServer(cert_file, key_file)
    
    # Set the environment variable as suggested in code comments
    original_ssl_verify = os.environ.get("HTTPX_SSL_VERIFY")
    os.environ["HTTPX_SSL_VERIFY"] = "0"
    
    try:
        base_url = server.start()
        
        provider = OpenAIEmbeddingProvider(
            base_url=base_url,
            api_key="test-key",
            model="text-embedding-3-small"
        )
        
        # This should fail initially (proving env var doesn't work)
        # When SSL config is fixed, this should pass
        embeddings = await provider.embed(["test text"])
        
        # If we get here, either the env var started working OR SSL config was fixed
        assert len(embeddings) == 1
        print("✓ HTTPX_SSL_VERIFY=0 environment variable now works, OR SSL config was fixed")
        
    finally:
        # Restore original environment
        if original_ssl_verify is None:
            os.environ.pop("HTTPX_SSL_VERIFY", None)
        else:
            os.environ["HTTPX_SSL_VERIFY"] = original_ssl_verify
        
        server.stop()
        cert_file.unlink(missing_ok=True)
        key_file.unlink(missing_ok=True)
        cert_file.parent.rmdir()


@pytest.mark.asyncio
async def test_regular_http_works_fine():
    """
    Control test: Verify that regular HTTP endpoints work fine.
    This proves the issue is specifically with HTTPS certificate verification.
    """
    # Create regular HTTP server (no SSL)
    server = http.server.HTTPServer(('localhost', 0), MockOpenAIEmbeddingServer)
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
            model="text-embedding-3-small"
        )
        
        # This should work fine with HTTP
        embeddings = await provider.embed(["test text"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536  # Mock embedding dimension
        
        print("✓ Regular HTTP works fine - confirms issue is HTTPS-specific")
        
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1)


if __name__ == "__main__":
    # Allow running this test directly for debugging
    asyncio.run(test_ssl_connection_error_reproduces_user_issue())