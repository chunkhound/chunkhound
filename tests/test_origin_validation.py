"""Tests for HTTP server origin validation.

Ensures the origin validation correctly rejects malicious origins
while allowing legitimate localhost requests.
"""

import pytest


class MockRequest:
    """Mock Starlette Request for testing."""

    def __init__(self, origin: str | None):
        # Use 'is not None' to distinguish None from empty string
        self._headers = {"Origin": origin} if origin is not None else {}

    @property
    def headers(self) -> dict[str, str]:
        return self._headers


class TestOriginValidation:
    """Test origin validation security."""

    @pytest.fixture
    def validate_origin(self):
        """Import the validation function."""
        from chunkhound.mcp_server.http_server import _validate_origin

        return _validate_origin

    # Valid origins that should be allowed
    @pytest.mark.parametrize(
        "origin",
        [
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "http://127.0.0.1",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
            "https://localhost",
            "https://localhost:443",
            "https://127.0.0.1",
            "https://127.0.0.1:8443",
        ],
    )
    def test_valid_origins_allowed(self, validate_origin, origin: str):
        """Valid localhost origins should be allowed."""
        request = MockRequest(origin)
        result = validate_origin(request)
        assert result is None, f"Origin {origin} should be allowed"

    def test_no_origin_header_allowed(self, validate_origin):
        """Requests without Origin header should be allowed."""
        request = MockRequest(None)
        result = validate_origin(request)
        assert result is None

    # Attack vectors that should be rejected with 403 (forbidden)
    @pytest.mark.parametrize(
        "origin,description",
        [
            # Hostname spoofing via subdomain
            ("http://127.0.0.1.attacker.com", "subdomain of IP"),
            ("http://localhost.attacker.com", "subdomain of localhost"),
            # External domains
            ("http://attacker.com", "external domain"),
            ("http://evil.localhost.com", "evil with localhost substring"),
            ("https://malicious.127.0.0.1.io", "malicious with IP substring"),
            # Different IP addresses
            ("http://192.168.1.1", "internal network IP"),
            ("http://10.0.0.1", "private IP"),
            ("http://0.0.0.0", "all interfaces IP"),
            # IPv6 (not in allowed list)
            ("http://[::1]", "IPv6 localhost"),
            # Different schemes
            ("ftp://localhost", "FTP scheme"),
            ("file://localhost", "file scheme"),
            ("javascript://localhost", "javascript scheme"),
        ],
    )
    def test_malicious_origins_rejected(
        self, validate_origin, origin: str, description: str
    ):
        """Malicious origins should be rejected with 403."""
        request = MockRequest(origin)
        result = validate_origin(request)
        assert result is not None, f"Origin {origin} ({description}) should be rejected"
        assert result.status_code == 403, f"Expected 403 for {origin}"

    @pytest.mark.parametrize(
        "origin",
        [
            "",  # Empty string -> 400
            "not-a-url",  # No scheme -> 403 (scheme check fails)
            "://localhost",  # Missing scheme -> 403 (empty scheme)
            "http://",  # Missing host -> 403 (hostname is None)
        ],
    )
    def test_malformed_origins_rejected(self, validate_origin, origin: str):
        """Malformed origins should be rejected (400 or 403)."""
        request = MockRequest(origin)
        result = validate_origin(request)
        assert result is not None, f"Malformed origin {origin!r} should be rejected"
        assert result.status_code in (400, 403), f"Expected 400/403 for {origin!r}"

    @pytest.mark.parametrize(
        "origin",
        [
            "http://127.0.0.1:8080.attacker.com",
            "http://localhost:3000.evil.com",
        ],
    )
    def test_port_suffix_attack_rejected(self, validate_origin, origin: str):
        """Port suffix attacks should be rejected with 400 (invalid port)."""
        request = MockRequest(origin)
        result = validate_origin(request)
        assert result is not None, f"Port suffix attack {origin!r} should be rejected"
        assert result.status_code == 400, f"Expected 400 for {origin!r}"


class TestOriginBypassPrevention:
    """Specific tests for the prefix-matching bypass that was fixed."""

    @pytest.fixture
    def validate_origin(self):
        from chunkhound.mcp_server.http_server import _validate_origin

        return _validate_origin

    def test_port_suffix_bypass_prevented(self, validate_origin):
        """The original bug: '127.0.0.1:1234.attacker.com' passed prefix check.

        Old code did: origin.startswith("http://127.0.0.1:")
        This would match "http://127.0.0.1:1234.attacker.com" incorrectly.

        The fix rejects this as malformed (400) since the port is invalid.
        """
        # This was the specific attack vector mentioned in the code review
        request = MockRequest("http://127.0.0.1:1234.attacker.com")
        result = validate_origin(request)
        assert result is not None, "Port suffix bypass should be prevented"
        assert result.status_code == 400  # Malformed port

    def test_localhost_port_suffix_bypass_prevented(self, validate_origin):
        """Same bypass with localhost."""
        request = MockRequest("http://localhost:8080.evil.com")
        result = validate_origin(request)
        assert result is not None, "Localhost port suffix bypass should be prevented"
        assert result.status_code == 400  # Malformed port

    def test_legitimate_port_still_works(self, validate_origin):
        """Ensure real ports still work after the fix."""
        for port in [80, 443, 3000, 5173, 8000, 8080, 9000]:
            request = MockRequest(f"http://127.0.0.1:{port}")
            result = validate_origin(request)
            assert result is None, f"Legitimate port {port} should be allowed"
