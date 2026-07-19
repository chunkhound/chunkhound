"""MCP (Model Context Protocol) server configuration for ChunkHound.

This module provides configuration for the MCP server, including
transport settings (stdio or HTTP) and server behavior.
"""

import argparse
import ipaddress
import os
from typing import Any, Literal

from pydantic import BaseModel, Field


def is_loopback_host(host: str) -> bool:
    """True if ``host`` is a loopback address/hostname.

    Handles "localhost" (case-insensitively) plus any valid loopback IP
    literal — not just the exact strings "127.0.0.1"/"::1" — so e.g.
    "127.0.0.2" or the expanded "0:0:0:0:0:0:0:1" IPv6 form are also
    recognized as loopback rather than tripping the non-loopback-host check.
    """
    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class MCPConfig(BaseModel):
    """Configuration for MCP server operation.

    Controls how the MCP server operates, including transport (stdio or
    HTTP) and HTTP-specific settings (host, port, auth token, CORS).
    """

    # Transport configuration
    transport: Literal["stdio", "http"] = Field(
        default="stdio", description="Transport type for MCP server"
    )

    # HTTP transport settings
    host: str = Field(
        default="127.0.0.1", description="Host to bind the HTTP transport to"
    )
    port: int = Field(
        default=5173,
        ge=1,
        le=65535,
        description="Port to bind the HTTP transport to",
    )
    auth_token: str | None = Field(
        default=None,
        description="Bearer token required to authenticate HTTP transport requests",
    )
    cors: bool = Field(default=False, description="Enable CORS for the HTTP transport")

    # Internal settings
    max_concurrent_requests: int = Field(
        default=1, description="Max concurrent requests (stdio is sequential)"
    )

    def is_stdio_transport(self) -> bool:
        """Check if using stdio transport."""
        return self.transport == "stdio"

    def get_transport_config(self) -> dict:
        """Get transport-specific configuration."""
        return {
            "max_concurrent_requests": 1,  # stdio is inherently sequential
        }

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add MCP-related CLI arguments."""
        parser.add_argument(
            "--stdio",
            action="store_true",
            help="Use stdio transport (default, without the daemon)",
        )

        parser.add_argument(
            "--show-setup",
            action="store_true",
            help="Display MCP setup instructions and exit",
        )

        parser.add_argument(
            "--transport",
            choices=["stdio", "http"],
            help="Transport type for MCP server (default: stdio)",
        )

        parser.add_argument(
            "--host",
            type=str,
            help="Host to bind the HTTP transport to (default: 127.0.0.1)",
        )

        parser.add_argument(
            "--port",
            type=int,
            help="Port to bind the HTTP transport to (default: 5173)",
        )

        parser.add_argument(
            "--auth-token",
            type=str,
            help=(
                "Bearer token required to authenticate HTTP transport requests. "
                "Generate one with: "
                'python -c "import secrets; print(secrets.token_urlsafe(32))"'
            ),
        )

        parser.add_argument(
            "--cors",
            action="store_true",
            help="Enable CORS for the HTTP transport (for browser-based clients)",
        )

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        """Load MCP config from environment variables."""
        config: dict[str, Any] = {}

        if transport := os.getenv("CHUNKHOUND_MCP__TRANSPORT"):
            config["transport"] = transport
        if host := os.getenv("CHUNKHOUND_MCP__HOST"):
            config["host"] = host
        if port_str := os.getenv("CHUNKHOUND_MCP__PORT"):
            try:
                config["port"] = int(port_str)
            except ValueError:
                raise ValueError(
                    f"CHUNKHOUND_MCP__PORT must be an integer, got: {port_str!r}"
                )
        if auth_token := os.getenv("CHUNKHOUND_MCP__AUTH_TOKEN"):
            config["auth_token"] = auth_token
        if cors := os.getenv("CHUNKHOUND_MCP__CORS"):
            config["cors"] = cors.lower() in ("true", "1", "yes")

        return config

    @classmethod
    def extract_cli_overrides(cls, args: Any) -> dict[str, Any]:
        """Extract MCP config from CLI arguments."""
        overrides: dict[str, Any] = {}

        if hasattr(args, "transport") and args.transport is not None:
            overrides["transport"] = args.transport
        if hasattr(args, "host") and args.host is not None:
            overrides["host"] = args.host
        if hasattr(args, "port") and args.port is not None:
            overrides["port"] = args.port
        if hasattr(args, "auth_token") and args.auth_token is not None:
            overrides["auth_token"] = args.auth_token
        if hasattr(args, "cors") and args.cors:
            overrides["cors"] = args.cors

        return overrides

    def __repr__(self) -> str:
        """String representation of MCP configuration."""
        return f"MCPConfig(transport={self.transport})"
