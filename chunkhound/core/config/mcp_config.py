"""MCP (Model Context Protocol) server configuration for ChunkHound.

This module provides configuration for the MCP server including
transport type, network settings, and server behavior.
"""

import argparse
import os
from typing import Any, Literal

from pydantic import BaseModel, Field


class MCPConfig(BaseModel):
    """Configuration for MCP server operation.

    Controls how the MCP server operates including transport type,
    network configuration, and server behavior.
    """

    # Transport configuration
    transport: Literal["stdio", "http"] = Field(
        default="stdio", description="Transport type for MCP server"
    )

    # HTTP transport settings
    host: str = Field(default="0.0.0.0", description="Host to bind HTTP server to")

    port: int = Field(
        default=3000, description="Port for HTTP server (0 for OS-assigned port)"
    )

    # Internal settings - not exposed to users
    cors: bool = Field(default=True, description="Internal CORS setting")
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["*"], description="Internal CORS origins"
    )
    max_concurrent_requests: int = Field(default=10, description="Internal concurrency")

    def get_server_url(self) -> str:
        """Get the full server URL for HTTP transport."""
        if self.transport != "http":
            raise ValueError("Server URL only available for HTTP transport")

        return f"http://{self.host}:{self.port}"

    def is_http_transport(self) -> bool:
        """Check if using HTTP transport."""
        return self.transport == "http"

    def is_stdio_transport(self) -> bool:
        """Check if using stdio transport."""
        return self.transport == "stdio"

    def get_transport_config(self) -> dict:
        """Get transport-specific configuration."""
        if self.transport == "http":
            return {
                "host": self.host,
                "port": self.port,
                "cors": self.cors,
                "allowed_origins": self.allowed_origins if self.cors else [],
                "max_concurrent_requests": self.max_concurrent_requests,
            }
        else:  # stdio
            return {
                "max_concurrent_requests": 1,  # stdio is inherently sequential
            }

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add MCP-related CLI arguments."""
        parser.add_argument(
            "--stdio",
            action="store_true",
            help="Use stdio transport (default)",
        )

        parser.add_argument(
            "--http",
            action="store_true",
            help="Use HTTP transport instead of stdio",
        )

        parser.add_argument(
            "--port",
            type=int,
            help="Port for HTTP transport",
        )

        parser.add_argument(
            "--host",
            help="Host for HTTP transport",
        )

        parser.add_argument(
            "--show-setup",
            action="store_true",
            help="Display MCP setup instructions and exit",
        )

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        """Load MCP config from environment variables."""
        config = {}

        if transport := os.getenv("CHUNKHOUND_MCP__TRANSPORT"):
            config["transport"] = transport
        if port := os.getenv("CHUNKHOUND_MCP__PORT"):
            config["port"] = int(port)
        if host := os.getenv("CHUNKHOUND_MCP__HOST"):
            config["host"] = host

        return config

    @classmethod
    def extract_cli_overrides(cls, args: Any) -> dict[str, Any]:
        """Extract MCP config from CLI arguments."""
        overrides = {}

        # Handle transport boolean flags mapping to transport string
        if hasattr(args, "http") and args.http:
            overrides["transport"] = "http"
        elif hasattr(args, "stdio") and args.stdio:
            overrides["transport"] = "stdio"

        if hasattr(args, "port") and args.port is not None:
            overrides["port"] = args.port
        if hasattr(args, "host") and args.host is not None:
            overrides["host"] = args.host

        return overrides

    def __repr__(self) -> str:
        """String representation of MCP configuration."""
        if self.transport == "http":
            return f"MCPConfig(transport={self.transport}, url={self.get_server_url()})"
        else:
            return f"MCPConfig(transport={self.transport})"
