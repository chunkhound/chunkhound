"""MCP (Model Context Protocol) server configuration for ChunkHound.

This module provides configuration for the stdio MCP server
including transport settings and server behavior.
"""

import argparse
import os
from typing import Any, Literal

from pydantic import BaseModel, Field


class MCPConfig(BaseModel):
    """Configuration for MCP server operation.

    Controls how the MCP server operates (stdio transport only).
    """

    # Transport configuration - stdio only
    transport: Literal["stdio"] = Field(
        default="stdio", description="Transport type for MCP server (stdio only)"
    )

    # Internal settings
    max_concurrent_requests: int = Field(
        default=1, description="Max concurrent requests (stdio is sequential)"
    )

    def is_stdio_transport(self) -> bool:
        """Check if using stdio transport (always True)."""
        return True

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
            help="Use stdio transport (default, only option)",
        )

        parser.add_argument(
            "--show-setup",
            action="store_true",
            help="Display MCP setup instructions and exit",
        )

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        """Load MCP config from environment variables."""
        # Stdio transport only - no environment overrides needed
        return {}

    @classmethod
    def extract_cli_overrides(cls, args: Any) -> dict[str, Any]:
        """Extract MCP config from CLI arguments."""
        # Stdio transport only - no CLI overrides needed
        return {}

    def __repr__(self) -> str:
        """String representation of MCP configuration."""
        return f"MCPConfig(transport={self.transport})"
