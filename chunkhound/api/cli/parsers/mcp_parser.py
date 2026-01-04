"""MCP command argument parser for ChunkHound CLI."""

import argparse
from pathlib import Path
from typing import Any, cast

from .common_arguments import add_common_arguments, add_config_arguments


def add_mcp_subparser(subparsers: Any) -> argparse.ArgumentParser:
    """Add MCP command subparser to the main parser.

    Supports two modes:
    - stdio (default): Standard MCP stdio protocol for single-project use
    - http: HTTP/SSE server for multi-project global database mode

    Args:
        subparsers: Subparsers object from the main argument parser

    Returns:
        The configured MCP subparser
    """
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Run Model Context Protocol server",
        description="Start the MCP server for integration with MCP-compatible clients",
    )

    # Create subparsers for transport modes
    mcp_subparsers = mcp_parser.add_subparsers(
        dest="mcp_transport",
        help="Transport mode (default: stdio for backwards compatibility)",
    )

    # HTTP mode subcommand
    http_parser = mcp_subparsers.add_parser(
        "http",
        help="Run HTTP/SSE server for multi-project mode",
        description=(
            "Start the MCP server with HTTP/SSE transport for global database mode. "
            "Enables multiple concurrent client connections and cross-project search."
        ),
    )
    http_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    http_parser.add_argument(
        "--port",
        type=int,
        default=5173,
        help="Port to listen on (default: 5173)",
    )
    http_parser.add_argument(
        "--query-only",
        action="store_true",
        help="Query-only mode - skip indexing, just search existing database (useful for global mode)",
    )
    http_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Initial project path (default: current directory)",
    )
    add_common_arguments(http_parser)
    add_config_arguments(
        http_parser, ["database", "embedding", "indexing", "llm", "mcp"]
    )

    # Stdio mode (can be explicit or default)
    stdio_parser = mcp_subparsers.add_parser(
        "stdio",
        help="Run stdio server (default mode)",
        description="Start the MCP server with stdio transport for single-project use.",
    )
    stdio_parser.add_argument(
        "--query-only",
        action="store_true",
        help="Query-only mode - skip indexing, just search existing database (useful for global mode)",
    )
    stdio_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Directory path to index (default: current directory)",
    )
    add_common_arguments(stdio_parser)
    add_config_arguments(
        stdio_parser, ["database", "embedding", "indexing", "llm", "mcp"]
    )

    # Default arguments for when no subcommand is specified (backwards compatibility)
    mcp_parser.add_argument(
        "--query-only",
        action="store_true",
        help="Query-only mode - skip indexing, just search existing database (useful for global mode)",
    )
    mcp_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Directory path to index (default: current directory)",
    )

    # Add common arguments to main parser for backwards compatibility
    add_common_arguments(mcp_parser)
    add_config_arguments(
        mcp_parser, ["database", "embedding", "indexing", "llm", "mcp"]
    )

    return cast(argparse.ArgumentParser, mcp_parser)


__all__: list[str] = ["add_mcp_subparser"]
