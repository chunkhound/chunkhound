"""MCP command argument parser for ChunkHound CLI."""

import argparse
from pathlib import Path
from typing import Any, cast

from .common_arguments import add_common_arguments, add_config_arguments


def add_mcp_subparser(subparsers: Any) -> argparse.ArgumentParser:
    """Add MCP command subparser to the main parser.

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

    # Optional positional argument with default to current directory
    mcp_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Directory path to index (default: current directory)",
    )

    # Daemon mode control
    mcp_parser.add_argument(
        "--no-daemon",
        action="store_true",
        help="Run MCP server directly without daemon (single client mode)",
    )

    mcp_parser.add_argument(
        "--read-only",
        action="store_true",
        help=(
            "Open the database read-only; disables indexing/watcher and "
            "runs without the daemon (DuckDB only). See database.read_only "
            "for env/JSON equivalents."
        ),
    )

    # HTTP transport mode
    mcp_parser.add_argument(
        "--transport",
        choices=["http"],
        default=None,
        help="Transport mode: 'http' starts a Streamable HTTP MCP server",
    )
    mcp_parser.add_argument(
        "--port",
        type=int,
        default=5173,
        help="Port for HTTP transport (default: 5173)",
    )
    mcp_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for HTTP transport (default: 0.0.0.0)",
    )

    # Add common arguments
    add_common_arguments(mcp_parser)

    # Add config-specific arguments - include LLM so MCP can override providers
    add_config_arguments(
        mcp_parser, ["database", "embedding", "indexing", "llm", "mcp"]
    )

    return cast(argparse.ArgumentParser, mcp_parser)


__all__: list[str] = ["add_mcp_subparser"]
