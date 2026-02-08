"""Main argument parser for ChunkHound CLI."""

import argparse

from chunkhound.version import __version__


def create_main_parser() -> argparse.ArgumentParser:
    """Create and configure the main argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="chunkhound",
        description=(
            "Codebase intelligence for AI coding agents—index, search, and research your code"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  chunkhound index
  chunkhound index /path/to/project
  chunkhound index . --db ./chunks.duckdb
  chunkhound index /code --include "*.py" --exclude "*/tests/*"
  chunkhound mcp --db ./chunks.duckdb
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"chunkhound {__version__}",
    )

    return parser


def setup_subparsers(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    """Set up subparsers for the main parser.

    Args:
        parser: Main argument parser

    Returns:
        Subparsers action for adding command parsers
    """
    return parser.add_subparsers(dest="command", help="Available commands")


__all__ = [
    "create_main_parser",
    "setup_subparsers",
]
