"""Search command argument parser for ChunkHound CLI."""

import argparse
from pathlib import Path
from typing import Any, cast

from .common_arguments import add_common_arguments, add_config_arguments


def add_search_subparser(subparsers: Any) -> argparse.ArgumentParser:
    """Add search command subparser to the main parser.

    Args:
        subparsers: Subparsers object from the main argument parser

    Returns:
        The configured search subparser
    """
    search_parser = subparsers.add_parser(
        "search",
        help="Search indexed codebase",
        description="Perform semantic or regex search on indexed code",
    )

    # Required query argument
    search_parser.add_argument(
        "query",
        help="Search query (text for semantic search, pattern for regex search)",
    )

    # Optional positional argument with default to current directory
    search_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Directory path to search (default: current directory)",
    )

    # Search type - mutually exclusive group
    search_type = search_parser.add_mutually_exclusive_group()
    search_type.add_argument(
        "--semantic",
        action="store_true",
        help="Perform semantic search (default)",
    )
    search_type.add_argument(
        "--single-hop",
        action="store_true",
        help="Force single-hop semantic search",
    )
    search_type.add_argument(
        "--multi-hop",
        action="store_true",
        help="Force multi-hop semantic search",
    )
    search_type.add_argument(
        "--regex",
        action="store_true",
        help="Perform regex pattern search",
    )

    # Search parameters (matching MCP server parameters)
    search_parser.add_argument(
        "--page-size",
        type=int,
        default=10,
        help="Number of results per page (default: 10)",
    )
    search_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting offset for pagination (default: 0)",
    )
    search_parser.add_argument(
        "--path-filter",
        type=str,
        help="Optional path filter (e.g., 'src/', 'tests/')",
    )

    # Multi-repo / global mode options
    search_parser.add_argument(
        "--tags",
        type=str,
        help="Filter by project tags (comma-separated, AND logic). "
        "Requires global mode. Example: --tags backend,python",
    )
    search_parser.add_argument(
        "--projects",
        type=str,
        help="Search specific projects (comma-separated names or paths). "
        "Requires global mode. Example: --projects myapp,lib",
    )

    # Add common arguments
    add_common_arguments(search_parser)

    # Add config-specific arguments - only database and embedding needed for search
    add_config_arguments(search_parser, ["database", "embedding"])

    return cast(argparse.ArgumentParser, search_parser)


__all__: list[str] = ["add_search_subparser"]
