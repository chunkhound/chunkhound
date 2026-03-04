"""Argument parser for the hidden ``_daemon`` CLI subcommand."""

import argparse
from typing import Any

from .common_arguments import add_common_arguments, add_config_arguments


def add_daemon_subparser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the internal ``_daemon`` subcommand (hidden from help).

    Args:
        subparsers: Subparsers object from the main argument parser.

    Returns:
        The configured daemon subparser.
    """
    parser = subparsers.add_parser(
        "_daemon",
        help=argparse.SUPPRESS,
        description="Internal: run ChunkHound as a multi-client daemon process",
    )

    parser.add_argument(
        "--project-dir",
        required=True,
        help="Absolute path of the project directory being indexed",
    )
    parser.add_argument(
        "--socket-path",
        required=True,
        help="Unix socket (POSIX) or TCP loopback address (Windows) for the daemon",
    )

    add_common_arguments(parser)
    add_config_arguments(parser, ["database", "embedding", "indexing", "llm", "mcp"])

    return parser


__all__: list[str] = ["add_daemon_subparser"]
