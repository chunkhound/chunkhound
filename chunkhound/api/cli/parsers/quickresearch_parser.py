"""Argument parser for the hidden ``_quickresearch`` CLI subcommand."""

import argparse
from pathlib import Path
from typing import Any, cast

from .common_arguments import (
    add_common_arguments,
    add_config_arguments,
    nonempty_path_filter,
)


def add_quickresearch_subparser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the internal ``_quickresearch`` subcommand (hidden from help)."""
    p = subparsers.add_parser(
        "_quickresearch",
        help=argparse.SUPPRESS,
        description=(
            "Internal: index a directory in memory, then perform deep code "
            "research. No index is persisted."
        ),
    )

    p.add_argument("query", help="Research question to investigate")

    p.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Directory to index and research (default: current directory)",
    )

    p.add_argument(
        "--path-filter",
        type=nonempty_path_filter,
        help="Optional path filter (e.g., 'src/', 'tests/')",
    )

    add_common_arguments(p)
    add_config_arguments(p, ["embedding", "llm", "research"])

    return cast(argparse.ArgumentParser, p)


__all__: list[str] = ["add_quickresearch_subparser"]
