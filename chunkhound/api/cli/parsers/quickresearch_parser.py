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

    # Parent hands down its PID so the orphan watchdog compares against an
    # authoritative reference instead of a post-fork getppid() snapshot (which
    # would already be the reparent PID if the parent died during interpreter
    # startup, and would misidentify a Docker-entrypoint parent as PID 1).
    p.add_argument(
        "--parent-pid",
        type=int,
        required=True,
        help=argparse.SUPPRESS,
    )

    add_common_arguments(p)
    add_config_arguments(p, ["embedding", "llm", "research"])

    return cast(argparse.ArgumentParser, p)


__all__: list[str] = ["add_quickresearch_subparser"]
