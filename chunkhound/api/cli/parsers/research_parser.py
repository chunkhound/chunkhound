"""Research command argument parser for ChunkHound CLI."""

import argparse
from pathlib import Path
from typing import Any, cast

from .common_arguments import (
    add_common_arguments,
    add_config_arguments,
    nonempty_path_filter,
)


def add_research_subparser(subparsers: Any) -> argparse.ArgumentParser:
    """Add research command subparser to the main parser.

    Args:
        subparsers: Subparsers object from the main argument parser

    Returns:
        The configured research subparser
    """
    research_parser = subparsers.add_parser(
        "research",
        help="Perform deep code research",
        description="Answer complex questions about codebase architecture and patterns. Synthesis budgets scale automatically based on repository size.",
    )

    # Required query argument
    research_parser.add_argument(
        "query",
        help="Research question to investigate",
    )

    # Optional positional argument with default to current directory
    research_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Directory path to research (default: current directory)",
    )

    research_parser.add_argument(
        "--path-filter",
        type=nonempty_path_filter,
        help="Optional path filter (e.g., 'src/', 'tests/')",
    )

    # Git commit inputs — mutually exclusive
    diff_group = research_parser.add_mutually_exclusive_group()
    diff_group.add_argument(
        "--commit-range",
        type=str,
        default=None,
        dest="commit_range",
        help="Git revision range to research (e.g. 'HEAD~10..HEAD', 'v1.0..v2.0').",
    )
    diff_group.add_argument(
        "--commit-hash",
        type=str,
        default=None,
        dest="commit_hash",
        help="Single commit hash — researches from that commit to HEAD.",
    )
    diff_group.add_argument(
        "--last-n",
        type=int,
        default=None,
        dest="last_n_commits",
        help="Research the last N commits (equivalent to HEAD~N..HEAD).",
    )
    research_parser.add_argument(
        "--vector-source",
        choices=["diff", "db", "both"],
        default="both",
        dest="vector_source",
        help="Search scope when commit input given: 'both' (default), 'diff', or 'db'.",
    )

    # Add common arguments
    add_common_arguments(research_parser)

    # Add config-specific arguments - database, embedding (for reranking), llm, and research
    add_config_arguments(research_parser, ["database", "embedding", "llm", "research"])

    return cast(argparse.ArgumentParser, research_parser)


__all__: list[str] = ["add_research_subparser"]
