"""Autodoc command argument parser for ChunkHound CLI."""

import argparse
from pathlib import Path
from typing import Any, cast

from .common_arguments import add_common_arguments, add_config_arguments


def add_autodoc_subparser(subparsers: Any) -> argparse.ArgumentParser:
    """Add autodoc command subparser to the main parser.

    Args:
        subparsers: Subparsers object from the main argument parser

    Returns:
        The configured autodoc subparser
    """
    autodoc_parser = subparsers.add_parser(
        "autodoc",
        help="Generate auto documentation for a scoped folder",
        description=(
            "Generate agent-facing documentation for a scoped folder using a two-phase "
            "pipeline: first identify 5-10 points of interest, then run deep "
            "code research for each point and assemble a unified document."
        ),
    )

    # Optional positional argument with default to current directory
    autodoc_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("."),
        help=(
            "Directory path to document (acts as scope, default: current directory). "
            "Paths are resolved relative to the project root used for indexing."
        ),
    )

    # Add common arguments
    add_common_arguments(autodoc_parser)

    # Autodoc requires database, embedding (for reranking), and llm configuration
    add_config_arguments(autodoc_parser, ["database", "embedding", "llm"])

    return cast(argparse.ArgumentParser, autodoc_parser)


__all__: list[str] = ["add_autodoc_subparser"]

