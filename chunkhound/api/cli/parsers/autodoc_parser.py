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
            "pipeline: first identify points of interest (count depends on "
            "--comprehensiveness), then run deep code research for each point and "
            "assemble a unified document."
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

    # Optional flag: stop after overview/points-of-interest phase
    autodoc_parser.add_argument(
        "--overview-only",
        action="store_true",
        help=(
            "Only run the initial overview pass and print the planned points of "
            "interest (count depends on --comprehensiveness), skipping per-point deep "
            "research and final assembly."
        ),
    )

    # Mandatory output directory for per-topic documents and index
    autodoc_parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help=(
            "Directory where an index file and one markdown file per point of "
            "interest will be written, in addition to streaming the combined "
            "document to stdout."
        ),
    )

    # Optional comprehensiveness level controlling HyDE PoI count and snippet budget
    autodoc_parser.add_argument(
        "--comprehensiveness",
        choices=["low", "medium", "high", "ultra"],
        default="medium",
        help=(
            "Control how many HyDE points of interest are generated and how "
            "much code is sampled for the overview: low=5 PoIs and smaller "
            "snippet budget, medium=10, high=15 with a larger snippet budget, "
            "ultra=20 PoIs with the largest snippet budget."
        ),
    )

    return cast(argparse.ArgumentParser, autodoc_parser)


__all__: list[str] = ["add_autodoc_subparser"]
