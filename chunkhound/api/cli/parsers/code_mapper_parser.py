"""Map command argument parser for ChunkHound CLI."""

import argparse
from pathlib import Path
from typing import Any, cast

from .common_arguments import (
    _parse_audience,
    add_common_arguments,
    add_config_arguments,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def add_map_subparser(subparsers: Any) -> argparse.ArgumentParser:
    """Add map command subparser to the main parser.

    Args:
        subparsers: Subparsers object from the main argument parser

    Returns:
        The configured map subparser
    """
    map_parser = subparsers.add_parser(
        "map",
        help="Generate agent-facing docs for a folder",
        description=(
            "Generate agent-facing documentation for a folder using a two-phase "
            "pipeline: plan points of interest, then run deep research per point "
            "and write topic artifacts plus an index."
        ),
    )

    # Optional positional argument with default to current directory
    map_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("."),
        help=(
            "Directory path to document (scope, default: current directory). "
            "Paths are resolved relative to the project root used for indexing."
        ),
    )

    # Add common arguments
    add_common_arguments(map_parser)

    # Code Mapper requires database, embedding (for reranking), and llm configuration
    add_config_arguments(map_parser, ["database", "embedding", "llm"])

    # Optional flag: stop after overview/points-of-interest phase
    map_parser.add_argument(
        "--plan",
        "--overview-only",
        dest="overview_only",
        action="store_true",
        help=(
            "Only run the planning pass and print the planned points of interest, "
            "skipping per-point deep research and final assembly."
        ),
    )

    map_parser.add_argument(
        "--audience",
        type=_parse_audience,
        default="balanced",
        help=(
            "Controls the intended audience for generated map topics. "
            "Accepted: 1|technical, 2|balanced, 3|end-user."
        ),
    )

    map_parser.add_argument(
        "--context",
        type=Path,
        default=None,
        help=(
            "Path to a markdown/text file used as authoritative context for HyDE "
            "planning. When set, this fully replaces repo-derived HyDE context "
            "(file lists and sampled code snippets) for both architectural and "
            "operational maps."
        ),
    )

    # Mandatory output directory for per-topic documents and index
    map_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        dest="out",
        help=(
            "Directory where an index file and one markdown file per point of "
            "interest will be written. Use --combined to also write a combined "
            "Code Mapper markdown file (or set CH_CODE_MAPPER_WRITE_COMBINED=1 for "
            "backward compatibility)."
        ),
    )

    # Optional: write a single combined markdown document (CLI overrides env when set)
    try:
        boolean_optional_action = argparse.BooleanOptionalAction
    except AttributeError:  # pragma: no cover - older Python
        boolean_optional_action = None

    if boolean_optional_action is not None:
        map_parser.add_argument(
            "--combined",
            action=boolean_optional_action,
            default=None,
            help=(
                "Write a combined Code Mapper markdown file. If omitted, falls back "
                "to CH_CODE_MAPPER_WRITE_COMBINED for backward compatibility."
            ),
        )
    else:
        map_parser.add_argument(
            "--combined",
            action="store_true",
            default=None,
            help=(
                "Write a combined Code Mapper markdown file. If omitted, falls back "
                "to CH_CODE_MAPPER_WRITE_COMBINED for backward compatibility."
            ),
        )

    map_parser.add_argument(
        "-j",
        "--jobs",
        type=_positive_int,
        default=None,
        help=(
            "Max concurrent point-of-interest deep research jobs. Must be >= 1. "
            "If omitted, falls back to CH_CODE_MAPPER_POI_CONCURRENCY or a default."
        ),
    )

    map_parser.set_defaults(comprehensiveness="medium")
    level_group = map_parser.add_mutually_exclusive_group()
    level_group.add_argument(
        "--minimal",
        action="store_const",
        const="minimal",
        dest="comprehensiveness",
        help="Alias for: --comprehensiveness minimal",
    )
    level_group.add_argument(
        "--low",
        action="store_const",
        const="low",
        dest="comprehensiveness",
        help="Alias for: --comprehensiveness low",
    )
    level_group.add_argument(
        "--medium",
        action="store_const",
        const="medium",
        dest="comprehensiveness",
        help="Alias for: --comprehensiveness medium",
    )
    level_group.add_argument(
        "--high",
        action="store_const",
        const="high",
        dest="comprehensiveness",
        help="Alias for: --comprehensiveness high",
    )
    level_group.add_argument(
        "--ultra",
        action="store_const",
        const="ultra",
        dest="comprehensiveness",
        help="Alias for: --comprehensiveness ultra",
    )
    level_group.add_argument(
        "--comprehensiveness",
        choices=["minimal", "low", "medium", "high", "ultra"],
        help=(
            "Control how many points of interest are generated and how much code is "
            "sampled for planning: minimal=1, low=5, medium=10, high=15, ultra=20. "
            "(HyDE file list cap scales: minimal=200, low=500, medium=2000, high=3000, "
            "ultra=5000.)"
        ),
    )

    return cast(argparse.ArgumentParser, map_parser)


__all__: list[str] = ["add_map_subparser"]
