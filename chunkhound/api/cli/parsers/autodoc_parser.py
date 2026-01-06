"""AutoDoc site generation command argument parser."""

import argparse
from pathlib import Path
from typing import Any, cast

from .common_arguments import (
    _parse_audience,
    add_common_arguments,
    add_config_arguments,
)


def add_autodoc_subparser(subparsers: Any) -> argparse.ArgumentParser:
    """Add AutoDoc site generator subparser to the main parser."""
    site_parser = subparsers.add_parser(
        "autodoc",
        help="Generate an AutoDoc Astro site from AutoDoc outputs",
        description=(
            "Transform an existing AutoDoc output folder into a polished "
            "Astro documentation site with a final technical-writer cleanup pass."
        ),
    )

    site_parser.add_argument(
        "map_in",
        metavar="map-in",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Directory containing Code Mapper outputs (index + topic files). "
            "If omitted, AutoDoc can prompt to generate maps first."
        ),
    )

    site_parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help=("Output directory for the generated Astro site."),
    )

    site_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help=(
            "Allow AutoDoc to delete and re-generate `src/pages/topics` inside "
            "--out-dir when it already exists (non-interactive runs require this)."
        ),
    )

    site_parser.add_argument(
        "--assets-only",
        action="store_true",
        help=(
            "Update only the generated Astro assets (layout/styles/config) in "
            "--out-dir without rewriting topic pages. Intended for iterating on "
            "UI changes when content does not change."
        ),
    )

    site_parser.add_argument(
        "--site-title",
        type=str,
        help="Override the generated site title.",
    )

    site_parser.add_argument(
        "--site-tagline",
        type=str,
        help="Override the generated site tagline.",
    )

    site_parser.add_argument(
        "--cleanup-mode",
        choices=["llm"],
        default="llm",
        help="Run the technical-writer cleanup pass via LLM (required).",
    )

    site_parser.add_argument(
        "--cleanup-batch-size",
        type=int,
        default=4,
        help="Number of topic sections to send per LLM cleanup batch.",
    )

    site_parser.add_argument(
        "--cleanup-max-tokens",
        type=int,
        default=4096,
        help="Maximum completion tokens per cleanup response.",
    )

    site_parser.add_argument(
        "--audience",
        type=_parse_audience,
        default="balanced",
        help=(
            "Controls the intended audience for the generated docs (LLM cleanup only). "
            "Accepted: 1|technical, 2|balanced, 3|end-user."
        ),
    )

    site_parser.add_argument(
        "--index-pattern",
        action="append",
        dest="index_patterns",
        help=(
            "Override index filename glob(s). Can be provided multiple times, "
            "e.g. --index-pattern '*_autodoc_index.md'."
        ),
    )

    site_parser.add_argument(
        "--map-out-dir",
        type=Path,
        help=(
            "When AutoDoc offers to run Code Mapper automatically (because the "
            "provided `map-in` directory does not contain an index), write the "
            "generated map outputs to this directory. If omitted, AutoDoc will prompt "
            "(TTY only)."
        ),
    )

    map_level_group = site_parser.add_mutually_exclusive_group()
    map_level_group.add_argument(
        "--map-minimal",
        action="store_const",
        const="minimal",
        dest="map_comprehensiveness",
        help="Alias for: --map-comprehensiveness minimal",
    )
    map_level_group.add_argument(
        "--map-low",
        action="store_const",
        const="low",
        dest="map_comprehensiveness",
        help="Alias for: --map-comprehensiveness low",
    )
    map_level_group.add_argument(
        "--map-medium",
        action="store_const",
        const="medium",
        dest="map_comprehensiveness",
        help="Alias for: --map-comprehensiveness medium",
    )
    map_level_group.add_argument(
        "--map-high",
        action="store_const",
        const="high",
        dest="map_comprehensiveness",
        help="Alias for: --map-comprehensiveness high",
    )
    map_level_group.add_argument(
        "--map-ultra",
        action="store_const",
        const="ultra",
        dest="map_comprehensiveness",
        help="Alias for: --map-comprehensiveness ultra",
    )
    map_level_group.add_argument(
        "--map-comprehensiveness",
        choices=["minimal", "low", "medium", "high", "ultra"],
        dest="map_comprehensiveness",
        help=(
            "When AutoDoc offers to run Code Mapper automatically, controls the "
            "mapping depth. If omitted, AutoDoc will prompt (TTY only)."
        ),
    )

    site_parser.add_argument(
        "--map-audience",
        type=_parse_audience,
        default=None,
        help=(
            "When AutoDoc offers to run Code Mapper automatically, controls the "
            "intended audience for generated map topics. If omitted, AutoDoc will "
            "prompt (TTY only). Accepted: 1|2|3 or technical|balanced|end-user."
        ),
    )

    site_parser.add_argument(
        "--map-context",
        type=Path,
        default=None,
        help=(
            "When AutoDoc offers to run Code Mapper automatically, pass a context "
            "file through to Code Mapper planning as `chunkhound map --context ...`. "
            "If omitted, AutoDoc will prompt (TTY only)."
        ),
    )

    add_common_arguments(site_parser)
    add_config_arguments(site_parser, ["llm"])

    return cast(argparse.ArgumentParser, site_parser)


__all__: list[str] = ["add_autodoc_subparser"]
