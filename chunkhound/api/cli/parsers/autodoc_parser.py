"""AutoDoc site generation command argument parser."""

import argparse
from pathlib import Path
from typing import Any, cast

from .common_arguments import add_common_arguments, add_config_arguments


def _parse_taint(value: str) -> str:
    normalized = value.strip().lower()
    mapping = {
        "1": "technical",
        "technical": "technical",
        "2": "balanced",
        "balanced": "balanced",
        "3": "end-user",
        "end-user": "end-user",
        "end_user": "end-user",
        "enduser": "end-user",
    }
    resolved = mapping.get(normalized)
    if resolved is None:
        raise argparse.ArgumentTypeError(
            "Invalid --taint value. Use 1|2|3 or technical|balanced|end-user."
        )
    return resolved


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
        choices=["llm", "minimal"],
        default="llm",
        help="Choose whether to run the technical-writer cleanup pass via LLM.",
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
        "--taint",
        type=_parse_taint,
        default="balanced",
        help=(
            "Controls how technical the generated docs are (LLM cleanup only). "
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

    site_parser.add_argument(
        "--map-comprehensiveness",
        choices=["minimal", "low", "medium", "high", "ultra"],
        help=(
            "When AutoDoc offers to run Code Mapper automatically, controls the "
            "mapping depth. If omitted, AutoDoc will prompt (TTY only)."
        ),
    )

    add_common_arguments(site_parser)
    add_config_arguments(site_parser, ["llm"])

    return cast(argparse.ArgumentParser, site_parser)


__all__: list[str] = ["add_autodoc_subparser"]
