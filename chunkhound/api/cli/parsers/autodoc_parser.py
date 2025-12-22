"""AutoDoc site generation command argument parser."""

import argparse
from pathlib import Path
from typing import Any, cast

from .common_arguments import add_common_arguments, add_config_arguments


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
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("."),
        help="Directory containing AutoDoc outputs (index + topic files).",
    )

    site_parser.add_argument(
        "--out-dir",
        type=Path,
        help=(
            "Output directory for the generated Astro site. "
            "Defaults to <input-dir>/autodoc/ when omitted."
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
        "--index-pattern",
        action="append",
        dest="index_patterns",
        help=(
            "Override index filename glob(s). Can be provided multiple times, "
            "e.g. --index-pattern '*_autodoc_index.md'."
        ),
    )

    add_common_arguments(site_parser)
    add_config_arguments(site_parser, ["llm"])

    return cast(argparse.ArgumentParser, site_parser)


__all__: list[str] = ["add_autodoc_subparser"]
