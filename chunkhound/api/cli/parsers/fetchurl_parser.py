"""Fetchurl command argument parser for ChunkHound CLI."""

import argparse
from typing import Any, cast

from .common_arguments import add_common_arguments, add_config_arguments


def add_fetchurl_subparser(subparsers: Any) -> argparse.ArgumentParser:
    """Add fetchurl command subparser to the main parser."""
    p = subparsers.add_parser(
        "fetchurl",
        help="Fetch a single URL and produce a focused Markdown answer",
        description=(
            "Fetch a single URL (HTML or PDF), extract content, and produce "
            "a Markdown answer. Requires configured LLM and reranker."
        ),
    )
    p.add_argument("url", help="URL to fetch (http:// or https:// only)")
    p.add_argument(
        "--query",
        "-q",
        dest="query",
        default="",
        metavar="TEXT",
        help=(
            "Optional question to focus the extraction on. When set, enables "
            "reranking on long pages to focus the answer."
        ),
    )

    add_common_arguments(p)
    add_config_arguments(p, ["fetchurl", "embedding", "llm"])

    return cast(argparse.ArgumentParser, p)
