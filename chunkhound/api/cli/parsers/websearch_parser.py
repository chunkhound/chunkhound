"""Websearch command argument parser for ChunkHound CLI."""

import argparse
from typing import Any, cast

from chunkhound.utils.websearch_core import WEBSEARCH_LIMIT_MAX

from .common_arguments import add_common_arguments, add_config_arguments


def _limit_type(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer, got {raw!r}") from None
    if not 1 <= value <= WEBSEARCH_LIMIT_MAX:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {WEBSEARCH_LIMIT_MAX}, got {value}"
        )
    return value


def add_websearch_subparser(subparsers: Any) -> argparse.ArgumentParser:
    """Add websearch command subparser to the main parser."""
    p = subparsers.add_parser(
        "websearch",
        help="Search the web via DuckDuckGo",
        description="Search DuckDuckGo and print results.",
    )

    p.add_argument("query", help="Search query")
    p.add_argument(
        "--limit",
        type=_limit_type,
        default=30,
        metavar="N",
        help=f"Max results to return (1-{WEBSEARCH_LIMIT_MAX}, default: 30)",
    )

    add_common_arguments(p)
    add_config_arguments(p, ["embedding", "llm", "research"])

    return cast(argparse.ArgumentParser, p)


__all__: list[str] = ["add_websearch_subparser"]
