"""Websearch command argument parser for ChunkHound CLI."""

import argparse
from typing import Any, cast

from .common_arguments import add_common_arguments, add_config_arguments


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
        type=int,
        default=30,
        metavar="N",
        help="Max results to return (default: 30)",
    )

    add_common_arguments(p)
    add_config_arguments(p, ["embedding", "llm", "research"])

    return cast(argparse.ArgumentParser, p)


__all__: list[str] = ["add_websearch_subparser"]
