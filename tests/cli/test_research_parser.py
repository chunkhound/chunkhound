from __future__ import annotations

import argparse
from pathlib import Path

from chunkhound.api.cli.parsers.research_parser import add_research_subparser


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_research_subparser(subparsers)
    return parser


def test_research_parser_defaults_and_flags() -> None:
    parser = _build_parser()

    args = parser.parse_args(["research", "why is x slow"])
    assert args.query == "why is x slow"
    assert args.path == Path(".")
    assert args.path_filter is None

    args = parser.parse_args(["research", "why is x slow", "--path-filter", "src/"])
    assert args.path == Path(".")
    assert args.path_filter == "src/"

    args = parser.parse_args(["research", "why is x slow", "repo", "--path-filter", "src/"])
    assert args.path == Path("repo")
    assert args.path_filter == "src/"

