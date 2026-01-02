from __future__ import annotations

import argparse

from chunkhound.api.cli.parsers.autodoc_parser import add_autodoc_subparser


def test_autodoc_parser_allows_omitting_map_in() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_autodoc_subparser(subparsers)

    args = parser.parse_args(["autodoc", "--out-dir", "site"])

    assert args.command == "autodoc"
    assert args.map_in is None


def test_autodoc_parser_supports_map_comprehensiveness_aliases() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_autodoc_subparser(subparsers)

    args = parser.parse_args(["autodoc", "--out-dir", "site", "--map-low"])

    assert args.command == "autodoc"
    assert args.map_comprehensiveness == "low"
