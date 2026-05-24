"""Argparse-level tests for the websearch subcommand parser."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from chunkhound.api.cli.parsers.websearch_parser import add_websearch_subparser


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_websearch_subparser(parser.add_subparsers(dest="command"))
    return parser


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    flags: set[str] = set()
    for action in parser._actions:  # noqa: SLF001 — test-only introspection
        flags.update(action.option_strings)
    return flags


def _websearch_subparser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command")
    return add_websearch_subparser(sub)


def test_websearch_parser_positional_query_required() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["websearch"])


def test_websearch_parser_limit_default_30() -> None:
    args = _build_parser().parse_args(["websearch", "hello world"])
    assert args.query == "hello world"
    assert args.limit == 30


def test_websearch_parser_limit_override() -> None:
    args = _build_parser().parse_args(["websearch", "q", "--limit", "7"])
    assert args.limit == 7


def test_websearch_parser_limit_non_int_rejected() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["websearch", "q", "--limit", "xyz"])


def test_websearch_parser_common_flags_registered(tmp_path: Path) -> None:
    cfg = tmp_path / "chunkhound.json"
    cfg.write_text("{}")
    args = _build_parser().parse_args(
        ["websearch", "q", "--verbose", "--debug", "--config", str(cfg)]
    )
    assert args.verbose is True
    assert args.debug is True
    assert args.config == cfg


def test_websearch_parser_config_sections_wired() -> None:
    flags = _option_strings(_websearch_subparser())
    # embedding section
    assert "--embedding-provider" in flags
    # llm section
    assert "--llm-provider" in flags
    # research section
    assert "--research-algorithm" in flags
