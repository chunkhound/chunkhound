import argparse
from pathlib import Path

import pytest

from chunkhound.api.cli.parsers.code_mapper_parser import add_map_subparser


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_map_subparser(subparsers)
    return parser


def test_map_parser_requires_out() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["map"])


def test_map_parser_defaults_and_flags() -> None:
    parser = _build_parser()

    args = parser.parse_args(["map", "--out", "out"])

    assert args.comprehensiveness == "medium"
    assert args.path == Path(".")
    assert args.overview_only is False
    assert args.combined is None
    assert args.audience == "balanced"

    args = parser.parse_args(
        ["map", "src", "--out", "out", "--plan", "--verbose"]
    )

    assert args.path == Path("src")
    assert args.overview_only is True
    assert args.verbose is True


def test_map_parser_accepts_taint_values() -> None:
    parser = _build_parser()
    base = ["map", "--out", "out", "--audience"]
    assert parser.parse_args([*base, "1"]).audience == "technical"
    assert parser.parse_args([*base, "technical"]).audience == "technical"
    assert parser.parse_args([*base, "2"]).audience == "balanced"
    assert parser.parse_args([*base, "balanced"]).audience == "balanced"
    assert parser.parse_args([*base, "3"]).audience == "end-user"
    assert parser.parse_args([*base, "end-user"]).audience == "end-user"


def test_map_parser_combined_flag() -> None:
    parser = _build_parser()

    args = parser.parse_args(["map", "--out", "out", "--combined"])
    assert args.combined is True

    if hasattr(argparse, "BooleanOptionalAction"):
        args = parser.parse_args(["map", "--out", "out", "--no-combined"])
        assert args.combined is False


def test_map_parser_level_shortcuts() -> None:
    parser = _build_parser()
    args = parser.parse_args(["map", "--ultra", "--out", "out"])
    assert args.comprehensiveness == "ultra"


def test_map_parser_jobs_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(["map", "--out", "out", "--jobs", "3"])
    assert args.jobs == 3


def test_map_parser_jobs_rejects_zero() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["map", "--out", "out", "--jobs", "0"])
