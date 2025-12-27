import argparse
from pathlib import Path

import pytest

from chunkhound.api.cli.parsers.code_mapper_parser import add_code_mapper_subparser


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_code_mapper_subparser(subparsers)
    return parser


def test_code_mapper_parser_requires_out_dir() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["code_mapper"])


def test_code_mapper_parser_defaults_and_flags() -> None:
    parser = _build_parser()

    args = parser.parse_args(["code_mapper", "--out-dir", "out"])

    assert args.comprehensiveness == "medium"
    assert args.path == Path(".")
    assert args.overview_only is False
    assert args.combined is None

    args = parser.parse_args(
        ["code_mapper", "src", "--out-dir", "out", "--overview-only", "--verbose"]
    )

    assert args.path == Path("src")
    assert args.overview_only is True
    assert args.verbose is True


def test_code_mapper_parser_combined_flag() -> None:
    parser = _build_parser()

    args = parser.parse_args(["code_mapper", "--out-dir", "out", "--combined"])
    assert args.combined is True

    if hasattr(argparse, "BooleanOptionalAction"):
        args = parser.parse_args(["code_mapper", "--out-dir", "out", "--no-combined"])
        assert args.combined is False
