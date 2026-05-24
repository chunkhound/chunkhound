"""Unit tests for ``build_forwarded_argv``.

These tests are the regression contract for argv reconstruction across
both subprocess forwarding paths (websearch → _quickresearch and the
mcp → _daemon proxy). They lock in the helper's handling of
BooleanOptionalAction, store_true/store_false, list, and scalar actions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chunkhound.api.cli.parsers.common_arguments import build_forwarded_argv


def _parse(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    return parser.parse_args(argv)


def _ssl_parser(default: bool | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--ssl-verify", action=argparse.BooleanOptionalAction, default=default
    )
    return p


def test_boolean_optional_true_emits_positive_flag() -> None:
    parser = _ssl_parser()
    args = _parse(parser, ["--ssl-verify"])
    assert build_forwarded_argv(parser, args) == ["--ssl-verify"]


def test_boolean_optional_false_emits_negative_flag() -> None:
    parser = _ssl_parser()
    args = _parse(parser, ["--no-ssl-verify"])
    assert build_forwarded_argv(parser, args) == ["--no-ssl-verify"]


def test_boolean_optional_none_default_skipped() -> None:
    parser = _ssl_parser()
    args = _parse(parser, [])
    assert build_forwarded_argv(parser, args) == []


def test_boolean_optional_matches_non_none_default_skipped() -> None:
    parser = _ssl_parser(default=True)
    args = _parse(parser, [])  # val == default == True
    assert build_forwarded_argv(parser, args) == []


def test_boolean_optional_flipped_from_non_none_default_emits_negative() -> None:
    parser = _ssl_parser(default=True)
    args = _parse(parser, ["--no-ssl-verify"])
    assert build_forwarded_argv(parser, args) == ["--no-ssl-verify"]


def test_store_true_flipped_emits_flag() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--debug", action="store_true")
    args = _parse(parser, ["--debug"])
    assert build_forwarded_argv(parser, args) == ["--debug"]


def test_store_true_default_skipped() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--debug", action="store_true")
    args = _parse(parser, [])
    assert build_forwarded_argv(parser, args) == []


def test_store_false_flipped_emits_flag() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-color", dest="color", action="store_false")
    args = _parse(parser, ["--no-color"])
    assert build_forwarded_argv(parser, args) == ["--no-color"]


def test_list_action_forwards_each_item() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--include", action="append", default=[])
    args = _parse(parser, ["--include", "a", "--include", "b"])
    assert build_forwarded_argv(parser, args) == [
        "--include",
        "a",
        "--include",
        "b",
    ]


def test_scalar_resolves_path_and_forwards(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", type=Path)
    target = tmp_path / "out.txt"
    args = _parse(parser, ["--out", str(target)])
    forwarded = build_forwarded_argv(parser, args)
    assert forwarded == ["--out", str(target.resolve())]


def test_scalar_equal_default_skipped() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--limit", type=int, default=30)
    args = argparse.Namespace(limit=30)
    assert build_forwarded_argv(parser, args) == []


def test_scalar_string_forwards_value() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--llm-model", default=None)
    args = _parse(parser, ["--llm-model", "gpt-x"])
    assert build_forwarded_argv(parser, args) == ["--llm-model", "gpt-x"]


def test_skip_dests_excludes_action() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    parser.add_argument("--llm-model", default=None)
    args = _parse(parser, ["--config", "x.yaml", "--llm-model", "m"])
    forwarded = build_forwarded_argv(parser, args, skip_dests={"config"})
    assert forwarded == ["--llm-model", "m"]
