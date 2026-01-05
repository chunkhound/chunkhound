"""Regression tests to ensure RapidYAML preserves YAML structure semantics."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chunkhound.core.types.common import FileId, Language
from chunkhound.parsers.parser_factory import ParserFactory
from chunkhound.parsers.rapid_yaml_parser import RapidYamlParser
from chunkhound.parsers.universal_parser import CASTConfig, UniversalParser

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "yaml"
EXPECTED_SYMBOLS: dict[str, set[str]] = {
    "basic_services.yaml": {
        "services",
        "services.web",
        "services.web.image",
        "services.web.env",
        "services.worker",
        "services.worker.image",
        "services.worker.command",
    },
    "sequences.yaml": {
        "pipelines",
        "pipelines.stage",
        "pipelines.stage.steps",
        "pipelines.stage.environment",
    },
    "multi_documents.yaml": {
        "kind.metadata",
        "kind.metadata.name",
        "kind.data.DEBUG",
        "kind.spec.template.spec.containers.name.image",
    },
    "anchors.yaml": {
        "defaults.retries",
        "defaults.timeout",
        "services.api.endpoint",
        "services.worker.queue",
    },
}


def _ensure_rapid_available() -> None:
    try:
        import ryml  # type: ignore  # noqa: F401
    except Exception:  # pragma: no cover - optional dependency
        pytest.skip("ryml module not available – RapidYAML parser disabled.")


def _build_fallback_parser() -> UniversalParser:
    os.environ["CHUNKHOUND_YAML_ENGINE"] = "tree"
    factory = ParserFactory(CASTConfig())
    parser = factory.create_parser(Language.YAML)
    os.environ.pop("CHUNKHOUND_YAML_ENGINE", None)
    if isinstance(parser, RapidYamlParser):
        return parser._fallback  # type: ignore[attr-defined]
    assert isinstance(parser, UniversalParser)
    return parser


def _build_rapid_parser() -> RapidYamlParser:
    fallback = _build_fallback_parser()
    return RapidYamlParser(fallback)


@pytest.mark.parametrize("fixture_name", sorted(EXPECTED_SYMBOLS.keys()))
def test_rapid_yaml_parser_covers_expected_symbols(fixture_name: str):
    _ensure_rapid_available()
    parser = _build_rapid_parser()
    fixture_path = FIXTURE_DIR / fixture_name
    content = fixture_path.read_text(encoding="utf-8")

    chunks = parser.parse_content(content, fixture_path, FileId(1))
    observed = {chunk.symbol for chunk in chunks}

    missing = EXPECTED_SYMBOLS[fixture_name] - observed
    assert not missing, f"{fixture_name} missing symbols: {sorted(missing)}"


@pytest.mark.parametrize("fixture_name", sorted(EXPECTED_SYMBOLS.keys()))
def test_rapid_yaml_chunk_line_alignment(fixture_name: str):
    _ensure_rapid_available()
    parser = _build_rapid_parser()
    fixture_path = FIXTURE_DIR / fixture_name
    content = fixture_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    total_lines = len(lines)

    chunks = parser.parse_content(content, fixture_path, FileId(2))
    for chunk in chunks:
        start_line = int(chunk.start_line)
        end_line = int(chunk.end_line)
        assert start_line >= 1, f"{fixture_name}: start line {start_line} invalid"
        assert end_line >= start_line, (
            f"{fixture_name}: end {end_line} < start {start_line}"
        )
        assert end_line <= total_lines, (
            f"{fixture_name}: end line {end_line} beyond file length {total_lines}"
        )

        slice_start = start_line - 1
        slice_end = end_line
        expected = "\n".join(lines[slice_start:slice_end]).strip()
        actual = chunk.code.strip()
        assert actual == expected, (
            f"{fixture_name}: chunk {chunk.symbol} ({start_line}-{end_line}) content mismatch"
        )


def test_tree_sitter_fallback_still_parses():
    """Ensure forcing tree-sitter still returns chunks (regression guard)."""
    os.environ["CHUNKHOUND_YAML_ENGINE"] = "tree"
    factory = ParserFactory(CASTConfig())
    parser = factory.create_parser(Language.YAML)
    os.environ.pop("CHUNKHOUND_YAML_ENGINE", None)

    sample = (FIXTURE_DIR / "basic_services.yaml").read_text(encoding="utf-8")
    chunks = parser.parse_content(sample, None, FileId(42))
    assert chunks, "tree-sitter fallback should still produce chunks"
