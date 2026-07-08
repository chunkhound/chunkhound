"""Tests for the RapidYAML-backed parser."""

from __future__ import annotations

import os

import pytest

from chunkhound.core.types.common import FileId, Language
from chunkhound.parsers.parser_factory import ParserFactory
from chunkhound.parsers.rapid_yaml_parser import RapidYamlParser
from chunkhound.parsers.universal_parser import CASTConfig, UniversalParser


def _build_fallback_parser() -> UniversalParser:
    factory = ParserFactory(CASTConfig())
    # Force a plain universal parser by temporarily disabling the wrapper
    os.environ["CHUNKHOUND_YAML_ENGINE"] = "tree"
    parser = factory.create_parser(Language.YAML)
    os.environ.pop("CHUNKHOUND_YAML_ENGINE", None)
    if isinstance(parser, RapidYamlParser):
        return parser._fallback  # type: ignore[attr-defined]
    assert isinstance(parser, UniversalParser)
    return parser


@pytest.mark.skipif(
    os.environ.get("CHUNKHOUND_YAML_ENGINE", "").lower() == "tree",
    reason="RapidYAML disabled via environment",
)
def test_rapid_yaml_parser_emits_chunks():
    try:
        import ryml  # type: ignore # noqa: F401
    except Exception:  # pragma: no cover - optional dependency
        pytest.skip("ryml module not available in test environment")

    fallback = _build_fallback_parser()
    parser = RapidYamlParser(fallback)

    content = (
        "services:\n"
        "  web:\n"
        "    image: nginx:1.25\n"
        "    env:\n"
        "      - PORT=8080\n"
        "  worker:\n"
        "    image: alpine:3.18\n"
    )

    chunks = parser.parse_content(content, None, FileId(99))
    assert chunks, "expected chunks from RapidYAML parser"
    assert all(chunk.language == Language.YAML for chunk in chunks)
    assert any("services.web" in chunk.symbol for chunk in chunks)


@pytest.mark.skipif(
    os.environ.get("CHUNKHOUND_YAML_ENGINE", "").lower() == "tree",
    reason="RapidYAML disabled via environment",
)
def test_rapid_yaml_parser_handles_helm_templates():
    try:
        import ryml  # type: ignore # noqa: F401
    except Exception:  # pragma: no cover - optional dependency
        pytest.skip("ryml module not available in test environment")

    fallback = _build_fallback_parser()
    parser = RapidYamlParser(fallback)

    templated = (
        "{{- if .Values.enabled }}\n"
        "metadata:\n"
        '  labels: {{- include "demo.labels" . | nindent 4 }}\n'
        "  annotations:\n"
        "    note: static\n"
        "{{- end }}\n"
    )

    chunks = parser.parse_content(templated, None, FileId(5))
    assert chunks, "sanitized Helm template should still produce chunks"
    assert any(chunk.symbol.startswith("metadata") for chunk in chunks)


def test_env_flag_disables_rapid_yaml(monkeypatch):
    monkeypatch.setenv("CHUNKHOUND_YAML_ENGINE", "tree")
    fallback = _build_fallback_parser()
    parser = RapidYamlParser(fallback)

    sample = "root:\n  child: value\n"
    chunks = parser.parse_content(sample, None, FileId(1))
    baseline = fallback.parse_content(sample, None, FileId(1))

    assert len(chunks) == len(baseline)


@pytest.mark.skipif(
    os.environ.get("CHUNKHOUND_YAML_ENGINE", "").lower() == "tree",
    reason="RapidYAML disabled via environment",
)
def test_rapid_yaml_parser_denylist_memo(tmp_path):
    try:
        import ryml  # noqa: F401
    except Exception:
        pytest.skip("ryml module not available")

    # Content that triggers complex-key detection ('containers as keys')
    bad = "? [a, b]: c\n"

    fallback = _build_fallback_parser()
    parser = RapidYamlParser(fallback)

    p = tmp_path / "bad.yaml"
    p.write_text(bad)

    # First parse: forces ryml failure; no tree-sitter fallback per policy
    chunks1 = parser.parse_content(bad, p, FileId(7))
    assert chunks1 == []

    # Second parse: path should be denylisted → no ryml attempt, same fallback result
    chunks2 = parser.parse_content(bad, p, FileId(7))
    assert chunks2 == []


@pytest.mark.skipif(
    os.environ.get("CHUNKHOUND_YAML_ENGINE", "").lower() == "tree",
    reason="RapidYAML disabled via environment",
)
def test_summary_logged_on_cleanup(caplog):
    try:
        import ryml  # noqa: F401
    except Exception:
        pytest.skip("ryml module not available")

    fallback = _build_fallback_parser()
    parser = RapidYamlParser(fallback)
    caplog.clear()
    caplog.set_level("INFO")
    # Trigger a small sanitized parse
    text = 'metadata:\n  labels: {{- include "chart.labels" . | nindent 4 }}\n'
    parser.parse_content(text, None, FileId(9))
    parser.cleanup()
    assert any("RapidYAML summary:" in rec.message for rec in caplog.records)
