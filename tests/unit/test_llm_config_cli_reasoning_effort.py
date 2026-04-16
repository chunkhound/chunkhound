"""CLI argument parsing for Codex reasoning effort."""

import argparse

import pytest

from chunkhound.core.config.llm_config import LLMConfig


def test_cli_accepts_xhigh_reasoning_effort() -> None:
    parser = argparse.ArgumentParser()
    LLMConfig.add_cli_arguments(parser)

    args = parser.parse_args(
        [
            "--llm-codex-reasoning-effort",
            "xhigh",
            "--llm-codex-reasoning-effort-utility",
            "xhigh",
            "--llm-codex-reasoning-effort-synthesis",
            "xhigh",
        ]
    )

    assert args.llm_codex_reasoning_effort == "xhigh"
    assert args.llm_codex_reasoning_effort_utility == "xhigh"
    assert args.llm_codex_reasoning_effort_synthesis == "xhigh"


def test_cli_provider_ollama_shows_migration_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = argparse.ArgumentParser()
    LLMConfig.add_cli_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--llm-provider", "ollama"])

    err = capsys.readouterr().err
    assert "provider='openai'" in err
    assert "http://localhost:11434/v1" in err


def test_cli_per_role_provider_ollama_shows_migration_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = argparse.ArgumentParser()
    LLMConfig.add_cli_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--llm-utility-provider", "ollama"])

    err = capsys.readouterr().err
    assert "provider='openai'" in err
    assert "http://localhost:11434/v1" in err


def test_cli_parses_llm_ssl_verify_flags() -> None:
    parser = argparse.ArgumentParser()
    LLMConfig.add_cli_arguments(parser)

    args = parser.parse_args(["--no-llm-ssl-verify"])

    assert args.llm_ssl_verify is False
