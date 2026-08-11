"""Contracts for research synthesis output-limit configuration."""

import argparse
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsError

from chunkhound.core.config.llm_config import LLMConfig


def _clear_output_limit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHUNKHOUND_LLM_OUTPUT_LIMITS_ENABLED", raising=False)
    monkeypatch.delenv("CHUNKHOUND_LLM_OUTPUT_LIMIT_FALLBACK", raising=False)


def test_output_limit_defaults_apply_only_to_synthesis_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_output_limit_env(monkeypatch)

    config = LLMConfig()
    utility, synthesis = config.get_provider_configs()

    assert config.output_limits_enabled is False
    assert config.output_limit_fallback == 64_000
    assert "output_limits_enabled" not in utility
    assert "output_limit_fallback" not in utility
    assert synthesis["output_limits_enabled"] is False
    assert synthesis["output_limit_fallback"] == 64_000


@pytest.mark.parametrize("enabled", [False, True])
def test_output_limit_mode_accepts_only_explicit_booleans(enabled: bool) -> None:
    config = LLMConfig(output_limits_enabled=enabled)

    assert config.output_limits_enabled is enabled


@pytest.mark.parametrize(
    "invalid",
    [None, 0, 1, "false", "true", [], {}, 1.0],
    ids=lambda value: f"{type(value).__name__}-{value!r}",
)
def test_output_limit_mode_rejects_non_boolean_native_values(invalid: object) -> None:
    with pytest.raises(ValidationError):
        LLMConfig(output_limits_enabled=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("fallback", [1, 64_000, 123_456])
def test_output_limit_fallback_accepts_positive_integers(fallback: int) -> None:
    assert LLMConfig(output_limit_fallback=fallback).output_limit_fallback == fallback


@pytest.mark.parametrize(
    "invalid",
    [None, False, True, 0, -1, 1.5, "64000", [], {}],
    ids=lambda value: f"{type(value).__name__}-{value!r}",
)
def test_output_limit_fallback_rejects_invalid_native_values(invalid: object) -> None:
    with pytest.raises(ValidationError):
        LLMConfig(output_limit_fallback=invalid)  # type: ignore[arg-type]


def test_output_limit_environment_values_are_canonically_decoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNKHOUND_LLM_OUTPUT_LIMITS_ENABLED", "off")
    monkeypatch.setenv("CHUNKHOUND_LLM_OUTPUT_LIMIT_FALLBACK", "77777")

    loaded = LLMConfig.load_from_env()
    config = LLMConfig(**loaded)
    direct = LLMConfig()

    assert loaded == {
        "output_limits_enabled": False,
        "output_limit_fallback": 77_777,
    }
    assert config.output_limits_enabled is False
    assert config.output_limit_fallback == 77_777
    assert direct.output_limits_enabled is False
    assert direct.output_limit_fallback == 77_777


def test_invalid_output_limit_environment_boolean_is_not_silently_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNKHOUND_LLM_OUTPUT_LIMITS_ENABLED", "sometimes")

    with pytest.raises(ValueError, match="CHUNKHOUND_LLM_OUTPUT_LIMITS_ENABLED"):
        LLMConfig.load_from_env()
    with pytest.raises(SettingsError, match="output_limits_enabled"):
        LLMConfig()


def test_invalid_output_limit_environment_fallback_is_not_silently_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNKHOUND_LLM_OUTPUT_LIMIT_FALLBACK", "lots")

    with pytest.raises(ValueError, match="CHUNKHOUND_LLM_OUTPUT_LIMIT_FALLBACK"):
        LLMConfig.load_from_env()
    with pytest.raises(SettingsError, match="output_limit_fallback"):
        LLMConfig()


def test_output_limit_cli_values_are_canonically_decoded() -> None:
    parser = argparse.ArgumentParser()
    LLMConfig.add_cli_arguments(parser)

    args = parser.parse_args(
        ["--no-llm-output-limits-enabled", "--llm-output-limit-fallback", "88888"]
    )

    assert LLMConfig.extract_cli_overrides(args) == {
        "output_limits_enabled": False,
        "output_limit_fallback": 88_888,
    }


def test_output_limit_dotenv_values_are_canonically_decoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_output_limit_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CHUNKHOUND_LLM_OUTPUT_LIMITS_ENABLED=true\n"
        "CHUNKHOUND_LLM_OUTPUT_LIMIT_FALLBACK=77777\n"
    )

    config = LLMConfig(_env_file=env_file)

    assert config.output_limits_enabled is True
    assert config.output_limit_fallback == 77_777


@pytest.mark.parametrize(
    ("name", "value", "field_name"),
    [
        ("CHUNKHOUND_LLM_OUTPUT_LIMITS_ENABLED", "sometimes", "output_limits_enabled"),
        ("CHUNKHOUND_LLM_OUTPUT_LIMIT_FALLBACK", "lots", "output_limit_fallback"),
    ],
)
def test_invalid_output_limit_dotenv_values_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    field_name: str,
) -> None:
    _clear_output_limit_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text(f"{name}={value}\n")

    with pytest.raises(SettingsError, match=field_name):
        LLMConfig(_env_file=env_file)


def test_output_limit_file_secret_values_are_canonically_decoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_output_limit_env(monkeypatch)
    (tmp_path / "CHUNKHOUND_LLM_OUTPUT_LIMITS_ENABLED").write_text("on")
    (tmp_path / "CHUNKHOUND_LLM_OUTPUT_LIMIT_FALLBACK").write_text("88888")

    config = LLMConfig(_secrets_dir=tmp_path)

    assert config.output_limits_enabled is True
    assert config.output_limit_fallback == 88_888


@pytest.mark.parametrize(
    ("name", "value", "field_name"),
    [
        ("CHUNKHOUND_LLM_OUTPUT_LIMITS_ENABLED", "sometimes", "output_limits_enabled"),
        ("CHUNKHOUND_LLM_OUTPUT_LIMIT_FALLBACK", "lots", "output_limit_fallback"),
    ],
)
def test_invalid_output_limit_file_secret_values_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    field_name: str,
) -> None:
    _clear_output_limit_env(monkeypatch)
    (tmp_path / name).write_text(value)

    with pytest.raises(SettingsError, match=field_name):
        LLMConfig(_secrets_dir=tmp_path)
