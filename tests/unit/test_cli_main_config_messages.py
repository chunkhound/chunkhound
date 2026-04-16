from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from chunkhound.api.cli import main as cli_main
from chunkhound.api.cli.utils.config_factory import create_validated_config


class _Parser:
    def __init__(self, args: SimpleNamespace) -> None:
        self._args = args

    def parse_args(self) -> SimpleNamespace:
        return self._args

    def print_help(self) -> None:
        raise AssertionError("print_help should not be called in this test")


@pytest.mark.asyncio
async def test_async_main_shows_web_configurator_banner_for_tty(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = SimpleNamespace(
        command="index",
        verbose=False,
        simulate=False,
        check_ignores=False,
    )
    monkeypatch.setattr(cli_main, "create_parser", lambda: _Parser(args))
    monkeypatch.setattr(cli_main, "setup_logging", lambda _verbose: None)
    monkeypatch.setattr(
        cli_main,
        "create_validated_config",
        lambda _args, _command: (
            object(),
            ["Missing required configuration: embedding provider"],
        ),
    )
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("sys.stderr.isatty", lambda: False)

    with pytest.raises(SystemExit) as excinfo:
        await cli_main.async_main()

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "Generate a config with the web configurator" in out
    assert "https://chunkhound.ai" in out
    assert ".chunkhound.json" in out


@pytest.mark.asyncio
async def test_async_main_shows_web_configurator_banner_for_llm_command_tty(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = SimpleNamespace(
        command="research",
        verbose=False,
        simulate=False,
        check_ignores=False,
    )
    monkeypatch.setattr(cli_main, "create_parser", lambda: _Parser(args))
    monkeypatch.setattr(cli_main, "setup_logging", lambda _verbose: None)
    monkeypatch.setattr(
        cli_main,
        "create_validated_config",
        lambda _args, _command: (
            object(),
            ["Missing required configuration: llm.api_key"],
        ),
    )
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("sys.stderr.isatty", lambda: False)

    with pytest.raises(SystemExit) as excinfo:
        await cli_main.async_main()

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "Generate a config with the web configurator" in out
    assert "https://chunkhound.ai" in out
    assert ".chunkhound.json" in out
    assert "--no-embeddings" not in out


@pytest.mark.asyncio
async def test_async_main_skips_web_configurator_banner_for_non_tty(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = SimpleNamespace(
        command="index",
        verbose=False,
        simulate=False,
        check_ignores=False,
    )
    monkeypatch.setattr(cli_main, "create_parser", lambda: _Parser(args))
    monkeypatch.setattr(cli_main, "setup_logging", lambda _verbose: None)
    monkeypatch.setattr(
        cli_main,
        "create_validated_config",
        lambda _args, _command: (
            object(),
            ["Missing required configuration: embedding provider"],
        ),
    )
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("sys.stderr.isatty", lambda: False)

    with pytest.raises(SystemExit) as excinfo:
        await cli_main.async_main()

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "Generate a config with the web configurator" not in out
    assert "To fix this, you can:" not in out
    assert "https://chunkhound.ai" not in out


@pytest.mark.asyncio
async def test_async_main_shows_web_configurator_banner_when_stdout_is_redirected_but_stderr_is_tty(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = SimpleNamespace(
        command="index",
        verbose=False,
        simulate=False,
        check_ignores=False,
    )
    monkeypatch.setattr(cli_main, "create_parser", lambda: _Parser(args))
    monkeypatch.setattr(cli_main, "setup_logging", lambda _verbose: None)
    monkeypatch.setattr(
        cli_main,
        "create_validated_config",
        lambda _args, _command: (
            object(),
            ["Missing required configuration: embedding provider"],
        ),
    )
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)

    with pytest.raises(SystemExit) as excinfo:
        await cli_main.async_main()

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "Generate a config with the web configurator" in out
    assert "https://chunkhound.ai" in out


def test_create_validated_config_reports_custom_llm_endpoint_missing_model(
    clean_environment,
    tmp_path,
):
    """Custom OpenAI-compatible LLM endpoints should fail via command validation."""
    args = SimpleNamespace(
        command="research",
        config=None,
        path=str(tmp_path),
        debug=False,
        verbose=False,
        llm_provider="openai",
        llm_base_url="http://localhost:11434/v1",
        llm_api_key=None,
        llm_utility_model=None,
        llm_synthesis_model=None,
        llm_utility_provider=None,
        llm_synthesis_provider=None,
        llm_codex_reasoning_effort=None,
        llm_codex_reasoning_effort_utility=None,
        llm_codex_reasoning_effort_synthesis=None,
        llm_map_hyde_provider=None,
        llm_map_hyde_model=None,
        llm_map_hyde_reasoning_effort=None,
        llm_autodoc_cleanup_provider=None,
        llm_autodoc_cleanup_model=None,
        llm_autodoc_cleanup_reasoning_effort=None,
    )

    config, errors = create_validated_config(args, "research")

    assert config.llm is not None
    assert any(
        "llm.explicit model selection required for custom OpenAI-compatible endpoint roles"
        in error
        for error in errors
    )


def test_create_validated_config_allows_autodoc_assets_only_without_llm() -> None:
    args = SimpleNamespace(
        command="autodoc",
        config=None,
        path=None,
        assets_only=True,
        debug=False,
        verbose=False,
    )

    _config, errors = create_validated_config(args, "autodoc")

    assert "No LLM provider configured" not in errors


def test_create_validated_config_allows_map_overview_only_without_llm() -> None:
    args = SimpleNamespace(
        command="map",
        config=None,
        path=None,
        overview_only=True,
        debug=False,
        verbose=False,
    )

    _config, errors = create_validated_config(args, "map")

    assert "No LLM provider configured" not in errors


def test_create_validated_config_reports_removed_ollama_provider_from_config_file(
    tmp_path,
) -> None:
    config_path = tmp_path / "legacy.json"
    config_path.write_text(
        json.dumps({"llm": {"provider": "ollama"}}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        command="research",
        config=str(config_path),
        path=str(tmp_path),
        debug=False,
        verbose=False,
    )

    config, errors = create_validated_config(args, "research")

    assert config.llm is None
    assert any("ollama" in error and "base_url" in error for error in errors)


def test_create_validated_config_preserves_fallback_state_on_invalid_json(tmp_path) -> None:
    config_path = tmp_path / "broken.json"
    config_path.write_text("{not-json", encoding="utf-8")
    args = SimpleNamespace(
        command="index",
        config=str(config_path),
        path=str(tmp_path),
        no_embeddings=True,
        debug=False,
        verbose=False,
    )

    config, errors = create_validated_config(args, "index")

    assert config.llm is None
    assert config.embedding is None
    assert config.target_dir == tmp_path.resolve()
    assert config.embeddings_disabled is True
    assert any("Invalid JSON in config file" in error for error in errors)


def test_create_validated_config_research_ignores_cleanup_only_override_requirements() -> None:
    args = SimpleNamespace(
        command="research",
        config=None,
        path=None,
        debug=False,
        verbose=False,
        llm_provider="codex-cli",
        llm_base_url=None,
        llm_api_key=None,
        llm_utility_model=None,
        llm_synthesis_model=None,
        llm_utility_provider=None,
        llm_synthesis_provider=None,
        llm_codex_reasoning_effort=None,
        llm_codex_reasoning_effort_utility=None,
        llm_codex_reasoning_effort_synthesis=None,
        llm_map_hyde_provider=None,
        llm_map_hyde_model=None,
        llm_map_hyde_reasoning_effort=None,
        llm_autodoc_cleanup_provider="anthropic",
        llm_autodoc_cleanup_model=None,
        llm_autodoc_cleanup_reasoning_effort=None,
    )

    _config, errors = create_validated_config(args, "research")

    assert errors == []
