import pytest


def test_codex_cli_provider_import_and_name():
    # Red test: module does not exist yet
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider  # type: ignore[attr-defined]

    provider = CodexCLIProvider(model="codex")
    assert provider.name == "codex-cli"


def test_codex_cli_estimate_tokens_ratio():
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider  # type: ignore[attr-defined]

    provider = CodexCLIProvider(model="codex")
    text = "x" * 400
    # Expect ~chars/4 tokens (like Claude CLI provider pattern)
    assert provider.estimate_tokens(text) == 100


def test_codex_cli_model_resolution_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider  # type: ignore[attr-defined]

    monkeypatch.delenv("CHUNKHOUND_CODEX_DEFAULT_MODEL", raising=False)
    resolved, source = CodexCLIProvider.describe_model_resolution("codex")
    assert resolved == "gpt-5.1-codex"
    assert source in {"default", "env:CHUNKHOUND_CODEX_DEFAULT_MODEL"}


def test_codex_cli_model_resolution_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider  # type: ignore[attr-defined]

    monkeypatch.setenv("CHUNKHOUND_CODEX_DEFAULT_MODEL", "gpt-5.2-codex")
    resolved, source = CodexCLIProvider.describe_model_resolution("codex")
    assert resolved == "gpt-5.2-codex"
    assert source == "env:CHUNKHOUND_CODEX_DEFAULT_MODEL"


def test_codex_cli_effort_resolution_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider  # type: ignore[attr-defined]

    monkeypatch.delenv("CHUNKHOUND_CODEX_REASONING_EFFORT", raising=False)
    resolved, source = CodexCLIProvider.describe_reasoning_effort_resolution(None)
    assert resolved == "low"
    assert source == "default"
