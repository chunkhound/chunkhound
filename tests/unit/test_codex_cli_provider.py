from unittest.mock import patch

import pytest


def test_codex_cli_provider_import_and_name():
    # Red test: module does not exist yet
    from chunkhound.providers.llm.codex_cli_provider import (
        CodexCLIProvider,  # type: ignore[attr-defined]
    )

    provider = CodexCLIProvider(model="codex")
    assert provider.name == "codex-cli"


def test_codex_cli_model_resolution_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from chunkhound.providers.llm.codex_cli_provider import (
        CodexCLIProvider,  # type: ignore[attr-defined]
    )

    monkeypatch.delenv("CHUNKHOUND_CODEX_DEFAULT_MODEL", raising=False)
    with patch.object(
        CodexCLIProvider,
        "get_highest_priority_available_model",
        return_value="test-discovered-model",
    ):
        resolved, source = CodexCLIProvider.describe_model_resolution("codex")
    assert resolved == "test-discovered-model"
    assert source == "discovered"


def test_codex_cli_model_resolution_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from chunkhound.providers.llm.codex_cli_provider import (
        CodexCLIProvider,  # type: ignore[attr-defined]
    )

    monkeypatch.delenv("CHUNKHOUND_CODEX_DEFAULT_MODEL", raising=False)
    with patch.object(
        CodexCLIProvider,
        "get_highest_priority_available_model",
        return_value=None,
    ):
        with pytest.raises(
            RuntimeError, match="Codex model discovery failed"
        ):
            CodexCLIProvider.describe_model_resolution("codex")


def test_codex_cli_model_resolution_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from chunkhound.providers.llm.codex_cli_provider import (
        CodexCLIProvider,  # type: ignore[attr-defined]
    )

    monkeypatch.setenv("CHUNKHOUND_CODEX_DEFAULT_MODEL", "test-env-override-model")
    resolved, source = CodexCLIProvider.describe_model_resolution("codex")
    assert resolved == "test-env-override-model"
    assert source == "env:CHUNKHOUND_CODEX_DEFAULT_MODEL"


def test_codex_cli_effort_resolution_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from chunkhound.providers.llm.codex_cli_provider import (
        CodexCLIProvider,  # type: ignore[attr-defined]
    )

    monkeypatch.delenv("CHUNKHOUND_CODEX_REASONING_EFFORT", raising=False)
    resolved, source = CodexCLIProvider.describe_reasoning_effort_resolution(None)
    assert resolved == "low"
    assert source == "default"
