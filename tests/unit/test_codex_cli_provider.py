from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def clear_codex_model_discovery_cache():
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    CodexCLIProvider.get_highest_priority_available_model.cache_clear()
    yield
    CodexCLIProvider.get_highest_priority_available_model.cache_clear()


def test_codex_cli_provider_import_and_name():
    # Red test: module does not exist yet
    from chunkhound.providers.llm.codex_cli_provider import (
        CodexCLIProvider,  # type: ignore[attr-defined]
    )

    with patch.object(CodexCLIProvider, "_codex_available", return_value=True):
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


def test_codex_cli_model_resolution_discovery_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chunkhound.providers.llm.codex_cli_provider import (
        CODEX_FALLBACK_MODEL,
        CodexCLIProvider,  # type: ignore[attr-defined]
    )

    monkeypatch.delenv("CHUNKHOUND_CODEX_DEFAULT_MODEL", raising=False)
    with patch.object(
        CodexCLIProvider,
        "get_highest_priority_available_model",
        return_value=None,
    ):
        resolved, source = CodexCLIProvider.describe_model_resolution("codex")
        assert resolved == CODEX_FALLBACK_MODEL
        assert source == "fallback"


def test_codex_cli_model_resolution_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_codex_cli_model_discovery_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    def fake_run(*args, **kwargs):  # noqa: ANN001, ARG001
        return type("Result", (), {"returncode": 1, "stdout": b""})()

    monkeypatch.setattr("subprocess.run", fake_run)

    assert CodexCLIProvider.get_highest_priority_available_model() is None


def test_codex_cli_model_discovery_no_visible_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    output = b'{"models":[{"slug":"hidden","visibility":"hidden","priority":10}]}\n'

    def fake_run(*args, **kwargs):  # noqa: ANN001, ARG001
        return type("Result", (), {"returncode": 0, "stdout": output})()

    monkeypatch.setattr("subprocess.run", fake_run)

    assert CodexCLIProvider.get_highest_priority_available_model() is None


def test_codex_cli_model_discovery_malformed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    def fake_run(*args, **kwargs):  # noqa: ANN001, ARG001
        return type("Result", (), {"returncode": 0, "stdout": b"not json\n"})()

    monkeypatch.setattr("subprocess.run", fake_run)

    assert CodexCLIProvider.get_highest_priority_available_model() is None


def test_codex_cli_model_discovery_priority_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    output = (
        b'{"models":['
        b'{"slug":"low","visibility":"list","priority":1},'
        b'{"slug":"high","visibility":"list","priority":20},'
        b'{"slug":"hidden","visibility":"hidden","priority":100}'
        b"]}\n"
    )

    def fake_run(*args, **kwargs):  # noqa: ANN001, ARG001
        return type("Result", (), {"returncode": 0, "stdout": output})()

    monkeypatch.setattr("subprocess.run", fake_run)

    assert CodexCLIProvider.get_highest_priority_available_model() == "high"
