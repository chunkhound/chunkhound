import pytest

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.llm_manager import LLMManager


def test_llm_manager_registry_includes_codex_cli():
    assert "codex-cli" in LLMManager._providers


def test_create_provider_uses_default_timeout_when_omitted():
    """When config omits 'timeout', the created provider uses DEFAULT_LLM_TIMEOUT."""
    provider_class = LLMManager._providers["claude-code-cli"]
    provider = provider_class()
    assert provider.timeout == DEFAULT_LLM_TIMEOUT


def test_create_provider_requires_model_for_custom_openai_endpoint():
    """Custom OpenAI-compatible endpoints must not fall back to cloud defaults."""
    manager = object.__new__(LLMManager)
    manager._providers = LLMManager._providers

    with pytest.raises(ValueError, match="require an explicit model"):
        manager._create_provider(  # type: ignore[attr-defined]
            {"provider": "openai", "base_url": "http://localhost:11434/v1"}
        )


def test_create_provider_requires_model_for_custom_grok_endpoint():
    """Custom OpenAI-compatible Grok endpoints must also set an explicit model."""
    manager = object.__new__(LLMManager)
    manager._providers = LLMManager._providers

    with pytest.raises(ValueError, match="require an explicit model"):
        manager._create_provider(  # type: ignore[attr-defined]
            {
                "provider": "grok",
                "base_url": "http://localhost:11434/v1",
                "api_key": "sk-test-key",
            }
        )


def test_create_provider_keeps_provider_default_model_when_omitted():
    """Manager should not inject an OpenAI default into non-OpenAI providers."""
    manager = object.__new__(LLMManager)
    manager._providers = LLMManager._providers

    provider = manager._create_provider({"provider": "opencode-cli"})  # type: ignore[attr-defined]
    assert provider.model == "opencode/grok-code"


def test_create_provider_does_not_pass_base_url_to_non_openai_compatible_provider():
    """Anthropic custom endpoints must continue to receive base_url."""
    manager = object.__new__(LLMManager)

    captured: dict[str, object] = {}

    class _FakeAnthropicProvider:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    manager._providers = {**LLMManager._providers, "anthropic": _FakeAnthropicProvider}

    manager._create_provider(  # type: ignore[attr-defined]
        {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
            "api_key": "sk-test-key",
            "base_url": "http://localhost:11434/v1",
        }
    )

    assert captured["base_url"] == "http://localhost:11434/v1"
