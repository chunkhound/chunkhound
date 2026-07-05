import pytest

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.core.exceptions.core import ConfigurationError
from chunkhound.llm_manager import LLMManager


def test_llm_manager_registry_includes_codex_cli():
    assert "codex-cli" in LLMManager._providers


def test_list_providers_includes_registry_backed_providers():
    manager = object.__new__(LLMManager)
    manager._providers = LLMManager._providers

    provider_names = manager.list_providers()

    assert "deepseek" in provider_names
    assert "grok" in provider_names


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

    with pytest.raises(ValueError) as exc:
        manager._create_provider(  # type: ignore[attr-defined]
            {
                "provider": "grok",
                "base_url": "http://localhost:11434/v1",
                "api_key": "sk-test-key",
            }
        )
    # Registry providers fail with "Model is required" (no baked-in default).
    assert "Model is required" in str(exc.value)


def test_create_provider_keeps_provider_default_model_when_omitted():
    """Manager should not inject an OpenAI default into non-OpenAI providers."""
    manager = object.__new__(LLMManager)
    manager._providers = LLMManager._providers

    provider = manager._create_provider({"provider": "opencode-cli"})  # type: ignore[attr-defined]
    assert provider.model == ""  # opencode-cli has no default — user must specify model


def test_create_provider_requires_model_for_gemini_public_factory():
    """Public factory must enforce Gemini's explicit-model contract too."""
    manager = object.__new__(LLMManager)
    manager._providers = LLMManager._providers

    with pytest.raises(ConfigurationError, match="Model is required for 'gemini'"):
        manager.create_provider_for_config(
            {"provider": "gemini", "api_key": "sk-test-key"}
        )


def test_create_provider_passes_base_url_to_anthropic_provider():
    """Anthropic provider receives base_url outside the OpenAI-compatible path."""
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


def test_llm_manager_registry_antigravity_providers():
    assert "antigravity-sdk" in LLMManager._providers
    assert "antigravity-cli" in LLMManager._providers

    manager = object.__new__(LLMManager)
    manager._target_dir = "/fake/workspace"

    # Mocking classes to capture arguments
    captured_sdk = {}
    captured_cli = {}

    class FakeSDKProvider:
        def __init__(self, **kwargs):
            captured_sdk.update(kwargs)

    class FakeCLIProvider:
        def __init__(self, **kwargs):
            captured_cli.update(kwargs)

    manager._providers = {
        **LLMManager._providers,
        "antigravity-sdk": FakeSDKProvider,
        "antigravity-cli": FakeCLIProvider,
    }

    # Verify SDK creation parameters
    manager._create_provider(  # type: ignore[attr-defined]
        {
            "provider": "antigravity-sdk",
            "model": "gemini-3.5-flash",
            "api_key": "sk-test-key",
            "timeout": 45,
            "max_retries": 2,
        }
    )
    assert captured_sdk["model"] == "gemini-3.5-flash"
    assert captured_sdk["api_key"] == "sk-test-key"
    assert captured_sdk["timeout"] == 45
    assert captured_sdk["max_retries"] == 2
    assert captured_sdk["target_dir"] == "/fake/workspace"

    # Verify CLI creation parameters (no key needed)
    manager._create_provider(  # type: ignore[attr-defined]
        {
            "provider": "antigravity-cli",
            "model": "gemini-3.1-pro",
            "timeout": 60,
        }
    )
    assert captured_cli["model"] == "gemini-3.1-pro"
    assert captured_cli["timeout"] == 60


def test_llm_manager_registry_antigravity_sdk_missing_dependency(monkeypatch):
    from chunkhound.providers.llm import antigravity_llm_provider
    monkeypatch.setattr(antigravity_llm_provider, "SDK_AVAILABLE", False)
    
    with pytest.raises(RuntimeError, match="chunkhound\\[antigravity\\]"):
        antigravity_llm_provider.AntigravityLLMProvider()
