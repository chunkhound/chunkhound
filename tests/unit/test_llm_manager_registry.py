from typing import Any

import pytest

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT, LLMConfig
from chunkhound.core.exceptions.core import ConfigurationError
from chunkhound.interfaces.llm_provider import (
    PROVIDER_MANAGED_OUTPUT,
    LLMProvider,
    LLMResponse,
    OutputLimitCapability,
    OutputLimitDecisionKind,
    OutputLimitMetadata,
)
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


@pytest.mark.parametrize(
    ("configured_mode", "expected_enabled"),
    [(None, False), (False, False), (True, True)],
    ids=["omitted", "explicit-provider-managed", "explicit-legacy"],
)
def test_synthesis_output_policy_flows_from_config_through_manager(
    monkeypatch: pytest.MonkeyPatch,
    configured_mode: bool | None,
    expected_enabled: bool,
) -> None:
    """The selected synthesis provider owns the fully resolved config policy."""

    class _RecordingProvider(LLMProvider):
        def __init__(self, model: str = "recording-model", **_: Any) -> None:
            self._model = model

        @property
        def name(self) -> str:
            return "recording"

        @property
        def model(self) -> str:
            return self._model

        @property
        def timeout(self) -> int:
            return 30

        @property
        def output_limit_metadata(self) -> OutputLimitMetadata:
            return OutputLimitMetadata(
                omission=OutputLimitCapability.REQUIRED,
                declared_max_tokens=91_337,
                declared_max_source="https://provider.example/docs/output-limits",
            )

        async def complete(
            self,
            prompt: str,
            system: str | None = None,
            max_completion_tokens: int = 4096,
            timeout: int | None = None,
        ) -> LLMResponse:
            raise NotImplementedError

        async def batch_complete(
            self,
            prompts: list[str],
            system: str | None = None,
            max_completion_tokens: int = 4096,
        ) -> list[LLMResponse]:
            raise NotImplementedError

        def estimate_tokens(self, text: str) -> int:
            return len(text)

        async def health_check(self) -> dict[str, Any]:
            return {"status": "ok"}

        def get_usage_stats(self) -> dict[str, Any]:
            return {}

    monkeypatch.setitem(LLMManager._providers, "openai", _RecordingProvider)
    kwargs: dict[str, Any] = {"output_limit_fallback": 78_901}
    if configured_mode is not None:
        kwargs["output_limits_enabled"] = configured_mode
    utility_config, synthesis_config = LLMConfig(**kwargs).get_provider_configs()

    manager = LLMManager(utility_config, synthesis_config)
    utility = manager.get_utility_provider()
    synthesis = manager.get_synthesis_provider()
    policy = synthesis.synthesis_output_limit_policy

    assert "output_limits_enabled" not in utility_config
    assert "output_limit_fallback" not in utility_config
    assert not hasattr(utility, "_synthesis_output_limit_policy")
    assert policy.output_limits_enabled is expected_enabled
    assert policy.fallback_tokens == 78_901
    assert policy.metadata.omission is OutputLimitCapability.REQUIRED
    assert policy.metadata.declared_max_tokens == 91_337
    assert policy.metadata.declared_max_source == (
        "https://provider.example/docs/output-limits"
    )

    if expected_enabled:
        explicit = policy.resolve(30_000)
        assert explicit.kind is OutputLimitDecisionKind.EXPLICIT
        assert explicit.max_tokens == 30_000
    else:
        managed = policy.resolve(PROVIDER_MANAGED_OUTPUT)
        assert managed.kind is OutputLimitDecisionKind.DECLARATION
        assert managed.max_tokens == 91_337


@pytest.mark.parametrize("provider_name", ["deepseek", "grok"])
@pytest.mark.parametrize(
    ("base_url", "expected_capability", "expected_kind", "expected_tokens"),
    [
        (
            None,
            OutputLimitCapability.SUPPORTED,
            OutputLimitDecisionKind.OMIT,
            None,
        ),
        (
            "https://compatible.example/v1",
            OutputLimitCapability.UNKNOWN,
            OutputLimitDecisionKind.FALLBACK,
            64_123,
        ),
    ],
    ids=["canonical", "custom"],
)
def test_registry_endpoint_capability_flows_through_selected_synthesis_provider(
    provider_name: str,
    base_url: str | None,
    expected_capability: OutputLimitCapability,
    expected_kind: OutputLimitDecisionKind,
    expected_tokens: int | None,
) -> None:
    """The manager applies canonical metadata only to canonical endpoints."""
    role_config: dict[str, Any] = {
        "provider": provider_name,
        "model": "contract-test-model",
        "api_key": "sk-test",
    }
    if base_url is not None:
        role_config["base_url"] = base_url

    manager = LLMManager(
        role_config,
        {
            **role_config,
            "output_limits_enabled": False,
            "output_limit_fallback": 64_123,
        },
    )

    synthesis = manager.get_synthesis_provider()
    decision = synthesis.resolve_synthesis_output_limit(PROVIDER_MANAGED_OUTPUT)

    assert synthesis.output_limit_metadata.omission is expected_capability
    assert decision.kind is expected_kind
    assert decision.max_tokens == expected_tokens
