import pytest

from chunkhound.api.cli.commands import autodoc_cleanup as autodoc_cleanup
from chunkhound.api.cli.utils.rich_output import RichOutputFormatter
from chunkhound.core.config.config import Config
from chunkhound.core.config.llm_config import LLMConfig


def test_cleanup_uses_autodoc_cleanup_overrides() -> None:
    llm_config = LLMConfig(
        provider="codex-cli",
        synthesis_provider="codex-cli",
        synthesis_model="gpt-base",
        utility_model="gpt-util",
        autodoc_cleanup_model="test-cleanup-model",
        autodoc_cleanup_reasoning_effort="medium",
    )

    _, synthesis = autodoc_cleanup._build_cleanup_provider_configs(llm_config)

    assert synthesis["provider"] == "codex-cli"
    assert synthesis["model"] == "test-cleanup-model"
    assert synthesis["reasoning_effort"] == "medium"


def test_cleanup_provider_override_drops_inherited_reasoning_effort() -> None:
    llm_config = LLMConfig(
        provider="opencode-cli",
        synthesis_provider="opencode-cli",
        synthesis_model="openai/gpt-5",
        utility_model="openai/gpt-5-nano",
        codex_reasoning_effort_synthesis="xhigh",
        autodoc_cleanup_provider="openai",
        autodoc_cleanup_model="gpt-5",
        api_key="test",
        base_url="https://source-provider.example/v1",
    )

    _, synthesis = autodoc_cleanup._build_cleanup_provider_configs(llm_config)

    assert synthesis["provider"] == "openai"
    assert synthesis["model"] == "gpt-5"
    assert "api_key" in synthesis  # preserved for keyed provider
    assert synthesis["base_url"] == "https://source-provider.example/v1"
    assert "reasoning_effort" not in synthesis


def test_cleanup_provider_override_requires_explicit_model_on_provider_switch() -> None:
    """Provider switches require an explicit model — even within the same family."""
    with pytest.raises(
        ValueError,
        match=(
            "autodoc_cleanup provider override requires an explicit "
            "autodoc_cleanup_model"
        ),
    ):
        LLMConfig(
            provider="openai",
            synthesis_model="gpt-5",
            utility_model="gpt-5-nano",
            autodoc_cleanup_provider="deepseek",
            api_key="test",
        )


def test_cleanup_provider_override_drops_effort_for_non_reasoning_effort_provider():
    config = LLMConfig(
        provider="opencode-cli",
        synthesis_model="openai/gpt-5-nano",
        utility_model="openai/gpt-5-nano",
        codex_reasoning_effort_synthesis="xhigh",
        autodoc_cleanup_provider="anthropic",
        autodoc_cleanup_model="claude-opus-4-7",
        autodoc_cleanup_reasoning_effort="high",
    )
    _, synthesis_config = autodoc_cleanup._build_cleanup_provider_configs(config)
    assert synthesis_config["provider"] == "anthropic"
    assert "reasoning_effort" not in synthesis_config


def test_cleanup_provider_override_preserves_explicit_effort() -> None:
    config = LLMConfig(
        provider="opencode-cli",
        synthesis_model="openai/gpt-5",
        utility_model="openai/gpt-5-nano",
        codex_reasoning_effort_synthesis="xhigh",
        autodoc_cleanup_provider="openai",
        autodoc_cleanup_model="gpt-5",
        autodoc_cleanup_reasoning_effort="high",
        api_key="test",
    )

    _, synthesis_config = autodoc_cleanup._build_cleanup_provider_configs(config)

    assert synthesis_config["provider"] == "openai"
    assert synthesis_config["reasoning_effort"] == "high"


def test_cleanup_provider_override_drops_inherited_structured_outputs_true() -> None:
    config = LLMConfig(
        provider="openai",
        synthesis_model="gpt-5",
        utility_model="gpt-5-nano",
        supports_structured_outputs=True,
        autodoc_cleanup_provider="deepseek",
        autodoc_cleanup_model="deepseek-v4-flash",
        api_key="test",
        base_url="https://api.openai.example/v1",
    )

    _, synthesis_config = autodoc_cleanup._build_cleanup_provider_configs(config)

    assert synthesis_config["provider"] == "deepseek"
    assert "api_key" in synthesis_config  # preserved for keyed provider
    assert synthesis_config["base_url"] == "https://api.openai.example/v1"
    assert "supports_structured_outputs" not in synthesis_config


def test_cleanup_provider_override_drops_inherited_structured_outputs_false() -> None:
    config = LLMConfig(
        provider="deepseek",
        synthesis_model="deepseek-v4-flash",
        utility_model="deepseek-v4-flash",
        supports_structured_outputs=False,
        autodoc_cleanup_provider="openai",
        autodoc_cleanup_model="gpt-5",
        api_key="test",
    )

    _, synthesis_config = autodoc_cleanup._build_cleanup_provider_configs(config)

    assert synthesis_config["provider"] == "openai"
    assert "supports_structured_outputs" not in synthesis_config


def test_cleanup_same_provider_preserves_explicit_structured_outputs() -> None:
    config = LLMConfig(
        provider="openai",
        synthesis_model="gpt-5",
        utility_model="gpt-5-nano",
        supports_structured_outputs=False,
        autodoc_cleanup_provider="openai",
        autodoc_cleanup_model="gpt-5-mini",
        api_key="test",
    )

    _, synthesis_config = autodoc_cleanup._build_cleanup_provider_configs(config)

    assert synthesis_config["provider"] == "openai"
    assert synthesis_config["supports_structured_outputs"] is False


def test_cleanup_same_provider_does_not_inherit_synthesis_reasoning_effort() -> None:
    config = LLMConfig(
        provider="openai",
        synthesis_model="gpt-5",
        utility_model="gpt-5-nano",
        codex_reasoning_effort_synthesis="high",
        autodoc_cleanup_provider="openai",
        autodoc_cleanup_model="gpt-5-mini",
        api_key="test",
    )

    _, synthesis_config = autodoc_cleanup._build_cleanup_provider_configs(config)

    assert synthesis_config["provider"] == "openai"
    assert "reasoning_effort" not in synthesis_config


def test_cleanup_accepts_local_openai_compatible_llm_without_api_key(
    clean_environment,
    tmp_path,
) -> None:
    config = Config(
        target_dir=tmp_path,
        llm={
            "provider": "openai",
            "model": "llama3.2",
            "base_url": "http://localhost:11434/v1",
        },
    )
    formatter = RichOutputFormatter()

    resolved = autodoc_cleanup._resolve_llm_config_for_cleanup(
        config=config,
        formatter=formatter,
    )

    assert resolved is not None
    assert resolved.is_provider_configured() is True


def test_cleanup_rejects_custom_openai_compatible_llm_without_explicit_model(
    clean_environment,
    tmp_path,
) -> None:
    config = Config(
        target_dir=tmp_path,
        llm={
            "provider": "openai",
            "base_url": "http://localhost:11434/v1",
        },
    )
    formatter = RichOutputFormatter()

    resolved = autodoc_cleanup._resolve_llm_config_for_cleanup(
        config=config,
        formatter=formatter,
    )

    assert resolved is None


def test_cleanup_override_preserves_custom_openai_compatible_base_url() -> None:
    llm_config = LLMConfig(
        provider="anthropic",
        synthesis_model="claude-sonnet-4-5-20250929",
        autodoc_cleanup_provider="openai",
        autodoc_cleanup_model="llama3.2",
        base_url="http://localhost:11434/v1",
        api_key="test-key",
    )

    _, synthesis = autodoc_cleanup._build_cleanup_provider_configs(llm_config)

    assert synthesis["provider"] == "openai"
    assert synthesis["model"] == "llama3.2"
    assert synthesis["base_url"] == "http://localhost:11434/v1"
