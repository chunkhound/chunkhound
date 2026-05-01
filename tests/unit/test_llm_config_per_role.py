import asyncio
import os
from pathlib import Path

import pytest
from pydantic import SecretStr

from chunkhound.core.config.llm_config import LLMConfig
from chunkhound.llm_manager import LLMManager
from tests.helpers import DummyProc


def test_llm_config_per_role_provider_overrides():
    # Red test: fields not yet present, or not applied
    cfg = LLMConfig(
        provider="openai",
        utility_provider="openai",  # keep existing utility
        synthesis_provider="codex-cli",  # switch synthesis to codex
        utility_model="gpt-5-nano",
        synthesis_model="codex",
    )

    util_conf, synth_conf = cfg.get_provider_configs()

    assert util_conf["provider"] == "openai"
    assert util_conf["model"] == "gpt-5-nano"

    assert synth_conf["provider"] == "codex-cli"
    assert synth_conf["model"] == "codex"


def test_llm_config_model_field_sets_both_roles():
    """Test that 'model' sets both utility and synthesis models."""
    cfg = LLMConfig(
        provider="grok",
        model="grok-4-1-fast-reas5oning",  # intentional typo to test
    )

    util_conf, synth_conf = cfg.get_provider_configs()

    assert util_conf["provider"] == "grok"
    assert util_conf["model"] == "grok-4-1-fast-reas5oning"

    assert synth_conf["provider"] == "grok"
    assert synth_conf["model"] == "grok-4-1-fast-reas5oning"


def test_llm_config_model_field_overridden_by_specific_models():
    """Test that utility_model and synthesis_model override the general model field."""
    cfg = LLMConfig(
        provider="grok",
        model="grok-4-1-fast-reasoning",
        utility_model="grok-4-1-fast-reas5oning",  # different model for utility
        synthesis_model="grok-4-1-fast-reas5oning",  # same as utility
    )

    util_conf, synth_conf = cfg.get_provider_configs()

    assert util_conf["model"] == "grok-4-1-fast-reas5oning"
    assert synth_conf["model"] == "grok-4-1-fast-reas5oning"


def test_llm_config_codex_reasoning_effort_per_role():
    cfg = LLMConfig(
        provider="codex-cli",
        utility_provider="codex-cli",
        synthesis_provider="codex-cli",
        utility_model="codex",
        synthesis_model="codex",
        codex_reasoning_effort="medium",
        codex_reasoning_effort_synthesis="high",
    )

    utility_config, synthesis_config = cfg.get_provider_configs()

    assert utility_config["reasoning_effort"] == "medium"
    assert synthesis_config["reasoning_effort"] == "high"

    cfg2 = LLMConfig(
        provider="codex-cli",
        utility_model="codex",
        synthesis_model="codex",
        codex_reasoning_effort_utility="minimal",
    )

    util2, synth2 = cfg2.get_provider_configs()
    assert util2["reasoning_effort"] == "minimal"
    assert "reasoning_effort" not in synth2


def test_openai_rejects_xhigh_reasoning_effort():
    with pytest.raises(ValueError, match="openai does not support"):
        LLMConfig(
            provider="openai",
            utility_model="gpt-5-nano",
            synthesis_model="gpt-5",
            codex_reasoning_effort="xhigh",
        )


def test_opencode_accepts_xhigh_reasoning_effort():
    cfg = LLMConfig(
        provider="opencode-cli",
        utility_model="opencode/gpt-5-nano",
        synthesis_model="opencode/gpt-5",
        codex_reasoning_effort="xhigh",
    )

    utility_config, synthesis_config = cfg.get_provider_configs()
    assert utility_config["reasoning_effort"] == "xhigh"
    assert synthesis_config["reasoning_effort"] == "xhigh"


@pytest.mark.asyncio
async def test_llm_codex_cli_status_reflects_configured_model_and_effort(
    monkeypatch,
    tmp_path: Path,
):
    """End-to-end check: LLMConfig -> LLMManager -> CodexCLI overlay config."""
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    cfg = LLMConfig(
        provider="codex-cli",
        utility_provider="codex-cli",
        synthesis_provider="codex-cli",
        utility_model="test-config-model",
        synthesis_model="test-config-model",
        codex_reasoning_effort_utility="low",
        codex_reasoning_effort_synthesis="high",
    )

    utility_config, synthesis_config = cfg.get_provider_configs()

    # Ensure we never touch a real Codex home or binary
    monkeypatch.setenv("CHUNKHOUND_CODEX_STDIN_FIRST", "0")
    monkeypatch.setenv("CHUNKHOUND_CODEX_CONFIG_OVERRIDE", "env")
    monkeypatch.setattr(
        CodexCLIProvider,
        "_get_base_codex_home",
        lambda self: None,
        raising=True,
    )
    monkeypatch.setattr(
        CodexCLIProvider,
        "_codex_available",
        lambda self: True,
        raising=True,
    )

    captured: dict[str, object] = {"env": None, "config_text": None}

    async def _fake_create_subprocess_exec(*args, **kwargs):  # noqa: ANN001
        env = kwargs.get("env", {})
        captured["env"] = env

        cfg_key = os.getenv("CHUNKHOUND_CODEX_CONFIG_ENV", "CODEX_CONFIG")
        cfg_path_str = env.get(cfg_key)

        model_name = "<missing>"
        effort_value = "<missing>"

        if isinstance(cfg_path_str, str):
            cfg_path = Path(cfg_path_str)
            if cfg_path.exists():
                text = cfg_path.read_text(encoding="utf-8")
                captured["config_text"] = text
                for line in text.splitlines():
                    if line.startswith("model ="):
                        model_name = line.split("=", 1)[1].strip().strip('"')
                    if line.startswith("model_reasoning_effort ="):
                        effort_value = line.split("=", 1)[1].strip().strip('"')

        # Simulate a `/status`-style response from Codex
        status_text = f"MODEL={model_name};REASONING_EFFORT={effort_value}"
        return DummyProc(rc=0, out=status_text.encode("utf-8"), err=b"")

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        _fake_create_subprocess_exec,
        raising=True,
    )

    llm_manager = LLMManager(utility_config, synthesis_config)
    provider = llm_manager.get_synthesis_provider()

    response = await provider.complete(prompt="/status")

    assert "MODEL=test-config-model" in response.content
    assert "REASONING_EFFORT=high" in response.content


def test_llm_manager_forwards_anthropic_extended_config(monkeypatch):
    captured: list[dict[str, object]] = []

    class FakeAnthropicProvider:
        def __init__(self, **kwargs):  # noqa: ANN001
            captured.append(kwargs)
            self.model = kwargs["model"]

    monkeypatch.setitem(LLMManager._providers, "anthropic", FakeAnthropicProvider)

    cfg = LLMConfig(
        provider="anthropic",
        api_key=SecretStr("sk-ant-test"),
        utility_model="claude-opus-4-7",
        synthesis_model="claude-opus-4-7",
        anthropic_thinking_enabled=True,
        anthropic_thinking_mode="adaptive",
        anthropic_thinking_display="summarized",
        anthropic_effort="xhigh",
        anthropic_prompt_caching=True,
        anthropic_cache_ttl="1h",
        anthropic_task_budget_tokens=20000,
        anthropic_context_management_enabled=True,
        anthropic_clear_thinking_keep_turns=2,
        anthropic_clear_tool_uses_trigger_tokens=1000,
        anthropic_clear_tool_uses_keep=3,
    )

    utility_config, synthesis_config = cfg.get_provider_configs()
    LLMManager(utility_config, synthesis_config)

    assert len(captured) == 2
    for kwargs in captured:
        assert kwargs["api_key"] == "sk-ant-test"
        assert kwargs["model"] == "claude-opus-4-7"
        assert kwargs["thinking_enabled"] is True
        assert kwargs["thinking_mode"] == "adaptive"
        assert kwargs["thinking_display"] == "summarized"
        assert kwargs["effort"] == "xhigh"
        assert kwargs["prompt_caching"] is True
        assert kwargs["cache_ttl"] == "1h"
        assert kwargs["task_budget_tokens"] == 20000
        assert kwargs["context_management_enabled"] is True
        assert kwargs["clear_thinking_keep_turns"] == 2
        assert kwargs["clear_tool_uses_trigger_tokens"] == 1000
        assert kwargs["clear_tool_uses_keep"] == 3


def test_grok_config_validation_with_api_key():
    """Test that Grok config is valid when API key is provided."""
    cfg = LLMConfig(
        provider="grok",
        api_key=SecretStr("sk-test-key"),
        model="grok-4-1-fast-reasoning",
    )

    assert cfg.is_provider_configured() is True
    assert cfg.get_missing_config() == []


def test_grok_config_validation_without_api_key():
    """Test that Grok config is invalid when API key is missing."""
    cfg = LLMConfig(
        provider="grok",
        model="grok-4-1-fast-reasoning",
    )

    assert cfg.is_provider_configured() is False
    missing = cfg.get_missing_config()
    assert len(missing) == 1
    assert "api_key" in missing[0]
    assert "CHUNKHOUND_LLM_API_KEY" in missing[0]


def test_grok_config_validation_per_role_utility():
    """Test Grok config validation for utility role specifically."""
    cfg = LLMConfig(
        provider="openai",  # default
        utility_provider="grok",
        utility_model="grok-4-1-fast-reasoning",
        api_key=SecretStr("sk-test-key"),
    )

    # Should be configured for utility role
    assert cfg.is_provider_configured() is True

    util_conf, synth_conf = cfg.get_provider_configs()
    assert util_conf["provider"] == "grok"
    assert util_conf["api_key"] == "sk-test-key"


def test_grok_config_validation_per_role_synthesis():
    """Test Grok config validation for synthesis role specifically."""
    cfg = LLMConfig(
        provider="grok",
        synthesis_provider="grok",
        synthesis_model="grok-4-1-fast-reasoning",
        api_key=SecretStr("sk-test-key"),
    )

    # Should be configured for synthesis role
    assert cfg.is_provider_configured() is True

    util_conf, synth_conf = cfg.get_provider_configs()
    assert synth_conf["provider"] == "grok"
    assert synth_conf["api_key"] == "sk-test-key"


def test_grok_config_validation_missing_api_key_per_role():
    """Test Grok config validation fails when API key missing for per-role config."""
    cfg = LLMConfig(
        provider="openai",  # default
        utility_provider="grok",
        synthesis_provider="grok",
        utility_model="grok-4-1-fast-reasoning",
        synthesis_model="grok-4-1-fast-reasoning",
        # No api_key provided
    )

    # Should not be configured since Grok requires API key
    assert cfg.is_provider_configured() is False
    missing = cfg.get_missing_config()
    assert len(missing) == 1
    assert "api_key" in missing[0]


def test_opencode_cli_reasoning_effort_in_provider_configs():
    """Test that reasoning_effort is included in opencode-cli provider configs."""
    cfg = LLMConfig(
        provider="opencode-cli",
        utility_provider="opencode-cli",
        synthesis_provider="opencode-cli",
        utility_model="openai/gpt-5-nano",
        synthesis_model="openai/gpt-5-nano",
        codex_reasoning_effort="medium",
        codex_reasoning_effort_synthesis="high",
    )

    utility_config, synthesis_config = cfg.get_provider_configs()

    assert utility_config["reasoning_effort"] == "medium"
    assert synthesis_config["reasoning_effort"] == "high"


def test_opencode_cli_is_provider_configured_without_api_key():
    """Test that opencode-cli is considered configured without an API key."""
    cfg = LLMConfig(
        provider="opencode-cli",
        utility_provider="opencode-cli",
        synthesis_provider="opencode-cli",
        utility_model="openai/gpt-5-nano",
        synthesis_model="openai/gpt-5-nano",
    )

    assert cfg.is_provider_configured() is True


def test_opencode_cli_get_missing_config_empty():
    """Test that opencode-cli returns no missing config items (no API key needed)."""
    cfg = LLMConfig(
        provider="opencode-cli",
        utility_provider="opencode-cli",
        synthesis_provider="opencode-cli",
        utility_model="openai/gpt-5-nano",
        synthesis_model="openai/gpt-5-nano",
    )

    assert cfg.get_missing_config() == []


def test_mixed_provider_get_missing_config_requires_api_key():
    """Mixed per-role providers should require an API key if either role needs one."""
    cfg = LLMConfig(
        provider="opencode-cli",
        utility_provider="opencode-cli",
        synthesis_provider="openai",
        utility_model="openai/gpt-5-nano",
        synthesis_model="gpt-5",
    )

    assert cfg.is_provider_configured() is False
    assert cfg.get_missing_config() == ["api_key (set CHUNKHOUND_LLM_API_KEY)"]


def test_opencode_cli_model_validator_empty_model_raises():
    """Ensure empty model with opencode-cli provider raises ValueError."""
    with pytest.raises(ValueError, match="opencode-cli requires a model"):
        LLMConfig(
            provider="opencode-cli",
            utility_model="",
            synthesis_model="openai/gpt-5-nano",
        )


def test_opencode_cli_model_validator_no_slash_raises():
    """Ensure model without provider/model format raises ValueError."""
    with pytest.raises(ValueError, match="opencode-cli requires a model"):
        LLMConfig(
            provider="opencode-cli",
            utility_model="openai/gpt-5-nano",
            synthesis_model="no-slash",
        )


def test_opencode_cli_model_validator_per_role_override():
    """Ensure per-role override provider triggers model validation."""
    with pytest.raises(ValueError, match="opencode-cli requires a model"):
        LLMConfig(
            provider="openai",
            synthesis_provider="opencode-cli",
            synthesis_model="bad-model",
            utility_model="whatever",
        )


def test_opencode_cli_model_validator_valid_models_pass():
    """Ensure valid provider/model format passes validation."""
    cfg = LLMConfig(
        provider="opencode-cli",
        utility_model="provider-a/model-1",
        synthesis_model="provider-b/model-2",
    )
    utility_config, synthesis_config = cfg.get_provider_configs()
    assert utility_config["provider"] == "opencode-cli"
    assert synthesis_config["provider"] == "opencode-cli"
