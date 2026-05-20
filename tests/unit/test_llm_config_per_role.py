import asyncio
import os
from pathlib import Path

import pytest
from pydantic import SecretStr

from chunkhound.core.config.llm_config import LLMConfig
from chunkhound.llm_manager import LLMManager
from chunkhound.providers.llm.openai_compatible_provider import OpenAICompatibleProvider
from tests.helpers import DummyProc


def test_llm_config_per_role_provider_overrides():
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


def test_openai_role_overrides_reject_xhigh_reasoning_effort():
    for provider_field in ("utility_provider", "synthesis_provider"):
        with pytest.raises(ValueError, match="openai does not support"):
            LLMConfig(
                provider="opencode-cli",
                utility_model="openai/gpt-5-nano",
                synthesis_model="openai/gpt-5",
                codex_reasoning_effort="xhigh",
                **{provider_field: "openai"},
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


def test_opencode_cli_model_convenience_field_sets_both_roles():
    """Test that 'model' convenience field works for opencode-cli.

    Pydantic v2 runs model_validator *before* model_post_init,
    so the validator must fall back to self.model when role-specific
    fields are empty. This test guards against regressions in that
    ordering-dependent fix.
    """
    cfg = LLMConfig(
        provider="opencode-cli",
        model="opencode/gpt-5-nano",
    )

    util_conf, synth_conf = cfg.get_provider_configs()

    assert util_conf["provider"] == "opencode-cli"
    assert util_conf["model"] == "opencode/gpt-5-nano"
    assert synth_conf["provider"] == "opencode-cli"
    assert synth_conf["model"] == "opencode/gpt-5-nano"


def test_deepseek_does_not_forward_structured_outputs_override_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[dict[str, object]] = []

    class FakeDeepSeekProvider(OpenAICompatibleProvider):
        def __init__(self, **kwargs):  # noqa: ANN001
            captured.append(kwargs)
            self._model = kwargs["model"]

        def _get_default_base_url(self) -> str | None:
            return None

        def _get_provider_name(self) -> str:
            return "deepseek"

    monkeypatch.setitem(LLMManager._providers, "deepseek", FakeDeepSeekProvider)

    cfg = LLMConfig(provider="deepseek", api_key=SecretStr("sk-test"))
    utility_config, synthesis_config = cfg.get_provider_configs()

    assert "supports_structured_outputs" not in utility_config
    assert "supports_structured_outputs" not in synthesis_config

    LLMManager(utility_config, synthesis_config)

    assert len(captured) == 2
    for kwargs in captured:
        assert kwargs["model"] == "deepseek-v4-flash"
        assert "supports_structured_outputs" not in kwargs


def test_deepseek_forwards_structured_outputs_override(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[dict[str, object]] = []

    class FakeDeepSeekProvider(OpenAICompatibleProvider):
        def __init__(self, **kwargs):  # noqa: ANN001
            captured.append(kwargs)
            self._model = kwargs["model"]

        def _get_default_base_url(self) -> str | None:
            return None

        def _get_provider_name(self) -> str:
            return "deepseek"

    monkeypatch.setitem(LLMManager._providers, "deepseek", FakeDeepSeekProvider)

    cfg = LLMConfig(
        provider="deepseek",
        api_key=SecretStr("sk-test"),
        supports_structured_outputs=True,
    )
    utility_config, synthesis_config = cfg.get_provider_configs()

    assert utility_config["supports_structured_outputs"] is True
    assert synthesis_config["supports_structured_outputs"] is True

    LLMManager(utility_config, synthesis_config)

    assert len(captured) == 2
    for kwargs in captured:
        assert kwargs["supports_structured_outputs"] is True


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


def test_opencode_cli_model_validator_empty_segments_raise():
    """Ensure provider/model format requires both provider and model."""
    for bad_model in ("/gpt-5", "openai/", "openai/   "):
        with pytest.raises(ValueError, match="empty"):
            LLMConfig(
                provider="opencode-cli",
                utility_model=bad_model,
                synthesis_model="openai/gpt-5-nano",
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


def test_opencode_cli_model_validator_map_hyde_provider_requires_explicit_model():
    """Switching map_hyde providers must not inherit the synthesis model."""
    with pytest.raises(ValueError, match="map_hyde provider override requires"):
        LLMConfig(
            provider="openai",
            synthesis_model="gpt-5",
            utility_model="gpt-5-nano",
            map_hyde_provider="opencode-cli",
            api_key="test",
        )

    cfg = LLMConfig(
        provider="openai",
        synthesis_model="gpt-5",
        utility_model="gpt-5-nano",
        map_hyde_provider="opencode-cli",
        map_hyde_model="openai/gpt-5",
        api_key="test",
    )
    assert cfg.map_hyde_provider == "opencode-cli"


def test_opencode_cli_model_validator_autodoc_provider_requires_explicit_model():
    """Switching autodoc_cleanup providers must not inherit the synthesis model."""
    with pytest.raises(ValueError, match="autodoc_cleanup provider override requires"):
        LLMConfig(
            provider="openai",
            synthesis_model="gpt-5",
            utility_model="gpt-5-nano",
            autodoc_cleanup_provider="opencode-cli",
            api_key="test",
        )

    cfg = LLMConfig(
        provider="openai",
        synthesis_model="gpt-5",
        utility_model="gpt-5-nano",
        autodoc_cleanup_provider="opencode-cli",
        autodoc_cleanup_model="openai/gpt-5",
        api_key="test",
    )
    assert cfg.autodoc_cleanup_provider == "opencode-cli"


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


class TestDeepSeekProviderDefaults:
    def test_default_models_both_roles(self):
        cfg = LLMConfig(provider="deepseek", api_key="sk-test")
        utility_cfg, synthesis_cfg = cfg.get_provider_configs()
        assert utility_cfg["model"] == "deepseek-v4-flash"
        assert synthesis_cfg["model"] == "deepseek-v4-flash"

    def test_supports_structured_outputs_override_is_propagated(self):
        cfg = LLMConfig(
            provider="deepseek",
            api_key="sk-test",
            supports_structured_outputs=False,
        )
        _, synthesis_cfg = cfg.get_provider_configs()
        assert synthesis_cfg["supports_structured_outputs"] is False


def test_map_hyde_and_autodoc_cleanup_do_not_inherit_global_codex_reasoning_effort():
    """map_hyde and autodoc_cleanup don't inherit global codex_reasoning_effort.

    utility/synthesis fall back to codex_reasoning_effort; the secondary roles
    don't — they only use their own role-specific effort field or nothing.
    """
    cfg = LLMConfig(
        provider="codex-cli",
        utility_model="codex",
        synthesis_model="codex",
        codex_reasoning_effort="high",  # global fallback
    )

    utility_config, synthesis_config = cfg.get_provider_configs()

    # utility and synthesis DO inherit global effort
    assert utility_config["reasoning_effort"] == "high"
    assert synthesis_config["reasoning_effort"] == "high"

    # map_hyde and autodoc_cleanup are NOT returned by get_provider_configs(),
    # so the only way they could get the effort is via the runtime build functions.
    # Verify the contract: get_provider_configs() output does NOT carry forward
    # reasoning_effort as a signal for secondary roles to blindly inherit.
    # (The secondary roles strip it when the provider changes; see llm.py and
    # autodoc_cleanup.py.)
    assert "map_hyde_reasoning_effort" not in utility_config
    assert "map_hyde_reasoning_effort" not in synthesis_config
    assert "autodoc_cleanup_reasoning_effort" not in utility_config
    assert "autodoc_cleanup_reasoning_effort" not in synthesis_config


def test_map_hyde_inherits_synthesis_model_but_not_global_codex_effort():
    cfg = LLMConfig(
        provider="openai",
        utility_model="gpt-5-nano",
        synthesis_model="gpt-5",
        codex_reasoning_effort="high",
    )

    role_cfg = cfg.get_provider_config_for_role("map_hyde")

    assert role_cfg["provider"] == "openai"
    assert role_cfg["model"] == "gpt-5"
    assert "reasoning_effort" not in role_cfg


def test_autodoc_cleanup_inherits_synthesis_model_but_not_global_codex_effort():
    cfg = LLMConfig(
        provider="openai",
        utility_model="gpt-5-nano",
        synthesis_model="gpt-5",
        codex_reasoning_effort="high",
    )

    role_cfg = cfg.get_provider_config_for_role("autodoc_cleanup")

    assert role_cfg["provider"] == "openai"
    assert role_cfg["model"] == "gpt-5"
    assert "reasoning_effort" not in role_cfg


def test_non_reasoning_providers_drop_global_effort_from_provider_configs():
    cfg = LLMConfig(
        provider="claude-code-cli",
        codex_reasoning_effort="high",
    )

    utility_config, synthesis_config = cfg.get_provider_configs()

    assert utility_config["provider"] == "claude-code-cli"
    assert synthesis_config["provider"] == "claude-code-cli"
    assert "reasoning_effort" not in utility_config
    assert "reasoning_effort" not in synthesis_config


def test_get_provider_config_for_role_propagates_structured_outputs_to_secondary_role(
) -> None:
    cfg = LLMConfig(
        provider="deepseek",
        api_key="sk-test",
        synthesis_model="deepseek-v4-flash",
        map_hyde_model="deepseek-v4-flash",
        supports_structured_outputs=False,
    )

    role_cfg = cfg.get_provider_config_for_role("map_hyde")

    assert role_cfg["provider"] == "deepseek"
    assert role_cfg["model"] == "deepseek-v4-flash"
    assert role_cfg["supports_structured_outputs"] is False


@pytest.mark.parametrize(
    ("role", "provider", "model"),
    [
        ("map_hyde", "openai", "gpt-5-mini"),
        ("autodoc_cleanup", "openai", "gpt-5-mini"),
    ],
)
def test_role_config_does_not_propagate_structured_outputs_across_provider_switch(
    role: str,
    provider: str,
    model: str,
) -> None:
    cfg = LLMConfig(
        provider="deepseek",
        api_key="sk-test",
        synthesis_model="deepseek-v4-flash",
        supports_structured_outputs=False,
        **{f"{role}_provider": provider, f"{role}_model": model},
    )

    role_cfg = cfg.get_provider_config_for_role(role)

    assert role_cfg["provider"] == provider
    assert role_cfg["model"] == model
    assert "supports_structured_outputs" not in role_cfg


def test_get_provider_config_for_role_utility_propagates_structured_outputs() -> None:
    """``supports_structured_outputs`` must propagate for the utility role."""
    cfg = LLMConfig(
        provider="deepseek",
        api_key="sk-test",
        utility_model="deepseek-v4-flash",
        synthesis_model="deepseek-v4-flash",
        supports_structured_outputs=False,
    )

    role_cfg = cfg.get_provider_config_for_role("utility")

    assert role_cfg["provider"] == "deepseek"
    assert role_cfg["supports_structured_outputs"] is False


def test_get_provider_config_for_role_synthesis_propagates_structured_outputs() -> None:
    """``supports_structured_outputs`` must propagate for the synthesis role."""
    cfg = LLMConfig(
        provider="deepseek",
        api_key="sk-test",
        utility_model="deepseek-v4-flash",
        synthesis_model="deepseek-v4-flash",
        supports_structured_outputs=False,
    )

    role_cfg = cfg.get_provider_config_for_role("synthesis")

    assert role_cfg["provider"] == "deepseek"
    assert role_cfg["supports_structured_outputs"] is False


def test_get_provider_config_for_role_supports_structured_outputs_none_is_omitted(
) -> None:
    """When ``supports_structured_outputs`` is None (default), it must not appear
    in any role config to avoid leaking provider-agnostic defaults."""
    cfg = LLMConfig(
        provider="openai",
        api_key="sk-test",
        utility_model="gpt-5-nano",
        synthesis_model="gpt-5",
    )

    for role in ("utility", "synthesis", "map_hyde", "autodoc_cleanup"):
        role_cfg = cfg.get_provider_config_for_role(role)
        assert "supports_structured_outputs" not in role_cfg, (
            f"{role} should not have supports_structured_outputs when unset"
        )


def test_get_provider_config_for_role_attaches_anthropic_options() -> None:
    cfg = LLMConfig(
        provider="openai",
        utility_model="gpt-5-nano",
        synthesis_provider="anthropic",
        synthesis_model="claude-opus-4-7",
        api_key="sk-test",
        anthropic_thinking_enabled=True,
        anthropic_thinking_mode="adaptive",
        anthropic_thinking_display="summarized",
        anthropic_effort="xhigh",
        anthropic_prompt_caching=True,
        anthropic_cache_ttl="1h",
        anthropic_task_budget_tokens=20000,
    )

    role_cfg = cfg.get_provider_config_for_role("synthesis")

    assert role_cfg["provider"] == "anthropic"
    assert role_cfg["model"] == "claude-opus-4-7"
    assert role_cfg["thinking_enabled"] is True
    assert role_cfg["thinking_mode"] == "adaptive"
    assert role_cfg["thinking_display"] == "summarized"
    assert role_cfg["effort"] == "xhigh"
    assert role_cfg["prompt_caching"] is True
    assert role_cfg["cache_ttl"] == "1h"
    assert role_cfg["task_budget_tokens"] == 20000


def test_get_provider_config_for_role_unknown_role_raises() -> None:
    cfg = LLMConfig(
        provider="openai",
        utility_model="gpt-5-nano",
        synthesis_model="gpt-5",
    )

    with pytest.raises(ValueError, match="Unknown role"):
        cfg.get_provider_config_for_role("nope")


@pytest.mark.parametrize(
    ("provider", "kwargs"),
    [
        ("ollama", {}),
        ("claude-code-cli", {}),
        ("codex-cli", {}),
        (
            "opencode-cli",
            {
                "utility_model": "openai/gpt-5-nano",
                "synthesis_model": "openai/gpt-5",
            },
        ),
    ],
)
def test_no_key_providers_omit_api_key_in_base_role_configs(
    provider: str,
    kwargs: dict[str, str],
) -> None:
    cfg = LLMConfig(
        provider=provider,
        api_key="sk-test",
        **kwargs,
    )

    utility_config, synthesis_config = cfg.get_provider_configs()

    assert utility_config["provider"] == provider
    assert synthesis_config["provider"] == provider
    assert "api_key" not in utility_config
    assert "api_key" not in synthesis_config


@pytest.mark.parametrize(
    ("role", "provider", "model"),
    [
        ("synthesis", "codex-cli", "codex"),
        ("map_hyde", "opencode-cli", "openai/gpt-5-mini"),
        ("autodoc_cleanup", "claude-code-cli", "claude-haiku-4-5"),
    ],
)
def test_get_provider_config_for_role_strips_api_key_for_no_key_provider_switch(
    role: str,
    provider: str,
    model: str,
) -> None:
    role_overrides = (
        {"synthesis_provider": provider, "synthesis_model": model}
        if role == "synthesis"
        else {f"{role}_provider": provider, f"{role}_model": model}
    )
    config_kwargs: dict[str, str] = {
        "provider": "openai",
        "api_key": "sk-test",
        "utility_model": "gpt-5-nano",
        **role_overrides,
    }
    if role != "synthesis":
        config_kwargs["synthesis_model"] = "gpt-5"

    cfg = LLMConfig(**config_kwargs)

    role_cfg = cfg.get_provider_config_for_role(role)

    assert role_cfg["provider"] == provider
    assert role_cfg["model"] == model
    assert "api_key" not in role_cfg





def test_supports_structured_outputs_not_passed_to_other_providers() -> None:
    """LLMManager must not forward supports_structured_outputs broadly."""
    captured_kwargs: list[dict[str, object]] = []

    class FakeAnthropicProvider:
        def __init__(self, **kwargs):  # noqa: ANN001
            captured_kwargs.append(kwargs)
            self.name = "anthropic"
            self.model = kwargs.get("model", "")

        async def complete(self, prompt, **kwargs):  # noqa: ANN001, ANN201
            pass

    from chunkhound.llm_manager import LLMManager

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(LLMManager._providers, "anthropic", FakeAnthropicProvider)

    cfg = LLMConfig(
        provider="anthropic",
        api_key="sk-test",
        supports_structured_outputs=True,
    )
    utility_config, synthesis_config = cfg.get_provider_configs()

    # The config dict should carry supports_structured_outputs (gate removed in #1)
    assert utility_config.get("supports_structured_outputs") is True
    assert synthesis_config.get("supports_structured_outputs") is True

    # But LLMManager must NOT pass it to non-OpenAICompatible providers
    LLMManager(utility_config, synthesis_config)

    for kwargs in captured_kwargs:
        assert "supports_structured_outputs" not in kwargs, (
            f"Expected no supports_structured_outputs in kwargs, got: {kwargs}"
        )

    monkeypatch.undo()
