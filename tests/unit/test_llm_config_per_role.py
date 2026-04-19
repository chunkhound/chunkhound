import asyncio
import os
from pathlib import Path

import pytest
from pydantic import SecretStr

from chunkhound.core.config.llm_config import LLMConfig
from chunkhound.llm_manager import LLMManager


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
    """Test that the convenience 'model' field sets both utility and synthesis models."""
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


class _DummyProc:
    def __init__(self, rc: int = 0, out: bytes = b"OK", err: bytes = b"") -> None:
        self.returncode = rc
        self._out = out
        self._err = err
        self.stdin = None

    async def communicate(self):  # pragma: no cover - exercised indirectly
        return self._out, self._err

    def kill(self) -> None:  # pragma: no cover - trivial
        return None

    async def wait(self) -> None:  # pragma: no cover - trivial
        return None


@pytest.mark.asyncio
async def test_llm_codex_cli_status_reflects_configured_model_and_effort(monkeypatch, tmp_path: Path):
    """End-to-end check: LLMConfig -> LLMManager -> CodexCLI overlay config."""
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    cfg = LLMConfig(
        provider="codex-cli",
        utility_provider="codex-cli",
        synthesis_provider="codex-cli",
        utility_model="gpt-5.1-codex",
        synthesis_model="gpt-5.1-codex",
        codex_reasoning_effort_utility="low",
        codex_reasoning_effort_synthesis="high",
    )

    utility_config, synthesis_config = cfg.get_provider_configs()

    # Ensure we never touch a real Codex home or binary
    monkeypatch.setenv("CHUNKHOUND_CODEX_STDIN_FIRST", "0")
    monkeypatch.setenv("CHUNKHOUND_CODEX_CONFIG_OVERRIDE", "env")
    monkeypatch.setattr(CodexCLIProvider, "_get_base_codex_home", lambda self: None, raising=True)
    monkeypatch.setattr(CodexCLIProvider, "_codex_available", lambda self: True, raising=True)

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
        return _DummyProc(rc=0, out=status_text.encode("utf-8"), err=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec, raising=True)

    llm_manager = LLMManager(utility_config, synthesis_config)
    provider = llm_manager.get_synthesis_provider()

    response = await provider.complete(prompt="/status")

    assert "MODEL=gpt-5.1-codex" in response.content
    assert "REASONING_EFFORT=high" in response.content


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


def test_map_hyde_provider_with_custom_base_url_does_not_require_api_key():
    """Custom Grok-compatible HyDE endpoints should not require an API key."""
    cfg = LLMConfig(
        provider="openai",
        model="llama3.2",
        map_hyde_provider="grok",
        base_url="http://localhost:11434/v1",
    )
    assert cfg.is_provider_configured() is True
    assert cfg.get_missing_config() == []


def test_autodoc_cleanup_provider_requires_api_key():
    """autodoc_cleanup_provider='anthropic' without api_key is flagged as unconfigured."""
    cfg = LLMConfig(
        provider="openai",
        model="llama3.2",
        autodoc_cleanup_provider="anthropic",
        base_url="http://localhost:11434/v1",
    )
    assert cfg.is_provider_configured() is False
    missing = cfg.get_missing_config()
    assert any("api_key" in item for item in missing)
    assert any("explicit provider-compatible model required" in item for item in missing)


def test_openai_custom_endpoint_without_api_key_is_valid():
    """Custom OpenAI-compatible endpoints should not require an API key."""
    cfg = LLMConfig(
        provider="openai",
        model="llama3.2",
        base_url="http://localhost:11434/v1",
    )

    assert cfg.is_provider_configured() is True
    assert cfg.get_missing_config() == []


def test_openai_custom_endpoint_requires_explicit_model():
    """Custom OpenAI-compatible endpoints must report missing model selection."""
    cfg = LLMConfig(
        provider="openai",
        base_url="http://localhost:11434/v1",
    )

    assert cfg.is_provider_configured() is False
    missing = cfg.get_missing_config()
    assert len(missing) == 1
    assert "explicit model selection required" in missing[0]


def test_openai_custom_endpoint_requires_explicit_model_for_per_role_provider():
    """Per-role custom OpenAI-compatible endpoints must also report missing models."""
    cfg = LLMConfig(
        provider="grok",
        utility_provider="openai",
        base_url="http://localhost:11434/v1",
        api_key=SecretStr("sk-test-key"),
    )

    assert cfg.is_provider_configured() is False
    missing = cfg.get_missing_config()
    assert len(missing) == 1
    assert "explicit model selection required" in missing[0]
    assert "utility" in missing[0]


def test_grok_custom_endpoint_requires_explicit_model():
    """Non-official Grok endpoints must not silently use cloud defaults."""
    cfg = LLMConfig(
        provider="grok",
        base_url="http://localhost:11434/v1",
        api_key=SecretStr("sk-test-key"),
    )

    assert cfg.is_provider_configured() is False
    missing = cfg.get_missing_config()
    assert len(missing) == 1
    assert "explicit model selection required" in missing[0]
    assert "utility" in missing[0]


def test_map_hyde_custom_endpoint_requires_explicit_model() -> None:
    """Custom endpoint HyDE roles must not inherit cloud defaults implicitly."""
    cfg = LLMConfig(
        provider="anthropic",
        map_hyde_provider="openai",
        base_url="http://localhost:11434/v1",
        api_key=SecretStr("sk-test-key"),
    )

    assert cfg.is_provider_configured() is False
    missing = cfg.get_missing_config()
    assert len(missing) == 1
    assert "explicit model selection required" in missing[0]
    assert "map_hyde" in missing[0]


def test_autodoc_cleanup_custom_endpoint_requires_explicit_model() -> None:
    """Cleanup roles on custom endpoints must set an explicit model."""
    cfg = LLMConfig(
        provider="anthropic",
        autodoc_cleanup_provider="openai",
        base_url="http://localhost:11434/v1",
        api_key=SecretStr("sk-test-key"),
    )

    assert cfg.is_provider_configured() is False
    missing = cfg.get_missing_config()
    assert len(missing) == 1
    assert "explicit model selection required" in missing[0]
    assert "autodoc_cleanup" in missing[0]


def test_map_hyde_cross_family_override_requires_explicit_model() -> None:
    """Cross-family HyDE overrides must not silently inherit synthesis models."""
    cfg = LLMConfig(
        provider="openai",
        synthesis_model="gpt-5",
        map_hyde_provider="anthropic",
        api_key=SecretStr("sk-test-key"),
    )

    assert cfg.is_provider_configured() is False
    missing = cfg.get_missing_config()
    assert len(missing) == 1
    assert "explicit provider-compatible model required" in missing[0]
    assert "map_hyde" in missing[0]


def test_autodoc_cleanup_cross_family_override_requires_explicit_model() -> None:
    """Cross-family cleanup overrides must set a compatible explicit model."""
    cfg = LLMConfig(
        provider="openai",
        synthesis_model="gpt-5",
        autodoc_cleanup_provider="anthropic",
        api_key=SecretStr("sk-test-key"),
    )

    assert cfg.is_provider_configured() is False
    missing = cfg.get_missing_config()
    assert len(missing) == 1
    assert "explicit provider-compatible model required" in missing[0]
    assert "autodoc_cleanup" in missing[0]


def test_openai_official_endpoint_without_api_key_is_invalid():
    """Official OpenAI endpoints must still require an API key."""
    cfg = LLMConfig(
        provider="openai",
        model="gpt-5",
        base_url="https://api.openai.com/v1",
    )

    assert cfg.is_provider_configured() is False
    missing = cfg.get_missing_config()
    assert len(missing) == 1
    assert "api_key" in missing[0]


def test_per_role_grok_on_custom_openai_base_url_does_not_require_api_key():
    """A shared custom endpoint should be treated as keyless for Grok-compatible roles."""
    cfg = LLMConfig(
        provider="openai",
        synthesis_provider="grok",
        model="llama3.2",
        synthesis_model="grok-4-1-fast-reasoning",
        base_url="http://localhost:11434/v1",
    )

    assert cfg.is_provider_configured() is True
    assert cfg.get_missing_config() == []


def test_grok_custom_endpoint_without_api_key_is_valid():
    """Custom Grok-compatible endpoints should not require an API key."""
    cfg = LLMConfig(
        provider="grok",
        model="grok-local",
        base_url="http://localhost:11434/v1",
    )

    assert cfg.is_provider_configured() is True
    assert cfg.get_missing_config() == []


def test_base_url_applies_to_anthropic_proxy_configs():
    """Anthropic custom endpoints must keep receiving the shared base_url."""
    cfg = LLMConfig(
        provider="openai",
        utility_provider="openai",
        synthesis_provider="anthropic",
        utility_model="llama3.2",
        synthesis_model="claude-sonnet-4-5-20250929",
        base_url="http://localhost:11434/v1",
        api_key=SecretStr("sk-test-key"),
    )

    utility_config, synthesis_config = cfg.get_provider_configs()

    assert utility_config["base_url"] == "http://localhost:11434/v1"
    assert synthesis_config["base_url"] == "http://localhost:11434/v1"


def test_ssl_verify_only_flows_when_base_url_is_set():
    cfg = LLMConfig(
        provider="openai",
        utility_model="gpt-5-nano",
        synthesis_model="gpt-5",
        ssl_verify=False,
    )

    utility_config, synthesis_config = cfg.get_provider_configs()

    assert "ssl_verify" not in utility_config
    assert "ssl_verify" not in synthesis_config

    cfg_with_base = LLMConfig(
        provider="openai",
        utility_model="gpt-5-nano",
        synthesis_model="gpt-5",
        base_url="http://localhost:11434/v1",
        ssl_verify=False,
    )

    utility_with_base, synthesis_with_base = cfg_with_base.get_provider_configs()

    assert utility_with_base["ssl_verify"] is False
    assert synthesis_with_base["ssl_verify"] is False


def test_ollama_provider_rejected():
    """Test that 'ollama' raises with a migration hint."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="ollama.*removed.*base_url"):
        LLMConfig(provider="ollama")


def test_ollama_rejected_on_per_role_provider():
    """Test that 'ollama' on per-role fields raises with a migration hint."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="ollama.*removed.*base_url"):
        LLMConfig(utility_provider="ollama")

    with pytest.raises(ValidationError, match="ollama.*removed.*base_url"):
        LLMConfig(synthesis_provider="ollama")


def test_opencode_cli_no_key_required():
    """Test that opencode-cli is configured without an API key."""
    cfg = LLMConfig(
        provider="opencode-cli",
        utility_provider="opencode-cli",
        synthesis_provider="opencode-cli",
    )
    assert cfg.is_provider_configured() is True


def test_opencode_cli_smart_model_defaults():
    """Test opencode-cli smart model defaults."""
    cfg = LLMConfig(provider="opencode-cli")
    assert cfg.get_default_models() == ("opencode/grok-code", "opencode/grok-code")


def test_utility_provider_uses_its_own_default_model():
    """Per-role provider overrides should use that provider's defaults."""
    cfg = LLMConfig(
        provider="openai",
        utility_provider="opencode-cli",
        synthesis_model="gpt-5",
    )

    utility_config, synthesis_config = cfg.get_provider_configs()

    assert utility_config["provider"] == "opencode-cli"
    assert utility_config["model"] == "opencode/grok-code"
    assert synthesis_config["provider"] == "openai"
    assert synthesis_config["model"] == "gpt-5"


def test_llm_config_default_timeout():
    """LLMConfig default timeout is 120 seconds."""
    cfg = LLMConfig()
    assert cfg.timeout == 120
