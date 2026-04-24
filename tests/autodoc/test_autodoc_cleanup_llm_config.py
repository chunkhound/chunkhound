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
        autodoc_cleanup_model="gpt-5.1-codex",
        autodoc_cleanup_reasoning_effort="medium",
    )

    _, synthesis = autodoc_cleanup._build_cleanup_provider_configs(llm_config)

    assert synthesis["provider"] == "codex-cli"
    assert synthesis["model"] == "gpt-5.1-codex"
    assert synthesis["reasoning_effort"] == "medium"


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
        }
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
        }
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
