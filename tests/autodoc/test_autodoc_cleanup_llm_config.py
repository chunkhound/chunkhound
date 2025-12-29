from chunkhound.api.cli.commands import autodoc as autodoc_command
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

    _, synthesis = autodoc_command._build_cleanup_provider_configs(llm_config)

    assert synthesis["provider"] == "codex-cli"
    assert synthesis["model"] == "gpt-5.1-codex"
    assert synthesis["reasoning_effort"] == "medium"
