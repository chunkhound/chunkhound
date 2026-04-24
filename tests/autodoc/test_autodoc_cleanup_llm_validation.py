from chunkhound.api.cli.commands import autodoc_cleanup as autodoc_cleanup
from chunkhound.api.cli.utils.rich_output import RichOutputFormatter
from chunkhound.core.config.config import Config


def test_cleanup_rejects_cleanup_override_missing_required_provider_settings() -> None:
    config = Config(
        llm={
            "provider": "codex-cli",
            "autodoc_cleanup_provider": "anthropic",
        }
    )
    formatter = RichOutputFormatter()

    resolved = autodoc_cleanup._resolve_llm_config_for_cleanup(
        config=config,
        formatter=formatter,
    )

    assert resolved is None


def test_cleanup_rejects_cross_family_override_without_explicit_model() -> None:
    config = Config(
        llm={
            "provider": "openai",
            "api_key": "test-key",
            "synthesis_model": "gpt-5",
            "autodoc_cleanup_provider": "anthropic",
        }
    )
    formatter = RichOutputFormatter()

    resolved = autodoc_cleanup._resolve_llm_config_for_cleanup(
        config=config,
        formatter=formatter,
    )

    assert resolved is None
