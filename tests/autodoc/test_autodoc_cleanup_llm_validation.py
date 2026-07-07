import pytest

from chunkhound.core.config.config import Config


def test_cleanup_rejects_cleanup_override_missing_required_provider_settings() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "autodoc_cleanup provider override "
            "requires an explicit autodoc_cleanup_model"
        ),
    ):
        Config(
            llm={
                "provider": "codex-cli",
                "autodoc_cleanup_provider": "anthropic",
            }
        )


def test_cleanup_rejects_cross_family_override_without_explicit_model() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "autodoc_cleanup provider override "
            "requires an explicit autodoc_cleanup_model"
        ),
    ):
        Config(
            llm={
                "provider": "openai",
                "api_key": "test-key",
                "synthesis_model": "gpt-5",
                "autodoc_cleanup_provider": "anthropic",
            }
        )
