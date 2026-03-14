from chunkhound.llm_manager import LLMManager


def test_llm_manager_registry_includes_codex_cli():
    # Red test: registry should include codex-cli provider key
    assert "codex-cli" in LLMManager._providers


def test_create_gemini_provider_with_base_url_in_config():
    """Regression: LLMManager must not pass base_url to Gemini (it doesn't accept it)."""
    manager = LLMManager.__new__(LLMManager)
    provider = manager._create_provider(
        {
            "provider": "gemini",
            "api_key": "test-key",
            "model": "gemini-3-pro-preview",
            "base_url": "https://custom.endpoint/v1",
        }
    )
    assert provider.name == "gemini"

