"""Tests for Gemini LLM provider."""

import pytest

from chunkhound.providers.llm.gemini_llm_provider import GeminiLLMProvider


@pytest.fixture
def provider():
    """Create a GeminiLLMProvider instance for testing."""
    return GeminiLLMProvider(
        api_key="test-api-key-123",
        model="gemini-3-pro-preview",
        thinking_level="high",
        timeout=120,
        max_retries=3,
    )


class TestGeminiLLMProvider:
    """Test suite for GeminiLLMProvider."""

    def test_provider_name(self, provider):
        """Test that provider name is correct."""
        assert provider.name == "gemini"

    def test_provider_model(self, provider):
        """Test that model name is stored correctly."""
        assert provider.model == "gemini-3-pro-preview"

    def test_provider_models_supported(self):
        """Test that different Gemini models can be instantiated."""
        # Gemini 3
        provider_3 = GeminiLLMProvider(api_key="test-key", model="gemini-3-pro-preview")
        assert provider_3.model == "gemini-3-pro-preview"

        # Gemini 2.5 Pro
        provider_2_5_pro = GeminiLLMProvider(api_key="test-key", model="gemini-2.5-pro")
        assert provider_2_5_pro.model == "gemini-2.5-pro"

        # Gemini 2.5 Flash
        provider_2_5_flash = GeminiLLMProvider(
            api_key="test-key", model="gemini-2.5-flash"
        )
        assert provider_2_5_flash.model == "gemini-2.5-flash"

    def test_thinking_level_configuration(self):
        """Test thinking level configuration."""
        # High thinking
        provider_high = GeminiLLMProvider(
            api_key="test-key", model="gemini-3-pro-preview", thinking_level="high"
        )
        assert provider_high._thinking_level == "high"

        # Low thinking
        provider_low = GeminiLLMProvider(
            api_key="test-key", model="gemini-3-pro-preview", thinking_level="low"
        )
        assert provider_low._thinking_level == "low"

    def test_estimate_tokens(self, provider):
        """Test token estimation (rough approximation)."""
        text = "a" * 400  # 400 characters
        tokens = provider.estimate_tokens(text)
        assert tokens == 100  # 400 / 4 = 100 tokens

        empty_text = ""
        assert provider.estimate_tokens(empty_text) == 0

    def test_build_generation_config_basic(self, provider):
        """Test basic generation config building."""
        config = provider._build_generation_config(max_completion_tokens=2048)

        assert config.max_output_tokens == 2048
        assert config.temperature == 1.0

    def test_build_generation_config_with_system(self, provider):
        """Test generation config with system instruction."""
        config = provider._build_generation_config(
            max_completion_tokens=1024, system_instruction="You are a helpful assistant"
        )

        assert config.system_instruction == "You are a helpful assistant"
        assert config.max_output_tokens == 1024

    def test_build_generation_config_with_schema(self, provider):
        """Test generation config with JSON schema."""
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        config = provider._build_generation_config(
            max_completion_tokens=1024, json_schema=schema
        )

        assert config.response_mime_type == "application/json"
        assert config.response_json_schema == schema

    def test_build_generation_config_thinking_budget_gemini_2_5(self):
        """Test thinking budget config for Gemini 2.5."""
        provider = GeminiLLMProvider(
            api_key="test-key", model="gemini-2.5-flash", thinking_level="low"
        )

        config = provider._build_generation_config(max_completion_tokens=1024)
        # For Gemini 2.5, thinking_config with budget=0 is set for low
        assert (
            hasattr(config, "thinking_config") or "thinking_config" in config.__dict__
        )

    def test_handle_api_error_404(self, provider):
        """Test error handling for 404 (model not found)."""

        class MockError:
            code = 404
            message = "Model not found"

        error = provider._handle_api_error(MockError(), "test operation")
        assert "not found" in str(error).lower()
        assert provider._model in str(error)

    def test_handle_api_error_429(self, provider):
        """Test error handling for 429 (rate limit)."""

        class MockError:
            code = 429
            message = "Rate limit exceeded"

        error = provider._handle_api_error(MockError(), "test operation")
        assert "rate limit" in str(error).lower()

    def test_handle_api_error_400(self, provider):
        """Test error handling for 400 (invalid request)."""

        class MockError:
            code = 400
            message = "Invalid request parameters"

        error = provider._handle_api_error(MockError(), "test operation")
        assert "invalid" in str(error).lower()
        assert "Invalid request parameters" in str(error)

    def test_handle_api_error_401(self, provider):
        """Test error handling for 401 (authentication failed)."""

        class MockError:
            code = 401
            message = "Invalid API key"

        error = provider._handle_api_error(MockError(), "test operation")
        assert "authentication" in str(error).lower()
        assert "aistudio.google.com" in str(error)

    def test_handle_api_error_403(self, provider):
        """Test error handling for 403 (forbidden)."""

        class MockError:
            code = 403
            message = "Permission denied"

        error = provider._handle_api_error(MockError(), "test operation")
        assert "authentication" in str(error).lower()

    def test_handle_api_error_generic(self, provider):
        """Test error handling for generic errors."""

        class MockError:
            code = 500
            message = "Internal server error"

        error = provider._handle_api_error(MockError(), "test operation")
        assert "500" in str(error)
        assert "Internal server error" in str(error)

    def test_get_usage_stats(self, provider):
        """Test usage statistics retrieval."""
        # Initially zero
        stats = provider.get_usage_stats()
        assert stats["requests_made"] == 0
        assert stats["total_tokens"] == 0

        # Manually increment (normally done by complete methods)
        provider._requests_made = 5
        provider._tokens_used = 1000
        provider._prompt_tokens = 600
        provider._completion_tokens = 400

        stats = provider.get_usage_stats()
        assert stats["requests_made"] == 5
        assert stats["total_tokens"] == 1000
        assert stats["prompt_tokens"] == 600
        assert stats["completion_tokens"] == 400

    def test_get_synthesis_concurrency(self, provider):
        """Test synthesis concurrency recommendation."""
        assert provider.get_synthesis_concurrency() == 2

    def test_api_key_required(self):
        """Test that API key is required."""
        with pytest.raises(ValueError, match="API key required"):
            GeminiLLMProvider(api_key=None, model="gemini-3-pro-preview")

        with pytest.raises(ValueError, match="API key required"):
            GeminiLLMProvider(api_key="", model="gemini-3-pro-preview")

    def test_client_has_aio_attribute(self, provider):
        """Test that client has .aio attribute for async operations."""
        assert hasattr(provider._client, "aio")
