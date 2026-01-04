"""Test configuration validation for OpenAI-compatible endpoints like Ollama.

This test reproduces the bug where ChunkHound incorrectly requires an OpenAI API key
for custom endpoints that don't need authentication.
"""

import pytest

from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory


class TestOllamaConfigValidation:
    """Test configuration validation for Ollama and other OpenAI-compatible endpoints."""

    def test_ollama_config_without_api_key_should_be_valid(self):
        """Test that Ollama config without API key should be considered valid.

        This test reproduces the bug reported by the user:
        - provider: "openai"
        - base_url: "http://localhost:11434"
        - model: "nomic-embed-text"
        - No API key provided

        Currently fails with: "OpenAI API key required"
        Should pass: Custom endpoints don't require API keys
        """
        config = EmbeddingConfig(
            provider="openai",
            base_url="http://localhost:11434",
            model="nomic-embed-text",
        )

        # This should return True but currently returns False (the bug)
        assert config.is_provider_configured(), (
            "Ollama config should be valid without API key"
        )

        # Missing config should be empty
        missing = config.get_missing_config()
        assert missing == [], (
            f"No config should be missing for Ollama, but got: {missing}"
        )

    def test_official_openai_config_without_api_key_should_be_invalid(self):
        """Test that official OpenAI config without API key should be invalid."""
        # Default OpenAI endpoint (no base_url)
        config = EmbeddingConfig(provider="openai", model="text-embedding-3-small")

        assert not config.is_provider_configured(), (
            "Official OpenAI should require API key"
        )

        missing = config.get_missing_config()
        assert "api_key" in str(missing), (
            "Should report missing API key for official OpenAI"
        )

    def test_official_openai_explicit_url_without_api_key_should_be_invalid(self):
        """Test that explicit official OpenAI URL without API key should be invalid."""
        config = EmbeddingConfig(
            provider="openai",
            base_url="https://api.openai.com/v1",
            model="text-embedding-3-small",
        )

        assert not config.is_provider_configured(), (
            "Official OpenAI URL should require API key"
        )

        missing = config.get_missing_config()
        assert "api_key" in str(missing), (
            "Should report missing API key for official OpenAI URL"
        )

    def test_custom_endpoint_with_api_key_should_be_valid(self):
        """Test that custom endpoint with API key should be valid."""
        config = EmbeddingConfig(
            provider="openai",
            base_url="http://localhost:11434",
            model="nomic-embed-text",
            api_key="custom-key",
        )

        assert config.is_provider_configured(), (
            "Custom endpoint with API key should be valid"
        )

        missing = config.get_missing_config()
        assert missing == [], "No config should be missing when all provided"

    def test_factory_validation_with_ollama_config(self):
        """Test that factory validation works with Ollama config (integration test)."""
        config = EmbeddingConfig(
            provider="openai",
            base_url="http://localhost:11434",
            model="nomic-embed-text",
        )

        # This should not raise an exception for custom endpoints
        # Currently raises: "Incomplete configuration for openai provider. Missing: api_key"
        try:
            # We don't actually create the provider (would need OpenAI lib)
            # Just test the validation logic
            if not config.is_provider_configured():
                missing = config.get_missing_config()
                raise ValueError(
                    f"Incomplete configuration for {config.provider} provider. "
                    f"Missing: {', '.join(missing)}"
                )
        except ValueError as e:
            pytest.fail(f"Factory validation should pass for Ollama config: {e}")

    def test_various_custom_endpoints_without_api_key(self):
        """Test various custom endpoint formats should work without API key."""
        custom_endpoints = [
            "http://localhost:11434",
            "http://127.0.0.1:11434",
            "http://my-server:8080",
            "https://my-custom-api.com/v1",
            "http://192.168.1.100:1234",
        ]

        for base_url in custom_endpoints:
            config = EmbeddingConfig(
                provider="openai", base_url=base_url, model="test-model"
            )

            assert config.is_provider_configured(), (
                f"Custom endpoint {base_url} should be valid without API key"
            )

            missing = config.get_missing_config()
            assert missing == [], (
                f"No config should be missing for {base_url}, but got: {missing}"
            )


class TestOllamaProviderIntegration:
    """Integration tests that actually create providers to catch provider-layer bugs."""

    def test_factory_can_create_ollama_provider_without_api_key(self):
        """Test that factory can create provider for Ollama config without API key.

        This test reproduces the actual bug: the config layer works correctly,
        but the provider creation fails because is_available() requires API key.
        """
        config = EmbeddingConfig(
            provider="openai",
            base_url="http://localhost:11434/v1",
            model="nomic-embed-text",
            # No api_key - this should work for custom endpoints
        )

        # Config validation should pass (this already works)
        assert config.is_provider_configured(), "Config should be valid"

        # Provider creation should NOT raise an exception (this currently fails)
        try:
            provider = EmbeddingProviderFactory.create_provider(config)
            assert provider is not None, "Provider should be created successfully"
        except ValueError as e:
            pytest.fail(f"Factory should create provider for Ollama config: {e}")

    def test_openai_provider_is_available_for_custom_endpoints(self):
        """Test that OpenAIEmbeddingProvider.is_available() works for custom endpoints.

        This test directly tests the buggy method: is_available() always requires
        API key regardless of endpoint type.
        """
        # Import here to avoid dependency issues if OpenAI lib not available
        try:
            from chunkhound.providers.embeddings.openai_provider import (
                OpenAIEmbeddingProvider,
            )
        except ImportError:
            pytest.skip(
                "OpenAI provider not available - install with: pip install openai"
            )

        provider = OpenAIEmbeddingProvider(
            base_url="http://localhost:11434/v1",
            model="nomic-embed-text",
            # No api_key provided
        )

        # This should return True but currently returns False (the bug)
        assert provider.is_available(), (
            "Custom endpoint should be available without API key"
        )

    @pytest.mark.asyncio
    async def test_openai_provider_client_init_custom_endpoint_without_api_key(self):
        """Test that OpenAI provider can initialize client for custom endpoints without API key.

        This test verifies the client initialization process works correctly
        for custom endpoints even when no API key is provided.
        """
        # Import here to avoid dependency issues if OpenAI lib not available
        try:
            from chunkhound.providers.embeddings.openai_provider import (
                OpenAIEmbeddingProvider,
            )
        except ImportError:
            pytest.skip(
                "OpenAI provider not available - install with: pip install openai"
            )

        provider = OpenAIEmbeddingProvider(
            base_url="http://localhost:11434/v1",
            model="nomic-embed-text",
            # No api_key provided
        )

        # This should not raise an exception
        try:
            await provider._ensure_client()
            assert provider._client is not None, "Client should be initialized"
        except ValueError as e:
            if "API key" in str(e):
                pytest.fail(f"Custom endpoint should not require API key: {e}")
            else:
                # Re-raise if it's a different error (e.g., network issues)
                raise
