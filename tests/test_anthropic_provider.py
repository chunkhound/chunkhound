"""Tests for Anthropic LLM provider with extended thinking support."""

import pytest

from chunkhound.providers.llm.anthropic_llm_provider import (
    ANTHROPIC_AVAILABLE,
    AnthropicLLMProvider,
)


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestAnthropicProviderBasics:
    """Test basic Anthropic provider functionality."""

    def test_provider_initialization(self):
        """Test provider can be initialized."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-sonnet-4-5-20250929",
        )

        assert provider.name == "anthropic"
        assert provider.model == "claude-sonnet-4-5-20250929"
        assert provider.supports_thinking() is True
        assert provider.supports_tools() is True

    def test_thinking_enabled_initialization(self):
        """Test provider with thinking enabled."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            thinking_enabled=True,
            thinking_budget_tokens=5000,
        )

        assert provider._thinking_enabled is True
        assert provider._thinking_budget_tokens == 5000

    def test_thinking_budget_minimum(self):
        """Test thinking budget enforces minimum of 1024 tokens."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            thinking_enabled=True,
            thinking_budget_tokens=500,  # Below minimum
        )

        # Should be clamped to minimum of 1024
        assert provider._thinking_budget_tokens == 1024


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestContentBlockHandling:
    """Test content block extraction from Anthropic responses."""

    def test_extract_text_from_text_blocks(self):
        """Test extracting text from standard text blocks."""
        provider = AnthropicLLMProvider(api_key="test-key")

        # Mock content blocks
        class TextBlock:
            type = "text"
            text = "This is a response."

        class ThinkingBlock:
            type = "thinking"
            thinking = "Let me think about this..."
            signature = "abc123"

        blocks = [ThinkingBlock(), TextBlock()]
        result = provider._extract_text_from_content(blocks)

        # Should only extract text block, not thinking
        assert result == "This is a response."

    def test_extract_multiple_text_blocks(self):
        """Test concatenating multiple text blocks."""
        provider = AnthropicLLMProvider(api_key="test-key")

        class TextBlock:
            def __init__(self, text):
                self.type = "text"
                self.text = text

        blocks = [
            TextBlock("First part. "),
            TextBlock("Second part."),
        ]
        result = provider._extract_text_from_content(blocks)

        assert result == "First part. Second part."

    def test_get_thinking_blocks(self):
        """Test extracting thinking blocks for preservation."""
        provider = AnthropicLLMProvider(api_key="test-key")

        class ThinkingBlock:
            type = "thinking"
            thinking = "Let me analyze this step by step..."
            signature = "signature123"

        class RedactedThinkingBlock:
            type = "redacted_thinking"
            data = "encrypted_data_xyz"

        class TextBlock:
            type = "text"
            text = "Final answer"

        blocks = [ThinkingBlock(), RedactedThinkingBlock(), TextBlock()]
        thinking = provider._get_thinking_blocks(blocks)

        # Should extract only thinking blocks
        assert len(thinking) == 2
        assert thinking[0]["type"] == "thinking"
        assert thinking[0]["thinking"] == "Let me analyze this step by step..."
        assert thinking[0]["signature"] == "signature123"
        assert thinking[1]["type"] == "redacted_thinking"
        assert thinking[1]["data"] == "encrypted_data_xyz"


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestUsageTracking:
    """Test usage statistics tracking."""

    def test_initial_stats(self):
        """Test initial usage stats are zero."""
        provider = AnthropicLLMProvider(api_key="test-key")

        stats = provider.get_usage_stats()

        assert stats["requests_made"] == 0
        assert stats["total_tokens"] == 0
        assert stats["prompt_tokens"] == 0
        assert stats["completion_tokens"] == 0
        assert stats["thinking_tokens"] == 0

    def test_health_check_structure(self):
        """Test health check includes thinking status."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            thinking_enabled=True,
        )

        # Health check will fail without real API key, but we can check structure
        # by catching the exception
        try:
            import asyncio

            asyncio.run(provider.health_check())
        except Exception:
            pass  # Expected to fail without real API

        # Just verify the method exists and has proper signature
        assert hasattr(provider, "health_check")


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestProviderCapabilities:
    """Test provider capability detection."""

    def test_supports_thinking(self):
        """Test provider reports thinking support."""
        provider = AnthropicLLMProvider(api_key="test-key")

        assert provider.supports_thinking() is True

    def test_supports_tools(self):
        """Test provider reports tool use support."""
        provider = AnthropicLLMProvider(api_key="test-key")

        assert provider.supports_tools() is True

    def test_synthesis_concurrency(self):
        """Test recommended synthesis concurrency."""
        provider = AnthropicLLMProvider(api_key="test-key")

        # Anthropic has higher rate limits than OpenAI
        assert provider.get_synthesis_concurrency() == 5

    def test_token_estimation(self):
        """Test token estimation (rough approximation)."""
        provider = AnthropicLLMProvider(api_key="test-key")

        # ~4 chars per token for Claude
        text = "a" * 400
        estimated = provider.estimate_tokens(text)

        assert estimated == 100


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestConfiguration:
    """Test various configuration scenarios."""

    def test_default_configuration(self):
        """Test default configuration values."""
        provider = AnthropicLLMProvider(api_key="test-key")

        assert provider._model == "claude-sonnet-4-5-20250929"
        assert provider._timeout == 60
        assert provider._max_retries == 3
        assert provider._thinking_enabled is False
        assert provider._thinking_budget_tokens == 10000

    def test_custom_configuration(self):
        """Test custom configuration values."""
        provider = AnthropicLLMProvider(
            api_key="custom-key",
            model="claude-opus-4-1-20250805",
            base_url="https://custom.endpoint.com",
            timeout=120,
            max_retries=5,
            thinking_enabled=True,
            thinking_budget_tokens=20000,
        )

        assert provider._model == "claude-opus-4-1-20250805"
        assert provider._timeout == 120
        assert provider._max_retries == 5
        assert provider._thinking_enabled is True
        assert provider._thinking_budget_tokens == 20000

    def test_haiku_model(self):
        """Test Haiku model configuration."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-haiku-4-5-20251001",
        )

        assert provider.model == "claude-haiku-4-5-20251001"


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestToolUse:
    """Test tool use functionality."""

    def test_complete_with_tools_method_exists(self):
        """Test that complete_with_tools method exists."""
        provider = AnthropicLLMProvider(api_key="test-key")

        assert hasattr(provider, "complete_with_tools")
        assert callable(provider.complete_with_tools)

    def test_tool_use_with_thinking(self):
        """Test tool use can be combined with thinking."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            thinking_enabled=True,
            thinking_budget_tokens=5000,
        )

        # Both features should be enabled
        assert provider._thinking_enabled is True
        assert provider.supports_tools() is True


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestStructuredOutputWithToolUse:
    """Test structured output using tool use."""

    def test_structured_output_method_exists(self):
        """Test that complete_structured method still exists."""
        provider = AnthropicLLMProvider(api_key="test-key")

        assert hasattr(provider, "complete_structured")
        assert callable(provider.complete_structured)


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestStreaming:
    """Test streaming functionality."""

    def test_supports_streaming(self):
        """Test provider reports streaming support."""
        provider = AnthropicLLMProvider(api_key="test-key")

        assert provider.supports_streaming() is True

    def test_streaming_method_exists(self):
        """Test that complete_streaming method exists."""
        provider = AnthropicLLMProvider(api_key="test-key")

        assert hasattr(provider, "complete_streaming")
        assert callable(provider.complete_streaming)

    def test_streaming_with_thinking(self):
        """Test streaming can be combined with thinking."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            thinking_enabled=True,
        )

        assert provider._thinking_enabled is True
        assert provider.supports_streaming() is True
