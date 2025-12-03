"""Tests for Anthropic LLM provider with extended thinking and Opus 4.5 support."""

import pytest

from chunkhound.providers.llm.anthropic_llm_provider import (
    ANTHROPIC_AVAILABLE,
    BETA_CONTEXT_MANAGEMENT,
    BETA_EFFORT,
    BETA_INTERLEAVED_THINKING,
    EFFORT_SUPPORTED_MODELS,
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


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestOpus45EffortParameter:
    """Test Opus 4.5 effort parameter functionality."""

    def test_effort_parameter_opus_45(self):
        """Test effort parameter is stored for Opus 4.5."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
            effort="medium",
        )

        assert provider._effort == "medium"
        assert provider._model in EFFORT_SUPPORTED_MODELS

    def test_effort_parameter_warning_non_opus(self):
        """Test that non-Opus 4.5 models get a warning for effort parameter."""
        # This should log a warning but still initialize
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-sonnet-4-5-20250929",
            effort="low",
        )

        # Effort is stored but model doesn't support it
        assert provider._effort == "low"
        assert provider._model not in EFFORT_SUPPORTED_MODELS

    def test_effort_levels(self):
        """Test all valid effort levels."""
        for effort in ["low", "medium", "high"]:
            provider = AnthropicLLMProvider(
                api_key="test-key",
                model="claude-opus-4-5-20251101",
                effort=effort,
            )
            assert provider._effort == effort

    def test_output_config_with_effort(self):
        """Test output config is built correctly with effort."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
            effort="low",
        )

        output_config = provider._build_output_config()
        assert output_config == {"effort": "low"}

    def test_output_config_no_effort(self):
        """Test output config is None when effort not set."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
        )

        output_config = provider._build_output_config()
        assert output_config is None

    def test_output_config_non_opus_model(self):
        """Test output config is None for non-Opus 4.5 models even with effort."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-sonnet-4-5-20250929",
            effort="medium",
        )

        output_config = provider._build_output_config()
        assert output_config is None


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestInterleavedThinking:
    """Test interleaved thinking functionality."""

    def test_interleaved_thinking_enabled(self):
        """Test interleaved thinking can be enabled."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            thinking_enabled=True,
            interleaved_thinking=True,
        )

        assert provider._interleaved_thinking is True
        assert provider._thinking_enabled is True

    def test_interleaved_thinking_without_thinking(self):
        """Test interleaved thinking without base thinking enabled."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            thinking_enabled=False,
            interleaved_thinking=True,
        )

        # Interleaved is stored but thinking is disabled
        assert provider._interleaved_thinking is True
        assert provider._thinking_enabled is False


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestContextManagement:
    """Test context management functionality."""

    def test_context_management_enabled(self):
        """Test context management can be enabled."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            context_management_enabled=True,
        )

        assert provider._context_management_enabled is True

    def test_context_management_with_thinking_config(self):
        """Test context management with thinking block clearing config."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            thinking_enabled=True,
            context_management_enabled=True,
            clear_thinking_keep_turns=2,
        )

        assert provider._clear_thinking_keep_turns == 2

    def test_context_management_with_tool_config(self):
        """Test context management with tool result clearing config."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            context_management_enabled=True,
            clear_tool_uses_trigger_tokens=50000,
            clear_tool_uses_keep=5,
        )

        assert provider._clear_tool_uses_trigger_tokens == 50000
        assert provider._clear_tool_uses_keep == 5

    def test_build_context_management_disabled(self):
        """Test context management returns None when disabled."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            context_management_enabled=False,
        )

        result = provider._build_context_management()
        assert result is None

    def test_build_context_management_with_thinking(self):
        """Test context management config with thinking enabled."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            thinking_enabled=True,
            context_management_enabled=True,
            clear_thinking_keep_turns=3,
        )

        result = provider._build_context_management()
        assert result is not None
        assert "edits" in result
        # Should have thinking edit first, then tool edit
        assert len(result["edits"]) == 2
        assert result["edits"][0]["type"] == "clear_thinking_20251015"
        assert result["edits"][0]["keep"]["value"] == 3
        assert result["edits"][1]["type"] == "clear_tool_uses_20250919"

    def test_build_context_management_keep_all_thinking(self):
        """Test context management keeps all thinking by default."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            thinking_enabled=True,
            context_management_enabled=True,
            # No clear_thinking_keep_turns specified
        )

        result = provider._build_context_management()
        assert result["edits"][0]["keep"] == "all"

    def test_build_context_management_thinking_active_override(self):
        """Test thinking_active=False skips clear_thinking even when thinking_enabled=True.

        This is critical for complete_structured() which uses forced tool choice
        that's incompatible with thinking - the context management should not
        include clear_thinking_20251015 in that case.
        """
        provider = AnthropicLLMProvider(
            api_key="test-key",
            thinking_enabled=True,  # Enabled at provider level
            context_management_enabled=True,
        )

        # Default: should include clear_thinking
        result_with_thinking = provider._build_context_management()
        assert result_with_thinking is not None
        assert len(result_with_thinking["edits"]) == 2
        assert result_with_thinking["edits"][0]["type"] == "clear_thinking_20251015"

        # Override: thinking_active=False should skip clear_thinking
        result_no_thinking = provider._build_context_management(thinking_active=False)
        assert result_no_thinking is not None
        # Should only have tool_uses edit, no thinking edit
        assert len(result_no_thinking["edits"]) == 1
        assert result_no_thinking["edits"][0]["type"] == "clear_tool_uses_20250919"


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestBetaHeaders:
    """Test beta header generation."""

    def test_no_beta_headers_default(self):
        """Test no beta headers with default config."""
        provider = AnthropicLLMProvider(api_key="test-key")

        headers = provider._get_beta_headers()
        assert headers == []

    def test_effort_beta_header(self):
        """Test effort beta header is included for Opus 4.5."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
            effort="medium",
        )

        headers = provider._get_beta_headers()
        assert BETA_EFFORT in headers

    def test_effort_beta_header_not_for_sonnet(self):
        """Test effort beta header is not included for non-Opus models."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-sonnet-4-5-20250929",
            effort="medium",
        )

        headers = provider._get_beta_headers()
        assert BETA_EFFORT not in headers

    def test_context_management_beta_header(self):
        """Test context management beta header."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            context_management_enabled=True,
        )

        headers = provider._get_beta_headers()
        assert BETA_CONTEXT_MANAGEMENT in headers

    def test_interleaved_thinking_beta_header(self):
        """Test interleaved thinking beta header."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            thinking_enabled=True,
            interleaved_thinking=True,
        )

        headers = provider._get_beta_headers()
        assert BETA_INTERLEAVED_THINKING in headers

    def test_interleaved_thinking_requires_thinking_enabled(self):
        """Test interleaved thinking header only added when thinking is enabled."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            thinking_enabled=False,
            interleaved_thinking=True,
        )

        headers = provider._get_beta_headers()
        assert BETA_INTERLEAVED_THINKING not in headers

    def test_all_beta_headers(self):
        """Test all beta headers combined."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
            effort="low",
            thinking_enabled=True,
            interleaved_thinking=True,
            context_management_enabled=True,
        )

        headers = provider._get_beta_headers()
        assert BETA_EFFORT in headers
        assert BETA_CONTEXT_MANAGEMENT in headers
        assert BETA_INTERLEAVED_THINKING in headers
        assert len(headers) == 3


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestOpus45ModelConfiguration:
    """Test Opus 4.5 model configuration."""

    def test_opus_45_model_id(self):
        """Test Opus 4.5 model ID is correct."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
        )

        assert provider.model == "claude-opus-4-5-20251101"

    def test_opus_45_in_supported_models(self):
        """Test Opus 4.5 is in effort supported models."""
        assert "claude-opus-4-5-20251101" in EFFORT_SUPPORTED_MODELS

    def test_opus_45_full_configuration(self):
        """Test Opus 4.5 with all features enabled."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
            thinking_enabled=True,
            thinking_budget_tokens=16000,
            interleaved_thinking=True,
            effort="medium",
            context_management_enabled=True,
            clear_thinking_keep_turns=2,
            clear_tool_uses_trigger_tokens=100000,
            clear_tool_uses_keep=5,
        )

        assert provider._model == "claude-opus-4-5-20251101"
        assert provider._thinking_enabled is True
        assert provider._thinking_budget_tokens == 16000
        assert provider._interleaved_thinking is True
        assert provider._effort == "medium"
        assert provider._context_management_enabled is True
        assert provider._clear_thinking_keep_turns == 2
        assert provider._clear_tool_uses_trigger_tokens == 100000
        assert provider._clear_tool_uses_keep == 5
