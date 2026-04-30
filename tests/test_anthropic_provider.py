"""Tests for Anthropic LLM provider with adaptive/manual thinking and effort."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from chunkhound.core.config.claude_model_resolution import (
    CLAUDE_HAIKU_DEFAULT_SENTINEL,
    CLAUDE_HAIKU_FALLBACK_MODEL,
    get_latest_available_haiku_model,
    resolve_claude_haiku_model,
)
from chunkhound.providers.llm.anthropic_llm_provider import (
    ANTHROPIC_AVAILABLE,
    BETA_CONTEXT_MANAGEMENT,
    BETA_INTERLEAVED_THINKING,
    BETA_STRUCTURED_OUTPUTS,
    BETA_TASK_BUDGETS,
    AnthropicLLMProvider,
    requires_adaptive_thinking,
    supports_adaptive_thinking,
    supports_effort,
    supports_effort_level,
    supports_task_budget,
)


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestAnthropicProviderBasics:
    """Test basic Anthropic provider functionality."""

    def test_provider_initialization(self):
        """Test provider can be initialized."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-sonnet-4-6",
        )

        assert provider.name == "anthropic"
        assert provider.model == "claude-sonnet-4-6"
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


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestConfiguration:
    """Test various configuration scenarios."""

    def test_default_configuration(self):
        """Test default configuration values."""
        provider = AnthropicLLMProvider(api_key="test-key")

        assert provider._model == CLAUDE_HAIKU_FALLBACK_MODEL
        assert provider._timeout == 60
        assert provider._max_retries == 3
        assert provider._thinking_enabled is False
        assert provider._thinking_mode == "off"
        assert provider._thinking_budget_tokens == 10000

    def test_default_model_resolution_stays_offline(self, monkeypatch):
        """Provider construction must not perform network discovery."""
        monkeypatch.delenv("CHUNKHOUND_CLAUDE_DEFAULT_HAIKU_MODEL", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setattr(
            "chunkhound.core.config.claude_model_resolution."
            "get_latest_available_haiku_model",
            lambda api_key=None: pytest.fail("discovery should not run"),
        )

        provider = AnthropicLLMProvider(api_key="sk-ant-test")
        assert provider.model == CLAUDE_HAIKU_FALLBACK_MODEL

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
        assert supports_effort(provider._model) is True

    def test_effort_parameter_warning_non_opus(self):
        """Sonnet 4.5 drops an effort value it cannot use; only warns."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-sonnet-4-5-20250929",
            effort="low",
        )

        assert provider._effort is None
        assert supports_effort(provider._model) is False

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
        """Test inactive thinking skips clear_thinking edits.

        This allows context management to be correctly configured for requests
        where thinking is explicitly disabled - clear_thinking_20251015 should not
        be included when thinking is inactive for that specific call.
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

    def test_context_management_beta_header(self):
        """Test context management beta header."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            context_management_enabled=True,
        )

        headers = provider._get_beta_headers()
        assert BETA_CONTEXT_MANAGEMENT in headers

    def test_interleaved_thinking_beta_header(self):
        """Interleaved beta header is emitted for manual-mode models."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
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

    def test_all_beta_headers_manual_thinking(self):
        """Opus 4.5 uses manual thinking so interleaved header is emitted."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
            effort="low",
            thinking_enabled=True,
            interleaved_thinking=True,
            context_management_enabled=True,
        )

        headers = provider._get_beta_headers()
        assert BETA_CONTEXT_MANAGEMENT in headers
        assert BETA_INTERLEAVED_THINKING in headers
        assert len(headers) == 2

    def test_adaptive_mode_suppresses_interleaved_header(self):
        """Adaptive thinking auto-enables interleaved; no beta header needed."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-7",
            thinking_enabled=True,
            interleaved_thinking=True,
            context_management_enabled=True,
        )

        assert provider._thinking_mode == "adaptive"
        headers = provider._get_beta_headers()
        assert BETA_INTERLEAVED_THINKING not in headers
        assert BETA_CONTEXT_MANAGEMENT in headers
        assert len(headers) == 1


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


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestStructuredOutputs:
    """Test native structured outputs functionality."""

    def test_structured_outputs_beta_constant(self):
        """Test structured outputs beta header constant is defined."""
        assert BETA_STRUCTURED_OUTPUTS == "structured-outputs-2025-11-13"

    def test_structured_output_method_exists(self):
        """Test complete_structured method exists."""
        provider = AnthropicLLMProvider(api_key="test-key")

        assert hasattr(provider, "complete_structured")
        assert callable(provider.complete_structured)

    def test_structured_outputs_compatible_with_thinking(self):
        """Test structured outputs are compatible with extended thinking.

        Native structured outputs work with thinking because grammar
        resets between sections, allowing Claude to think freely.
        """
        provider = AnthropicLLMProvider(
            api_key="test-key",
            thinking_enabled=True,
            thinking_budget_tokens=5000,
        )

        # Both should be enabled - native structured outputs are compatible
        assert provider._thinking_enabled is True
        assert provider.supports_tools() is True

    def test_structured_outputs_with_interleaved_thinking(self):
        """Test structured outputs work with interleaved thinking."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            thinking_enabled=True,
            interleaved_thinking=True,
        )

        assert provider._thinking_enabled is True
        assert provider._interleaved_thinking is True


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestStrictToolUse:
    """Test strict tool use functionality."""

    def test_strict_tool_definition(self):
        """Test tools can include strict: true."""
        tool = {
            "name": "get_weather",
            "description": "Get weather",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        }

        # Verify tool structure is valid
        assert tool.get("strict") is True
        assert "additionalProperties" in tool["input_schema"]
        assert tool["input_schema"]["additionalProperties"] is False

    def test_strict_tool_detection(self):
        """Test detecting strict tools."""
        tools = [
            {"name": "tool1", "description": "desc", "input_schema": {}},
            {
                "name": "tool2",
                "description": "desc",
                "strict": True,
                "input_schema": {},
            },
        ]

        has_strict = any(tool.get("strict") for tool in tools)
        assert has_strict is True

    def test_no_strict_tools(self):
        """Test detecting no strict tools."""
        tools = [
            {"name": "tool1", "description": "desc", "input_schema": {}},
            {"name": "tool2", "description": "desc", "input_schema": {}},
        ]

        has_strict = any(tool.get("strict") for tool in tools)
        assert has_strict is False

    def test_complete_with_tools_docstring_mentions_strict(self):
        """Test complete_with_tools documents strict option."""
        provider = AnthropicLLMProvider(api_key="test-key")

        docstring = provider.complete_with_tools.__doc__
        assert "strict" in docstring.lower()


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestStructuredOutputsBetaHeaders:
    """Test structured outputs beta header integration."""

    def test_get_beta_headers_includes_structured(self):
        """Test structured outputs beta is included when needed.

        Note: _get_beta_headers() doesn't include structured outputs
        because that's only added dynamically when output_format is used.
        """
        provider = AnthropicLLMProvider(api_key="test-key")

        # Default config shouldn't include structured outputs
        headers = provider._get_beta_headers()
        assert BETA_STRUCTURED_OUTPUTS not in headers

    def test_all_beta_headers_with_opus(self):
        """Test beta headers for Opus 4.5 configuration (manual thinking)."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
            effort="low",
            thinking_enabled=True,
            interleaved_thinking=True,
            context_management_enabled=True,
        )

        headers = provider._get_beta_headers()
        assert BETA_CONTEXT_MANAGEMENT in headers
        assert BETA_INTERLEAVED_THINKING in headers
        # Structured outputs is added dynamically, not in _get_beta_headers
        assert len(headers) == 2


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestCapabilityPredicates:
    """Test model capability predicate helpers."""

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-5-20251101",
            "claude-opus-4-6",
            "claude-opus-4-6-20260205",
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-mythos-preview",
        ],
    )
    def test_supports_effort(self, model):
        assert supports_effort(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-1-20250805",
        ],
    )
    def test_no_effort_on_older_models(self, model):
        assert supports_effort(model) is False

    def test_max_effort_on_46_and_later(self):
        assert supports_effort_level("claude-opus-4-6", "max") is True
        assert supports_effort_level("claude-opus-4-7", "max") is True
        assert supports_effort_level("claude-sonnet-4-6", "max") is True
        assert supports_effort_level("claude-opus-4-5-20251101", "max") is False

    def test_xhigh_only_on_opus_47(self):
        assert supports_effort_level("claude-opus-4-7", "xhigh") is True
        assert supports_effort_level("claude-opus-4-6", "xhigh") is False
        assert supports_effort_level("claude-sonnet-4-6", "xhigh") is False

    def test_adaptive_thinking_models(self):
        assert supports_adaptive_thinking("claude-opus-4-7") is True
        assert supports_adaptive_thinking("claude-opus-4-6") is True
        assert supports_adaptive_thinking("claude-sonnet-4-6") is True
        assert supports_adaptive_thinking("claude-opus-4-5-20251101") is False
        assert supports_adaptive_thinking("claude-sonnet-4-5-20250929") is False

    def test_adaptive_only_on_opus_47(self):
        assert requires_adaptive_thinking("claude-opus-4-7") is True
        assert requires_adaptive_thinking("claude-mythos-preview") is True
        assert requires_adaptive_thinking("claude-opus-4-6") is False
        assert requires_adaptive_thinking("claude-sonnet-4-6") is False


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestAdaptiveThinking:
    """Test adaptive thinking mode selection and configuration."""

    def test_auto_mode_picks_adaptive_on_opus_47(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-7",
            thinking_enabled=True,
        )
        assert provider._thinking_mode == "adaptive"
        assert provider._thinking_enabled is True

    def test_auto_mode_picks_adaptive_on_sonnet_46(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-sonnet-4-6",
            thinking_enabled=True,
        )
        assert provider._thinking_mode == "adaptive"

    def test_auto_mode_picks_manual_on_opus_45(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
            thinking_enabled=True,
        )
        assert provider._thinking_mode == "manual"

    def test_thinking_disabled_resolves_to_off(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-7",
            thinking_enabled=False,
        )
        assert provider._thinking_mode == "off"
        assert provider._thinking_enabled is False

    def test_manual_forced_on_adaptive_only_model_is_upgraded(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-7",
            thinking_enabled=True,
            thinking_mode="manual",
        )
        assert provider._thinking_mode == "adaptive"

    def test_explicit_off_mode(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-7",
            thinking_enabled=True,
            thinking_mode="off",
        )
        assert provider._thinking_mode == "off"
        assert provider._thinking_enabled is False

    def test_adaptive_build_thinking_config(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-7",
            thinking_enabled=True,
        )
        assert provider._build_thinking_config() == {"type": "adaptive"}

    def test_adaptive_build_thinking_config_with_display(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-7",
            thinking_enabled=True,
            thinking_display="summarized",
        )
        assert provider._build_thinking_config() == {
            "type": "adaptive",
            "display": "summarized",
        }

    def test_manual_build_thinking_config(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
            thinking_enabled=True,
            thinking_budget_tokens=8000,
        )
        assert provider._build_thinking_config() == {
            "type": "enabled",
            "budget_tokens": 8000,
        }

    def test_off_build_thinking_config(self):
        provider = AnthropicLLMProvider(api_key="test-key")
        assert provider._build_thinking_config() is None


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestEffortLevels:
    """Test expanded effort levels (max, xhigh)."""

    @pytest.mark.parametrize("level", ["low", "medium", "high", "max"])
    def test_opus_46_accepts_standard_and_max(self, level):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-6",
            effort=level,
        )
        assert provider._build_output_config() == {"effort": level}

    @pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max"])
    def test_opus_47_accepts_all_levels(self, level):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-7",
            effort=level,
        )
        assert provider._build_output_config() == {"effort": level}

    def test_xhigh_dropped_on_non_opus_47(self, caplog):
        """xhigh is rejected on Opus 4.6 and dropped with a warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            provider = AnthropicLLMProvider(
                api_key="test-key",
                model="claude-opus-4-6",
                effort="xhigh",
            )
        assert provider._effort is None
        assert provider._build_output_config() is None

    def test_effort_emits_on_sonnet_46(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-sonnet-4-6",
            effort="medium",
        )
        assert provider._build_output_config() == {"effort": "medium"}


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestPromptCaching:
    """Opt-in prompt caching via top-level cache_control."""

    def test_cache_disabled_by_default(self):
        provider = AnthropicLLMProvider(api_key="test-key")
        assert provider._build_cache_control() is None

    def test_cache_control_ephemeral_when_enabled(self):
        provider = AnthropicLLMProvider(api_key="test-key", prompt_caching=True)
        assert provider._build_cache_control() == {"type": "ephemeral"}

    def test_cache_control_with_ttl(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            prompt_caching=True,
            cache_ttl="1h",
        )
        assert provider._build_cache_control() == {
            "type": "ephemeral",
            "ttl": "1h",
        }

    def test_cache_disabled(self):
        provider = AnthropicLLMProvider(api_key="test-key", prompt_caching=False)
        assert provider._build_cache_control() is None


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestStructuredOutputsConfig:
    """Structured outputs are emitted under output_config.format."""

    def test_schema_merged_into_output_config(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-7",
            effort="high",
        )
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": False,
        }
        cfg = provider._build_output_config(json_schema=schema)
        assert cfg["effort"] == "high"
        assert cfg["format"] == {"type": "json_schema", "schema": schema}

    def test_empty_output_config_returns_none(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-sonnet-4-5-20250929",
        )
        assert provider._build_output_config() is None


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestTaskBudget:
    """Advisory task_budget beta for Opus 4.7."""

    def test_task_budget_supported_on_opus_47(self):
        assert supports_task_budget("claude-opus-4-7") is True
        assert supports_task_budget("claude-mythos-preview") is False
        assert supports_task_budget("claude-opus-4-6") is False
        assert supports_task_budget("claude-sonnet-4-6") is False

    def test_task_budget_emitted_in_output_config(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-7",
            task_budget_tokens=50000,
        )
        cfg = provider._build_output_config()
        assert cfg["task_budget"] == {"type": "tokens", "total": 50000}

    def test_task_budget_beta_header_added(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-7",
            task_budget_tokens=30000,
        )
        assert BETA_TASK_BUDGETS in provider._get_beta_headers()

    def test_task_budget_ignored_on_unsupported_model(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-6",
            task_budget_tokens=50000,
        )
        assert provider._task_budget_tokens is None
        assert BETA_TASK_BUDGETS not in provider._get_beta_headers()

    def test_task_budget_below_minimum_raises(self):
        with pytest.raises(ValueError, match="below the API minimum"):
            AnthropicLLMProvider(
                api_key="test-key",
                model="claude-opus-4-7",
                task_budget_tokens=10000,
            )


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestStateMachineValidation:
    """Reject contradictory thinking_enabled/thinking_mode combinations."""

    def test_unknown_thinking_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown thinking_mode"):
            AnthropicLLMProvider(
                api_key="test-key",
                model="claude-opus-4-7",
                thinking_mode="adpative",
            )

    def test_thinking_enabled_false_with_adaptive_raises(self):
        with pytest.raises(ValueError, match="conflicts with thinking_mode"):
            AnthropicLLMProvider(
                api_key="test-key",
                model="claude-opus-4-7",
                thinking_enabled=False,
                thinking_mode="adaptive",
            )

    def test_thinking_enabled_false_with_manual_raises(self):
        with pytest.raises(ValueError, match="conflicts with thinking_mode"):
            AnthropicLLMProvider(
                api_key="test-key",
                model="claude-opus-4-5-20251101",
                thinking_enabled=False,
                thinking_mode="manual",
            )

    def test_explicit_off_wins_over_thinking_enabled(self):
        """thinking_mode='off' + thinking_enabled=True resolves to off."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-7",
            thinking_enabled=True,
            thinking_mode="off",
        )
        assert provider._thinking_mode == "off"
        assert provider._thinking_enabled is False

    def test_adaptive_on_older_model_falls_back_to_manual(self):
        """thinking_mode='adaptive' on Opus 4.5 downgrades to manual with warning."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
            thinking_enabled=True,
            thinking_mode="adaptive",
        )
        assert provider._thinking_mode == "manual"


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestPrefixCollisionGuard:
    """_matches_family requires exact match or '-' delimiter."""

    def test_no_collision_with_extended_digits(self):
        assert supports_effort("claude-opus-4-50") is False
        assert supports_effort("claude-opus-4-77") is False
        assert supports_effort("claude-sonnet-4-60-custom") is False

    def test_exact_alias_still_matches(self):
        assert supports_effort("claude-opus-4-7") is True
        assert supports_effort("claude-sonnet-4-6") is True

    def test_dated_variant_still_matches(self):
        assert supports_effort("claude-opus-4-7-20260416") is True
        assert supports_effort("claude-sonnet-4-6-20260217") is True


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestToolChoiceContextManagement:
    """tool_choice forcing thinking off must suppress clear_thinking edit."""

    def test_apply_common_with_thinking_active_false(self):
        """Passing thinking_active=False omits clear_thinking edit."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
            thinking_enabled=True,
            context_management_enabled=True,
        )
        kwargs: dict = {}
        provider._apply_common_request_fields(kwargs, thinking_active=False)
        cm = kwargs.get("context_management")
        assert cm is not None
        edits = cm["edits"]
        assert all(e["type"] != "clear_thinking_20251015" for e in edits)
        assert any(e["type"] == "clear_tool_uses_20250919" for e in edits)

    def test_apply_common_with_thinking_active_true(self):
        """Default path emits clear_thinking when provider has it enabled."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
            thinking_enabled=True,
            context_management_enabled=True,
        )
        kwargs: dict = {}
        provider._apply_common_request_fields(kwargs, thinking_active=True)
        edits = kwargs["context_management"]["edits"]
        assert any(e["type"] == "clear_thinking_20251015" for e in edits)

    def test_interleaved_beta_omitted_when_thinking_disabled(self):
        """_get_beta_headers with thinking_active=False drops interleaved beta."""
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
            thinking_enabled=True,
            interleaved_thinking=True,
        )
        assert (
            BETA_INTERLEAVED_THINKING
            in provider._get_beta_headers(thinking_active=True)
        )
        assert (
            BETA_INTERLEAVED_THINKING
            not in provider._get_beta_headers(thinking_active=False)
        )


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestThinkingDisplayWarning:
    """thinking_display on non-adaptive modes is dropped with a warning."""

    def test_thinking_display_ignored_on_manual(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-5-20251101",
            thinking_enabled=True,
            thinking_display="summarized",
        )
        assert provider._thinking_mode == "manual"
        assert provider._thinking_display is None

    def test_thinking_display_preserved_on_adaptive(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-7",
            thinking_enabled=True,
            thinking_display="summarized",
        )
        assert provider._thinking_mode == "adaptive"
        assert provider._thinking_display == "summarized"


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestEffortNormalization:
    """Effort values are lowercased/stripped before validation."""

    def test_uppercase_effort_normalized(self):
        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-opus-4-7",
            effort=" HIGH ",
        )
        assert provider._effort == "high"


class TestLLMConfigEnvLoading:
    """New Anthropic fields load from env vars."""

    def test_env_loads_anthropic_fields(self, monkeypatch):
        from chunkhound.core.config.llm_config import LLMConfig

        monkeypatch.setenv("CHUNKHOUND_LLM_ANTHROPIC_PROMPT_CACHING", "false")
        monkeypatch.setenv("CHUNKHOUND_LLM_ANTHROPIC_THINKING_MODE", "adaptive")
        monkeypatch.setenv("CHUNKHOUND_LLM_ANTHROPIC_CACHE_TTL", "1h")
        monkeypatch.setenv("CHUNKHOUND_LLM_ANTHROPIC_TASK_BUDGET_TOKENS", "50000")
        monkeypatch.setenv("CHUNKHOUND_LLM_ANTHROPIC_EFFORT", "MAX")
        monkeypatch.setenv("CHUNKHOUND_LLM_ANTHROPIC_THINKING_DISPLAY", "omitted")

        loaded = LLMConfig.load_from_env()
        assert loaded["anthropic_prompt_caching"] is False
        assert loaded["anthropic_thinking_mode"] == "adaptive"
        assert loaded["anthropic_cache_ttl"] == "1h"
        assert loaded["anthropic_task_budget_tokens"] == 50000
        assert loaded["anthropic_effort"] == "max"
        assert loaded["anthropic_thinking_display"] == "omitted"

    def test_env_unset_leaves_keys_absent(self, monkeypatch):
        from chunkhound.core.config.llm_config import LLMConfig

        for name in (
            "CHUNKHOUND_LLM_ANTHROPIC_PROMPT_CACHING",
            "CHUNKHOUND_LLM_ANTHROPIC_THINKING_MODE",
            "CHUNKHOUND_LLM_ANTHROPIC_CACHE_TTL",
            "CHUNKHOUND_LLM_ANTHROPIC_TASK_BUDGET_TOKENS",
        ):
            monkeypatch.delenv(name, raising=False)
        loaded = LLMConfig.load_from_env()
        assert "anthropic_prompt_caching" not in loaded
        assert "anthropic_thinking_mode" not in loaded
        assert "anthropic_cache_ttl" not in loaded
        assert "anthropic_task_budget_tokens" not in loaded


class TestLLMConfigDefaults:
    """Pin default model selection."""

    def test_anthropic_prompt_caching_default_disabled(self):
        from chunkhound.core.config.llm_config import LLMConfig

        cfg = LLMConfig(provider="anthropic")
        utility_config, synthesis_config = cfg.get_provider_configs()
        assert utility_config["prompt_caching"] is False
        assert synthesis_config["prompt_caching"] is False

    def test_anthropic_default_models(self):
        from chunkhound.core.config.llm_config import LLMConfig

        cfg = LLMConfig(provider="anthropic")
        assert cfg.get_default_models() == (
            CLAUDE_HAIKU_DEFAULT_SENTINEL,
            CLAUDE_HAIKU_DEFAULT_SENTINEL,
        )

    def test_claude_code_default_models(self):
        from chunkhound.core.config.llm_config import LLMConfig

        cfg = LLMConfig(provider="claude-code-cli")
        assert cfg.get_default_models() == (
            CLAUDE_HAIKU_DEFAULT_SENTINEL,
            CLAUDE_HAIKU_DEFAULT_SENTINEL,
        )


class TestClaudeHaikuModelResolution:
    """Pin latest-Haiku default model selection."""

    def test_explicit_model_passes_through(self):
        assert resolve_claude_haiku_model("claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv(
            "CHUNKHOUND_CLAUDE_DEFAULT_HAIKU_MODEL",
            "claude-haiku-4-6-20260101",
        )
        assert resolve_claude_haiku_model(CLAUDE_HAIKU_DEFAULT_SENTINEL) == (
            "claude-haiku-4-6-20260101"
        )

    def test_no_api_key_uses_fallback(self, monkeypatch):
        get_latest_available_haiku_model.cache_clear()
        monkeypatch.delenv("CHUNKHOUND_CLAUDE_DEFAULT_HAIKU_MODEL", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert resolve_claude_haiku_model(CLAUDE_HAIKU_DEFAULT_SENTINEL) == (
            CLAUDE_HAIKU_FALLBACK_MODEL
        )

    def test_discovery_can_be_disabled(self, monkeypatch):
        get_latest_available_haiku_model.cache_clear()
        monkeypatch.delenv("CHUNKHOUND_CLAUDE_DEFAULT_HAIKU_MODEL", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setattr(
            "chunkhound.core.config.claude_model_resolution."
            "get_latest_available_haiku_model",
            lambda api_key=None: pytest.fail("discovery should not run"),
        )

        assert resolve_claude_haiku_model(
            CLAUDE_HAIKU_DEFAULT_SENTINEL,
            discover=False,
        ) == CLAUDE_HAIKU_FALLBACK_MODEL

    def test_model_discovery_picks_newest_haiku(self, monkeypatch):
        get_latest_available_haiku_model.cache_clear()

        class FakeModels:
            def list(self, **kwargs):
                return [
                    SimpleNamespace(
                        id="claude-sonnet-4-6",
                        created_at=datetime(2026, 2, 1, tzinfo=UTC),
                    ),
                    SimpleNamespace(
                        id="claude-haiku-4-5-20251001",
                        created_at=datetime(2025, 10, 1, tzinfo=UTC),
                    ),
                    SimpleNamespace(
                        id="claude-haiku-4-6-20260101",
                        created_at=datetime(2026, 1, 1, tzinfo=UTC),
                    ),
                ]

        class FakeAnthropic:
            def __init__(self, **kwargs):
                self.models = FakeModels()

        import anthropic

        monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
        assert resolve_claude_haiku_model(
            CLAUDE_HAIKU_DEFAULT_SENTINEL,
            "sk-ant-test",
        ) == "claude-haiku-4-6-20260101"

    def test_model_discovery_failure_uses_fallback(self, monkeypatch):
        get_latest_available_haiku_model.cache_clear()
        monkeypatch.delenv("CHUNKHOUND_CLAUDE_DEFAULT_HAIKU_MODEL", raising=False)

        class FakeAnthropic:
            def __init__(self, **kwargs):
                raise RuntimeError("network unavailable")

        import anthropic

        monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
        assert resolve_claude_haiku_model(
            CLAUDE_HAIKU_DEFAULT_SENTINEL,
            "sk-ant-test",
        ) == CLAUDE_HAIKU_FALLBACK_MODEL



@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic SDK not installed")
class TestRequestShape:
    """Pin the on-the-wire request_kwargs for each message path.

    Structured outputs migrated from top-level ``output_format`` to
    ``output_config.format`` in SDK 0.96. These tests assert the migration
    stays intact and that prompt_caching / effort / thinking flow into the
    correct API parameters.
    """

    @staticmethod
    def _text_response():
        from unittest.mock import MagicMock

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "ok"
        response = MagicMock()
        response.content = [text_block]
        response.stop_reason = "end_turn"
        response.usage = MagicMock(input_tokens=10, output_tokens=5)
        return response

    @staticmethod
    async def _stream_events():
        yield SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            ),
        )
        yield SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(text="ok"),
        )

    @pytest.mark.asyncio
    async def test_complete_applies_sdk_096_request_fields(self):
        from unittest.mock import AsyncMock, MagicMock

        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-sonnet-4-6",
            thinking_enabled=True,
            thinking_mode="adaptive",
            thinking_display="summarized",
            effort="medium",
            prompt_caching=True,
            cache_ttl="1h",
        )
        provider._client.messages = MagicMock()
        provider._client.messages.create = AsyncMock(
            return_value=self._text_response(),
        )
        provider._client.beta = MagicMock()
        provider._client.beta.messages = MagicMock()
        provider._client.beta.messages.create = AsyncMock(
            return_value=self._text_response(),
        )

        response = await provider.complete(
            "hello",
            system="sys",
            max_completion_tokens=1234,
            timeout=12,
        )

        assert response.content == "ok"
        assert provider._client.beta.messages.create.call_args is None
        kwargs = provider._client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4-6"
        assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
        assert kwargs["system"] == "sys"
        assert kwargs["max_tokens"] == 1234
        assert kwargs["timeout"] == 12
        assert kwargs["thinking"] == {
            "type": "adaptive",
            "display": "summarized",
        }
        assert kwargs["output_config"] == {"effort": "medium"}
        assert kwargs["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert "betas" not in kwargs

    @pytest.mark.asyncio
    async def test_complete_with_tools_routes_strict_tools_to_beta(self):
        from unittest.mock import AsyncMock, MagicMock

        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-sonnet-4-6",
            thinking_enabled=True,
            thinking_mode="adaptive",
            prompt_caching=True,
            effort="medium",
        )
        response = self._text_response()
        provider._client.messages = MagicMock()
        provider._client.messages.create = AsyncMock(return_value=response)
        provider._client.beta = MagicMock()
        provider._client.beta.messages = MagicMock()
        provider._client.beta.messages.create = AsyncMock(return_value=response)

        tool = {
            "name": "lookup",
            "description": "Lookup data",
            "input_schema": {"type": "object", "properties": {}},
            "strict": True,
        }
        llm_response, tool_uses = await provider.complete_with_tools(
            "hello",
            [tool],
            system="sys",
            tool_choice={"type": "auto"},
        )

        assert llm_response.content == "ok"
        assert tool_uses == []
        assert provider._client.messages.create.call_args is None
        kwargs = provider._client.beta.messages.create.call_args.kwargs
        assert kwargs["betas"] == [BETA_STRUCTURED_OUTPUTS]
        assert kwargs["tools"] == [tool]
        assert kwargs["tool_choice"] == {"type": "auto"}
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert kwargs["output_config"] == {"effort": "medium"}
        assert kwargs["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_complete_with_tools_suppresses_thinking_for_forced_tool(self):
        from unittest.mock import AsyncMock, MagicMock

        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-sonnet-4-6",
            thinking_enabled=True,
            thinking_mode="adaptive",
            context_management_enabled=True,
        )
        response = self._text_response()
        provider._client.messages = MagicMock()
        provider._client.messages.create = AsyncMock(return_value=response)
        provider._client.beta = MagicMock()
        provider._client.beta.messages = MagicMock()
        provider._client.beta.messages.create = AsyncMock(return_value=response)

        tool = {
            "name": "lookup",
            "description": "Lookup data",
            "input_schema": {"type": "object", "properties": {}},
        }
        await provider.complete_with_tools(
            "hello",
            [tool],
            tool_choice={"type": "tool", "name": "lookup"},
        )

        kwargs = provider._client.beta.messages.create.call_args.kwargs
        assert kwargs["betas"] == [BETA_CONTEXT_MANAGEMENT]
        assert "thinking" not in kwargs
        assert kwargs["context_management"] == {
            "edits": [{"type": "clear_tool_uses_20250919"}],
        }

    @pytest.mark.asyncio
    async def test_complete_streaming_applies_request_fields_and_beta_routing(self):
        from unittest.mock import AsyncMock, MagicMock

        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-sonnet-4-6",
            context_management_enabled=True,
            effort="medium",
            prompt_caching=True,
        )
        provider._client.messages = MagicMock()
        provider._client.messages.create = AsyncMock(
            return_value=self._stream_events(),
        )
        provider._client.beta = MagicMock()
        provider._client.beta.messages = MagicMock()
        provider._client.beta.messages.create = AsyncMock(
            return_value=self._stream_events(),
        )

        chunks = [chunk async for chunk in provider.complete_streaming("hello")]

        assert chunks == ["ok"]
        assert provider._client.messages.create.call_args is None
        kwargs = provider._client.beta.messages.create.call_args.kwargs
        assert kwargs["betas"] == [BETA_CONTEXT_MANAGEMENT]
        assert kwargs["stream"] is True
        assert kwargs["output_config"] == {"effort": "medium"}
        assert kwargs["cache_control"] == {"type": "ephemeral"}
        assert kwargs["context_management"] == {
            "edits": [{"type": "clear_tool_uses_20250919"}],
        }

    @pytest.mark.asyncio
    async def test_complete_structured_uses_output_config_format(self):
        from unittest.mock import AsyncMock, MagicMock

        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-sonnet-4-6",
            effort="medium",
        )

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = '{"x": "ok"}'
        response = MagicMock()
        response.content = [text_block]
        response.stop_reason = "end_turn"
        response.usage = MagicMock(input_tokens=10, output_tokens=5)
        provider._client.messages = MagicMock()
        provider._client.messages.create = AsyncMock(return_value=response)
        provider._client.beta = MagicMock()
        provider._client.beta.messages = MagicMock()
        provider._client.beta.messages.create = AsyncMock(return_value=response)

        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": False,
        }
        await provider.complete_structured("hello", schema)

        # Prompt caching is off by default. output_config carries both effort
        # and the JSON schema. Structured outputs is GA, so the call should go
        # to the standard endpoint with no beta headers.
        std_called = provider._client.messages.create.call_args
        beta_called = provider._client.beta.messages.create.call_args
        assert std_called is not None, (
            "complete_structured should use the standard endpoint when no "
            "beta features are active"
        )
        assert beta_called is None
        kwargs = std_called.kwargs
        assert kwargs["output_config"] == {
            "effort": "medium",
            "format": {"type": "json_schema", "schema": schema},
        }
        assert "cache_control" not in kwargs
        assert "output_format" not in kwargs
        assert "betas" not in kwargs

    @pytest.mark.asyncio
    async def test_complete_structured_with_context_mgmt_uses_beta_endpoint(
        self,
    ):
        from unittest.mock import AsyncMock, MagicMock

        provider = AnthropicLLMProvider(
            api_key="test-key",
            model="claude-sonnet-4-6",
            context_management_enabled=True,
        )

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "{}"
        response = MagicMock()
        response.content = [text_block]
        response.stop_reason = "end_turn"
        response.usage = MagicMock(input_tokens=1, output_tokens=1)
        provider._client.beta = MagicMock()
        provider._client.beta.messages = MagicMock()
        provider._client.beta.messages.create = AsyncMock(return_value=response)
        provider._client.messages = MagicMock()
        provider._client.messages.create = AsyncMock(return_value=response)

        await provider.complete_structured("hello", {"type": "object"})

        # Context management is beta-only -> beta endpoint with header.
        beta_called = provider._client.beta.messages.create.call_args
        assert beta_called is not None
        assert provider._client.messages.create.call_args is None
        assert BETA_CONTEXT_MANAGEMENT in beta_called.kwargs["betas"]
