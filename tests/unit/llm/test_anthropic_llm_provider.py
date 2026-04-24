"""Tests for Anthropic LLM provider."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.providers.llm.anthropic_llm_provider import AnthropicLLMProvider


@pytest.mark.asyncio
async def test_internal_runtime_error_not_double_wrapped_complete():
    """RuntimeError raised inside complete() must pass through unwrapped."""
    provider = AnthropicLLMProvider(api_key="test-key")
    mock_resp = MagicMock()
    mock_resp.content = []  # empty → raises RuntimeError at content_blocks check
    mock_resp.stop_reason = "end_turn"
    provider._client = MagicMock()
    provider._client.messages.create = AsyncMock(return_value=mock_resp)

    with pytest.raises(RuntimeError) as exc:
        await provider.complete("test")

    msg = str(exc.value)
    assert "LLM returned empty response" in msg
    assert "LLM completion failed" not in msg


@pytest.mark.asyncio
async def test_internal_runtime_error_not_double_wrapped_complete_structured():
    """RuntimeError raised inside complete_structured() must pass through unwrapped."""
    provider = AnthropicLLMProvider(api_key="test-key")
    mock_resp = MagicMock()
    mock_resp.stop_reason = "max_tokens"  # triggers RuntimeError("Structured output truncated...")
    mock_resp.usage = None
    provider._create_message = AsyncMock(return_value=mock_resp)

    with pytest.raises(RuntimeError) as exc:
        await provider.complete_structured("test", {"type": "object"})

    msg = str(exc.value)
    assert "Structured output truncated" in msg
    assert "LLM structured completion failed" not in msg
