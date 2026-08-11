"""Tests for Anthropic LLM provider."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.interfaces.llm_provider import (
    PROVIDER_MANAGED_OUTPUT,
    OutputLimitCapability,
)
from chunkhound.providers.llm.anthropic_llm_provider import AnthropicLLMProvider


@pytest.mark.asyncio
async def test_provider_managed_always_sends_numeric_fallback_for_text():
    """Anthropic's required max_tokens field never receives enum/None values."""
    provider = AnthropicLLMProvider(api_key="test-key")
    provider.configure_synthesis_output_limit_policy(
        output_limits_enabled=False,
        fallback_tokens=76_001,
    )
    response = MagicMock()
    response.content = [MagicMock(type="text", text="ok")]
    response.stop_reason = "end_turn"
    response.usage = None
    provider._create_message = AsyncMock(return_value=response)

    await provider.complete("test", max_completion_tokens=PROVIDER_MANAGED_OUTPUT)

    assert provider.output_limit_metadata.omission is OutputLimitCapability.REQUIRED
    assert provider._create_message.call_args.args[0]["max_tokens"] == 76_001


@pytest.mark.asyncio
async def test_provider_managed_structured_sends_numeric_fallback_and_schema():
    """Structured requests retain their schema while resolving the required cap."""
    provider = AnthropicLLMProvider(api_key="test-key")
    provider.configure_synthesis_output_limit_policy(
        output_limits_enabled=False,
        fallback_tokens=76_002,
    )
    response = MagicMock()
    response.content = [MagicMock(type="text", text='{"answer":"42"}')]
    response.stop_reason = "end_turn"
    response.usage = None
    provider._create_message = AsyncMock(return_value=response)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

    result = await provider.complete_structured(
        "test",
        schema,
        max_completion_tokens=PROVIDER_MANAGED_OUTPUT,
    )

    request = provider._create_message.call_args.args[0]
    assert result == {"answer": "42"}
    assert request["max_tokens"] == 76_002
    assert request["output_config"]["format"]["schema"] == schema


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
async def test_max_tokens_error_wins_over_empty_response():
    """Anthropic's native truncation signal must beat empty-content checks."""
    provider = AnthropicLLMProvider(api_key="test-key")
    mock_resp = MagicMock()
    mock_resp.content = []
    mock_resp.stop_reason = "max_tokens"
    mock_resp.usage = None
    provider._client = MagicMock()
    provider._client.messages.create = AsyncMock(return_value=mock_resp)

    with pytest.raises(RuntimeError, match="token limit exceeded") as exc:
        await provider.complete("test")

    assert "empty response" not in str(exc.value)


@pytest.mark.asyncio
async def test_internal_runtime_error_not_double_wrapped_complete_structured():
    """RuntimeError raised inside complete_structured() must pass through unwrapped."""
    provider = AnthropicLLMProvider(api_key="test-key")
    mock_resp = MagicMock()
    mock_resp.stop_reason = (
        "max_tokens"  # triggers RuntimeError("Structured output truncated...")
    )
    mock_resp.usage = None
    provider._create_message = AsyncMock(return_value=mock_resp)

    with pytest.raises(RuntimeError) as exc:
        await provider.complete_structured("test", {"type": "object"})

    msg = str(exc.value)
    assert "Structured output truncated" in msg
    assert "LLM structured completion failed" not in msg
