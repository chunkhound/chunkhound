"""High-value functional tests for OpenAILLMProvider (Responses API path)."""

import pytest
from unittest.mock import AsyncMock, patch

from chunkhound.providers.llm.openai_llm_provider import OpenAILLMProvider
from chunkhound.interfaces.llm_provider import LLMResponse


@pytest.fixture
def mock_openai_client():
    with patch("chunkhound.providers.llm.openai_compatible_provider.AsyncOpenAI") as mock:
        client = mock.return_value
        client.responses.create = AsyncMock()      # Responses API (default path)
        client.chat.completions.create = AsyncMock()  # fallback for older models
        yield client


class TestOpenAILLMProvider:
    """Only tests real user-facing behavior + config application."""

    @pytest.mark.asyncio
    async def test_complete_returns_llmresponse_with_content(self, mock_openai_client):
        """Core contract: complete() must return LLMResponse with valid text."""
        mock_resp = AsyncMock()
        mock_resp.output = [
            AsyncMock(
                type="message",
                content=[AsyncMock(type="output_text", text="Chunking is working perfectly!")]
            )
        ]
        mock_resp.usage = AsyncMock(total_tokens=42)
        mock_resp.status = "completed"
        mock_openai_client.responses.create.return_value = mock_resp

        provider = OpenAILLMProvider(api_key="sk-test")  # default = gpt-5-nano-mini → Responses
        response = await provider.complete("Explain chunking")

        assert isinstance(response, LLMResponse)
        assert response.content == "Chunking is working perfectly!"
        assert response.tokens_used == 42
        assert response.model == "gpt-5-nano-mini"

    @pytest.mark.asyncio
    async def test_configuration_is_respected_in_api_call(self, mock_openai_client):
        """Valuable: model, max tokens, reasoning_effort, timeout must actually be sent."""
        provider = OpenAILLMProvider(
            api_key="sk-test",
            model="gpt-4o",
            reasoning_effort="low",
            timeout=30
        )
        mock_openai_client.responses.create.return_value = AsyncMock(
            output=[AsyncMock(type="message", content=[AsyncMock(type="output_text", text="ok")])],
            usage=AsyncMock(total_tokens=10),
            status="completed"
        )

        await provider.complete("Test config", max_completion_tokens=500)

        call = mock_openai_client.responses.create.call_args[1]
        assert call["model"] == "gpt-4o"
        assert call["max_output_tokens"] == 500
        assert call["timeout"] == 30
        assert call.get("reasoning") == {"effort": "low"}

    @pytest.mark.asyncio
    async def test_api_errors_propagate_to_caller(self, mock_openai_client):
        """Critical: errors must bubble up (MCP server depends on this)."""
        mock_openai_client.responses.create.side_effect = Exception("429 rate limit")

        provider = OpenAILLMProvider(api_key="sk-test")
        with pytest.raises(RuntimeError) as exc:
            await provider.complete("boom")

        assert "LLM completion failed" in str(exc.value)
        assert "rate limit" in str(exc.value).lower()