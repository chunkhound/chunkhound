"""High-value functional tests for GrokLLMProvider (chat completions path)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.interfaces.llm_provider import LLMResponse
from chunkhound.providers.llm.grok_llm_provider import GrokLLMProvider


@pytest.fixture
def mock_grok_client():
    with patch(
        "chunkhound.providers.llm.openai_compatible_provider.AsyncOpenAI"
    ) as mock:
        client = mock.return_value
        client.chat.completions.create = AsyncMock()
        yield client


@pytest.fixture
def provider():
    """Create a GrokLLMProvider instance for testing."""
    return GrokLLMProvider(
        api_key="test-api-key-123",
        model="grok-4-1-fast-reasoning",
        timeout=DEFAULT_LLM_TIMEOUT,
        max_retries=3,
    )


def test_default_timeout():
    """Default timeout resolves to 120."""
    from chunkhound.providers.llm.grok_llm_provider import GrokLLMProvider
    p = GrokLLMProvider(api_key="test-key")
    assert p.timeout == DEFAULT_LLM_TIMEOUT


class TestGrokLLMProvider:
    """Grok uses the shared OpenAI-compatible base — just verify contract."""

    @pytest.mark.asyncio
    async def test_complete_returns_llmresponse(self, mock_grok_client):
        """Core: must return LLMResponse (same contract as every other provider)."""
        mock_resp = AsyncMock()
        mock_resp.choices = [AsyncMock(message=AsyncMock(content="Grok is ready"))]
        mock_resp.usage = AsyncMock(total_tokens=15)
        mock_resp.choices[0].finish_reason = "stop"
        mock_grok_client.chat.completions.create.return_value = mock_resp

        provider = GrokLLMProvider(api_key="gsk-test")
        response = await provider.complete("Say hello")

        assert isinstance(response, LLMResponse)
        assert response.content == "Grok is ready"
        assert response.tokens_used == 15
        assert response.model == "grok-4-1-fast-reasoning"

    @pytest.mark.asyncio
    async def test_configuration_is_respected(self, mock_grok_client):
        """Valuable: config must reach the underlying OpenAI client."""
        provider = GrokLLMProvider(
            api_key="gsk-test",
            model="grok-beta",
            reasoning_effort="high",
        )
        mock_grok_client.chat.completions.create.return_value = AsyncMock(
            choices=[AsyncMock(message=AsyncMock(content="ok"))],
            usage=AsyncMock(total_tokens=5),
        )

        await provider.complete("hi", max_completion_tokens=200)

        call = mock_grok_client.chat.completions.create.call_args[1]
        assert call["model"] == "grok-beta"
        assert call["max_completion_tokens"] == 200
        assert call["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_configuration_omits_reasoning_effort_when_unset(self, mock_grok_client):
        """None should omit reasoning_effort instead of sending a 'none' string."""
        provider = GrokLLMProvider(api_key="gsk-test", model="grok-beta")
        mock_grok_client.chat.completions.create.return_value = AsyncMock(
            choices=[AsyncMock(message=AsyncMock(content="ok"))],
            usage=AsyncMock(total_tokens=5),
        )

        await provider.complete("hi", max_completion_tokens=200)

        call = mock_grok_client.chat.completions.create.call_args[1]
        assert "reasoning_effort" not in call

    @pytest.mark.asyncio
    async def test_errors_propagate(self, mock_grok_client):
        """Errors must not be swallowed."""
        mock_grok_client.chat.completions.create.side_effect = Exception("xAI outage")

        provider = GrokLLMProvider(api_key="gsk-test")
        with pytest.raises(RuntimeError) as exc:
            await provider.complete("fail")

        assert "LLM completion failed" in str(exc.value)

    @pytest.mark.asyncio
    async def test_truncation_error_wins_over_empty_response(self, mock_grok_client):
        """Grok should inherit shared truncation-before-empty behavior."""
        mock_resp = AsyncMock()
        mock_resp.choices = [AsyncMock(message=AsyncMock(content="  "))]
        mock_resp.usage = AsyncMock(
            prompt_tokens=123,
            completion_tokens=0,
            total_tokens=123,
        )
        mock_resp.choices[0].finish_reason = "length"
        mock_grok_client.chat.completions.create.return_value = mock_resp

        provider = GrokLLMProvider(api_key="gsk-test")

        with pytest.raises(RuntimeError, match="token limit exceeded") as exc:
            await provider.complete("hi")

        assert "empty response" not in str(exc.value)

    def test_get_usage_stats(self, provider):
        """Test usage statistics retrieval."""
        # Initially zero
        stats = provider.get_usage_stats()
        assert stats["requests_made"] == 0
        assert stats["total_tokens"] == 0
        assert stats["prompt_tokens"] == 0
        assert stats["completion_tokens"] == 0

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
        assert provider.get_synthesis_concurrency() == 5

    def test_base_url_default(self, provider):
        """Test that default base URL is set correctly."""
        assert str(provider._client.base_url) == "https://api.x.ai/v1/"

    def test_base_url_custom(self):
        """Test custom base URL."""
        provider = GrokLLMProvider(
            api_key="test-key", model="grok-beta", base_url="https://custom.api.x.ai/v1"
        )
        assert str(provider._client.base_url) == "https://custom.api.x.ai/v1/"


class TestNativeStructuredOutputPath:
    """Verify the native structured-output path for providers that support it."""

    SCHEMA = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    def _make_mock_response(self, content: str):
        choice = MagicMock()
        choice.message.content = content
        choice.finish_reason = "stop"

        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 20
        usage.total_tokens = 30

        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = usage
        return resp

    @pytest.mark.asyncio
    async def test_native_path_sends_response_format(self, mock_grok_client):
        """Native path must include response_format with json_schema."""
        mock_grok_client.chat.completions.create.return_value = (
            self._make_mock_response('{"answer": "42"}')
        )

        provider = GrokLLMProvider(api_key="sk-test", reasoning_effort="medium")
        await provider.complete_structured(
            "What is the answer?",
            json_schema=self.SCHEMA,
        )

        call_kwargs = mock_grok_client.chat.completions.create.call_args[1]
        assert "response_format" in call_kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["reasoning_effort"] == "medium"

    @pytest.mark.asyncio
    async def test_constructor_override_false_via_openai(self, mock_grok_client):
        """Constructor override must force the fallback structured-output path."""
        mock_grok_client.chat.completions.create.return_value = (
            self._make_mock_response('{"answer": "42"}')
        )

        provider = GrokLLMProvider(
            api_key="sk-test",
            supports_structured_outputs=False,
            reasoning_effort="low",
        )
        assert provider._supports_structured_outputs is False
        await provider.complete_structured(
            "What is the answer?",
            json_schema=self.SCHEMA,
        )

        call_kwargs = mock_grok_client.chat.completions.create.call_args[1]
        response_format = call_kwargs.get("response_format")
        assert response_format == {"type": "json_object"}, (
            f"Expected json_object response_format, got: {response_format}"
        )
        assert call_kwargs["reasoning_effort"] == "low"
