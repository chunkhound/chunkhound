"""High-value functional tests for GrokLLMProvider (chat completions path)."""

import pytest
from unittest.mock import AsyncMock, patch

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.providers.llm.grok_llm_provider import GrokLLMProvider
from chunkhound.interfaces.llm_provider import LLMResponse


@pytest.fixture
def mock_grok_client():
    with patch("chunkhound.providers.llm.openai_compatible_provider.AsyncOpenAI") as mock:
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
        provider = GrokLLMProvider(api_key="gsk-test", model="grok-beta")
        mock_grok_client.chat.completions.create.return_value = AsyncMock(
            choices=[AsyncMock(message=AsyncMock(content="ok"))],
            usage=AsyncMock(total_tokens=5)
        )

        await provider.complete("hi", max_completion_tokens=200)

        call = mock_grok_client.chat.completions.create.call_args[1]
        assert call["model"] == "grok-beta"
        assert call["max_completion_tokens"] == 200

    @pytest.mark.asyncio
    async def test_errors_propagate(self, mock_grok_client):
        """Errors must not be swallowed."""
        mock_grok_client.chat.completions.create.side_effect = Exception("xAI outage")

        provider = GrokLLMProvider(api_key="gsk-test")
        with pytest.raises(RuntimeError) as exc:
            await provider.complete("fail")

        assert "LLM completion failed" in str(exc.value)

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
            api_key="test-key",
            model="grok-beta",
            base_url="https://custom.api.x.ai/v1"
        )
        assert str(provider._client.base_url) == "https://custom.api.x.ai/v1/"

