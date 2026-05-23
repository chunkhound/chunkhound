"""High-value functional tests for DeepSeekLLMProvider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chunkhound.providers.llm.deepseek_llm_provider import DeepSeekLLMProvider
from chunkhound.interfaces.llm_provider import LLMResponse


@pytest.fixture
def mock_deepseek_client():
    with patch(
        "chunkhound.providers.llm.openai_compatible_provider.AsyncOpenAI"
    ) as mock:
        client = mock.return_value
        client.chat.completions.create = AsyncMock()
        yield client


@pytest.fixture
def provider():
    """Create a DeepSeekLLMProvider instance for testing."""
    return DeepSeekLLMProvider(
        api_key="sk-deepseek-test",
        model="deepseek-v4-flash",
        timeout=60,
        max_retries=3,
    )


class TestDeepSeekLLMProvider:
    """DeepSeek uses the OpenAI-compatible base with structured outputs disabled."""

    @pytest.mark.asyncio
    async def test_complete_returns_llmresponse(self, mock_deepseek_client):
        """Core: must return LLMResponse (same contract as every other provider)."""
        mock_resp = AsyncMock()
        mock_resp.choices = [AsyncMock(message=AsyncMock(content="DeepSeek is ready"))]
        mock_resp.usage = AsyncMock(total_tokens=15)
        mock_resp.choices[0].finish_reason = "stop"
        mock_deepseek_client.chat.completions.create.return_value = mock_resp

        provider = DeepSeekLLMProvider(api_key="sk-deepseek-test")
        response = await provider.complete("Say hello")

        assert isinstance(response, LLMResponse)
        assert response.content == "DeepSeek is ready"
        assert response.tokens_used == 15
        assert response.model == "deepseek-v4-flash"

    @pytest.mark.asyncio
    async def test_uses_max_tokens_not_max_completion_tokens(
        self, mock_deepseek_client
    ):
        """Critical contract: DeepSeek API expects max_tokens, not max_completion_tokens."""
        mock_deepseek_client.chat.completions.create.return_value = AsyncMock(
            choices=[AsyncMock(message=AsyncMock(content="ok"))],
            usage=AsyncMock(total_tokens=5),
        )

        provider = DeepSeekLLMProvider(api_key="sk-deepseek-test")
        await provider.complete("hi", max_completion_tokens=200)

        call = mock_deepseek_client.chat.completions.create.call_args[1]
        assert "max_tokens" in call
        assert call["max_tokens"] == 200
        assert "max_completion_tokens" not in call

    @pytest.mark.asyncio
    async def test_structured_fallback_uses_prompt_injection(self, mock_deepseek_client):
        """DeepSeek must not send json_schema response_format; use prompt fallback."""
        mock_resp = AsyncMock()
        mock_resp.choices = [AsyncMock(message=AsyncMock(content='{"answer": "42"}'))]
        mock_resp.usage = AsyncMock(total_tokens=10)
        mock_resp.choices[0].finish_reason = "stop"
        mock_deepseek_client.chat.completions.create.return_value = mock_resp

        provider = DeepSeekLLMProvider(api_key="sk-deepseek-test")
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }

        await provider.complete_structured("What is the answer?", schema)

        call = mock_deepseek_client.chat.completions.create.call_args[1]
        assert call.get("response_format") == {"type": "json_object"}
        assert "json_schema" not in str(call.get("response_format", {}))
        # Schema should be injected into the system prompt
        messages = call["messages"]
        system_msg = messages[0]["content"]
        assert '"answer"' in system_msg

    @pytest.mark.asyncio
    async def test_structured_fallback_with_explicit_system_prompt(self, mock_deepseek_client):
        """Prompt fallback must preserve existing system instructions."""
        mock_resp = AsyncMock()
        mock_resp.choices = [AsyncMock(message=AsyncMock(content='{"answer": "42"}'))]
        mock_resp.usage = AsyncMock(total_tokens=10)
        mock_resp.choices[0].finish_reason = "stop"
        mock_deepseek_client.chat.completions.create.return_value = mock_resp

        provider = DeepSeekLLMProvider(api_key="sk-deepseek-test")
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }

        await provider.complete_structured(
            "What is the answer?", schema, system="You are a helpful assistant."
        )

        call = mock_deepseek_client.chat.completions.create.call_args[1]
        messages = call["messages"]
        system_msg = messages[0]["content"]
        assert "You are a helpful assistant." in system_msg
        assert '"answer"' in system_msg

    def test_get_synthesis_concurrency(self, provider):
        """DeepSeek concurrency should be 10 for developer use."""
        assert provider.get_synthesis_concurrency() == 10

    def test_default_base_url(self, provider):
        """Default base URL must point to DeepSeek API."""
        assert str(provider._client.base_url).rstrip("/") == "https://api.deepseek.com"

    def test_provider_name(self, provider):
        assert provider.name == "deepseek"

    def test_class_level_structured_outputs_disabled(self):
        """Class-level flag must default to False so config-less instances opt out."""
        provider = DeepSeekLLMProvider(api_key="sk-test")
        assert provider._supports_structured_outputs is False

    def test_constructor_override_true(self):
        """User must be able to force-enable native structured outputs if desired."""
        provider = DeepSeekLLMProvider(
            api_key="sk-test",
            supports_structured_outputs=True,
        )
        assert provider._supports_structured_outputs is True

    @pytest.mark.asyncio
    async def test_structured_native_opt_in_uses_json_schema(
        self, mock_deepseek_client
    ):
        mock_resp = AsyncMock()
        mock_resp.choices = [AsyncMock(message=AsyncMock(content='{"answer": "42"}'))]
        mock_resp.usage = AsyncMock(total_tokens=10)
        mock_resp.choices[0].finish_reason = "stop"
        mock_deepseek_client.chat.completions.create.return_value = mock_resp

        provider = DeepSeekLLMProvider(
            api_key="sk-test",
            supports_structured_outputs=True,
        )
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }

        result = await provider.complete_structured("What is the answer?", schema)

        assert result == {"answer": "42"}
        call = mock_deepseek_client.chat.completions.create.call_args[1]
        assert call["response_format"]["type"] == "json_schema"

    @pytest.mark.asyncio
    async def test_errors_propagate(self, mock_deepseek_client):
        """Errors must not be swallowed."""
        mock_deepseek_client.chat.completions.create.side_effect = Exception(
            "DeepSeek API error"
        )

        provider = DeepSeekLLMProvider(api_key="sk-deepseek-test")
        with pytest.raises(RuntimeError) as exc:
            await provider.complete("fail")

        assert "LLM completion failed" in str(exc.value)
