"""High-value functional tests for OpenAILLMProvider (Responses API path)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.interfaces.llm_provider import LLMResponse
from chunkhound.providers.llm.openai_llm_provider import OpenAILLMProvider


@pytest.fixture
def mock_openai_client():
    with patch("chunkhound.providers.llm.openai_compatible_provider.AsyncOpenAI") as mock:
        client = mock.return_value
        client.responses.create = AsyncMock()      # Responses API (default path)
        client.chat.completions.create = AsyncMock()  # fallback for older models
        yield client


def test_default_timeout():
    """Default timeout resolves to 120."""

    p = OpenAILLMProvider(api_key="test-key")
    assert p.timeout == DEFAULT_LLM_TIMEOUT


def test_custom_endpoint_without_api_key_uses_placeholder_key():
    """Custom endpoints should construct an SDK client without a real API key."""
    with patch(
        "chunkhound.providers.llm.openai_compatible_provider.AsyncOpenAI"
    ) as mock_client:
        OpenAILLMProvider(
            api_key=None,
            model="llama3.2",
            base_url="http://localhost:11434/v1",
        )

    kwargs = mock_client.call_args.kwargs
    assert kwargs["api_key"] == "not-required"
    assert kwargs["base_url"] == "http://localhost:11434/v1"


def test_official_openai_endpoint_keeps_real_api_key_contract():
    """Official OpenAI endpoints must not use the custom-endpoint placeholder key."""
    with patch(
        "chunkhound.providers.llm.openai_compatible_provider.AsyncOpenAI"
    ) as mock_client:
        OpenAILLMProvider(
            api_key="sk-real",
            model="gpt-5",
            base_url="https://api.openai.com/v1",
        )

    kwargs = mock_client.call_args.kwargs
    assert kwargs["api_key"] == "sk-real"
    assert kwargs["base_url"] == "https://api.openai.com/v1"


def test_custom_endpoint_ssl_verify_false_creates_insecure_http_client():
    """Explicit ssl_verify=false should only affect custom base_url traffic."""
    with (
        patch("chunkhound.providers.llm.openai_compatible_provider.AsyncOpenAI") as mock_client,
        patch("chunkhound.providers.llm.openai_compatible_provider.httpx.AsyncClient") as mock_http_client,
    ):
        OpenAILLMProvider(
            api_key=None,
            model="llama3.2",
            base_url="https://localhost:11434/v1",
            ssl_verify=False,
        )

    kwargs = mock_client.call_args.kwargs
    assert kwargs["base_url"] == "https://localhost:11434/v1"
    assert kwargs["http_client"] == mock_http_client.return_value
    assert mock_http_client.call_args.kwargs["verify"] is False


def test_ssl_verify_is_ignored_without_llm_base_url():
    """ssl_verify must not affect default endpoint routing when base_url is unset."""
    with patch(
        "chunkhound.providers.llm.openai_compatible_provider.AsyncOpenAI"
    ) as mock_client:
        OpenAILLMProvider(
            api_key="sk-real",
            model="gpt-5",
            ssl_verify=False,
        )

    kwargs = mock_client.call_args.kwargs
    assert "http_client" not in kwargs


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

    @pytest.mark.asyncio
    async def test_internal_runtime_error_not_double_wrapped_chat_completions(self, mock_openai_client):
        """RuntimeError from internal checks must pass through unwrapped (Chat Completions path)."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = None
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage = MagicMock(total_tokens=5, prompt_tokens=3, completion_tokens=2)
        mock_openai_client.chat.completions.create.return_value = mock_resp

        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-3.5-turbo")
        with pytest.raises(RuntimeError) as exc:
            await provider.complete("test")

        msg = str(exc.value)
        assert "LLM returned empty response" in msg
        assert "LLM completion failed" not in msg

    @pytest.mark.asyncio
    async def test_internal_runtime_error_not_double_wrapped_responses_api(self, mock_openai_client):
        """RuntimeError from internal checks must pass through unwrapped (Responses API path)."""
        mock_resp = MagicMock()
        mock_resp.output = []  # content_parts stays empty → content = None
        mock_resp.usage = MagicMock(total_tokens=5, input_tokens=3, output_tokens=2)
        mock_resp.status = "completed"
        mock_openai_client.responses.create.return_value = mock_resp

        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-4o")
        with pytest.raises(RuntimeError) as exc:
            await provider.complete("test")

        msg = str(exc.value)
        assert "LLM returned empty response" in msg
        assert "LLM completion failed" not in msg

    @pytest.mark.asyncio
    async def test_internal_runtime_error_not_double_wrapped_complete_structured(self, mock_openai_client):
        """RuntimeError from internal checks must pass through unwrapped (complete_structured path)."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = None
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage = MagicMock(total_tokens=5, prompt_tokens=3, completion_tokens=2)
        mock_openai_client.chat.completions.create.return_value = mock_resp

        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-3.5-turbo")
        with pytest.raises(RuntimeError) as exc:
            await provider.complete_structured("test", json_schema={"type": "object"})

        msg = str(exc.value)
        assert "LLM structured completion returned empty response" in msg
        assert "LLM structured completion failed" not in msg

    @pytest.mark.asyncio
    async def test_internal_runtime_error_not_double_wrapped_complete_structured_responses_api(
        self, mock_openai_client
    ):
        """RuntimeError from _complete_structured_with_responses_api must pass through unwrapped."""
        mock_resp = MagicMock()
        mock_resp.output = []   # empty output list → content_parts=[], raises RuntimeError
        mock_resp.status = "completed"
        mock_resp.usage = MagicMock(total_tokens=5, input_tokens=3, output_tokens=2)
        mock_openai_client.responses.create.return_value = mock_resp

        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-4o")
        with pytest.raises(RuntimeError) as exc:
            await provider.complete_structured("test", json_schema={"type": "object"})

        msg = str(exc.value)
        assert "LLM structured completion returned empty response" in msg
        assert "LLM structured completion failed" not in msg
