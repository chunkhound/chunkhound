"""High-value functional tests for OpenAILLMProvider."""

from unittest.mock import AsyncMock, patch

import pytest

from chunkhound.interfaces.llm_provider import LLMResponse
from chunkhound.providers.llm.openai_llm_provider import OpenAILLMProvider


@pytest.fixture
def mock_openai_client():
    with patch(
        "chunkhound.providers.llm.openai_compatible_provider.AsyncOpenAI"
    ) as mock:
        client = mock.return_value
        client.responses.create = AsyncMock()  # Responses API (default path)
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
                content=[
                    AsyncMock(
                        type="output_text",
                        text="Chunking is working perfectly!",
                    )
                ],
            )
        ]
        mock_resp.usage = AsyncMock(total_tokens=42)
        mock_resp.status = "completed"
        mock_openai_client.responses.create.return_value = mock_resp

        # default = gpt-5-nano-mini → Responses
        provider = OpenAILLMProvider(api_key="sk-test")
        response = await provider.complete("Explain chunking")

        assert isinstance(response, LLMResponse)
        assert response.content == "Chunking is working perfectly!"
        assert response.tokens_used == 42
        assert response.model == "gpt-5-nano-mini"

    @pytest.mark.asyncio
    async def test_configuration_is_respected_in_api_call(self, mock_openai_client):
        """Model, token, effort, and timeout settings must reach the API."""
        provider = OpenAILLMProvider(
            api_key="sk-test",
            model="gpt-4o",
            reasoning_effort="low",
            timeout=30,
        )
        mock_openai_client.responses.create.return_value = AsyncMock(
            output=[
                AsyncMock(
                    type="message",
                    content=[AsyncMock(type="output_text", text="ok")],
                )
            ],
            usage=AsyncMock(total_tokens=10),
            status="completed",
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
    async def test_structured_native_responses_path_sends_json_schema_payload(
        self, mock_openai_client
    ):
        """Responses API structured path must send native json_schema payload."""
        mock_resp = AsyncMock()
        mock_resp.output = [
            AsyncMock(
                type="message",
                content=[AsyncMock(type="output_text", text='{"answer": "42"}')],
            )
        ]
        mock_resp.usage = AsyncMock(input_tokens=10, output_tokens=20, total_tokens=30)
        mock_resp.status = "completed"
        mock_openai_client.responses.create.return_value = mock_resp

        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-5")
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }

        result = await provider.complete_structured("What is the answer?", schema)

        assert result == {"answer": "42"}
        call = mock_openai_client.responses.create.call_args[1]
        assert call["text"]["format"]["type"] == "json_schema"
        assert call["text"]["format"]["name"] == "structured_response"
        assert call["text"]["format"]["strict"] is True
        assert call["text"]["format"]["schema"] == schema

    @pytest.mark.asyncio
    async def test_structured_opt_out_uses_prompt_fallback_without_native_schema(
        self, mock_openai_client
    ):
        """GPT-5 models must honor opt-out without losing Responses API routing."""
        mock_resp = AsyncMock()
        mock_resp.output = [
            AsyncMock(
                type="message",
                content=[AsyncMock(type="output_text", text='{"answer": "42"}')],
            )
        ]
        mock_resp.usage = AsyncMock(input_tokens=10, output_tokens=20, total_tokens=30)
        mock_resp.status = "completed"
        mock_openai_client.responses.create.return_value = mock_resp

        provider = OpenAILLMProvider(
            api_key="sk-test",
            model="gpt-5",
            supports_structured_outputs=False,
        )
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }

        result = await provider.complete_structured("What is the answer?", schema)

        assert result == {"answer": "42"}
        mock_openai_client.chat.completions.create.assert_not_called()

        call = mock_openai_client.responses.create.call_args[1]
        assert "text" not in call
        assert '"answer"' in call["instructions"]

    @pytest.mark.asyncio
    async def test_structured_opt_out_keeps_responses_api_for_responses_only_models(
        self, mock_openai_client
    ):
        """Responses-only models must not fall back to chat completions."""
        mock_resp = AsyncMock()
        mock_resp.output = [
            AsyncMock(
                type="message",
                content=[AsyncMock(type="output_text", text='{"answer": "42"}')],
            )
        ]
        mock_resp.usage = AsyncMock(input_tokens=10, output_tokens=20, total_tokens=30)
        mock_resp.status = "completed"
        mock_openai_client.responses.create.return_value = mock_resp

        provider = OpenAILLMProvider(
            api_key="sk-test",
            model="gpt-5-pro",
            supports_structured_outputs=False,
        )
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }

        result = await provider.complete_structured("What is the answer?", schema)

        assert result == {"answer": "42"}
        mock_openai_client.chat.completions.create.assert_not_called()

        call = mock_openai_client.responses.create.call_args[1]
        assert "text" not in call
        assert '"answer"' in call["instructions"]

    @pytest.mark.asyncio
    async def test_structured_opt_out_empty_response_not_double_wrapped(
        self, mock_openai_client
    ):
        """Empty-response RuntimeError must not be double-wrapped."""
        mock_resp = AsyncMock()
        mock_resp.output = [
            AsyncMock(
                type="message",
                content=[AsyncMock(type="output_text", text="")],
            )
        ]
        mock_resp.usage = AsyncMock(input_tokens=10, output_tokens=0, total_tokens=10)
        mock_resp.status = "completed"
        mock_openai_client.responses.create.return_value = mock_resp

        provider = OpenAILLMProvider(
            api_key="sk-test",
            model="gpt-5",
            supports_structured_outputs=False,
        )
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }

        with pytest.raises(RuntimeError) as exc:
            await provider.complete_structured("What is the answer?", schema)

        msg = str(exc.value)
        assert "empty response" in msg
        assert not msg.startswith("LLM structured completion failed")

    @pytest.mark.asyncio
    async def test_structured_opt_out_incomplete_response_beats_empty_response(
        self, mock_openai_client
    ):
        """Incomplete Responses API status must preserve token-limit diagnostics."""
        mock_resp = AsyncMock()
        mock_resp.output = [
            AsyncMock(
                type="message",
                content=[AsyncMock(type="output_text", text="")],
            )
        ]
        mock_resp.usage = AsyncMock(input_tokens=123, output_tokens=0, total_tokens=123)
        mock_resp.status = "incomplete"
        mock_openai_client.responses.create.return_value = mock_resp

        provider = OpenAILLMProvider(
            api_key="sk-test",
            model="gpt-5",
            supports_structured_outputs=False,
        )
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }

        with pytest.raises(RuntimeError) as exc:
            await provider.complete_structured("What is the answer?", schema)

        msg = str(exc.value)
        assert "incomplete" in msg
        assert "token limit exceeded" in msg
        assert "empty response" not in msg

    @pytest.mark.asyncio
    async def test_chat_completions_path_for_non_responses_model(
        self, mock_openai_client
    ):
        """Non-responses models must route to Chat Completions via parent class."""
        mock_resp = AsyncMock()
        mock_resp.choices = [AsyncMock(message=AsyncMock(content="Chat response"))]
        mock_resp.usage = AsyncMock(total_tokens=12)
        mock_resp.choices[0].finish_reason = "stop"
        mock_openai_client.chat.completions.create.return_value = mock_resp

        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-3.5-turbo")
        response = await provider.complete("Hello")

        assert isinstance(response, LLMResponse)
        assert response.content == "Chat response"
        mock_openai_client.chat.completions.create.assert_called_once()
        mock_openai_client.responses.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_completions_structured_path_for_non_responses_model(
        self, mock_openai_client
    ):
        """Structured completion for non-responses models must use Chat Completions."""
        mock_resp = AsyncMock()
        mock_resp.choices = [AsyncMock(message=AsyncMock(content='{"result": "ok"}'))]
        mock_resp.usage = AsyncMock(total_tokens=8)
        mock_resp.choices[0].finish_reason = "stop"
        mock_openai_client.chat.completions.create.return_value = mock_resp

        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-3.5-turbo")
        schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        }

        result = await provider.complete_structured("Test", schema)

        assert result == {"result": "ok"}
        mock_openai_client.chat.completions.create.assert_called_once()
        mock_openai_client.responses.create.assert_not_called()

        call = mock_openai_client.chat.completions.create.call_args[1]
        assert call["response_format"]["type"] == "json_schema"

    @pytest.mark.asyncio
    async def test_chat_completions_structured_truncation_error_wins(
        self, mock_openai_client
    ):
        """Truncation must win over generic empty-content errors."""
        mock_resp = AsyncMock()
        mock_resp.choices = [AsyncMock(message=AsyncMock(content=""))]
        mock_resp.usage = AsyncMock(
            prompt_tokens=123,
            completion_tokens=0,
            total_tokens=123,
        )
        mock_resp.choices[0].finish_reason = "length"
        mock_openai_client.chat.completions.create.return_value = mock_resp

        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-3.5-turbo")
        schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        }

        with pytest.raises(RuntimeError, match="token limit exceeded") as exc:
            await provider.complete_structured("Test", schema)

        assert "empty response" not in str(exc.value)

    @pytest.mark.asyncio
    async def test_chat_completions_structured_empty_response_error(
        self, mock_openai_client
    ):
        """Empty non-truncated structured output must fail clearly."""
        mock_resp = AsyncMock()
        mock_resp.choices = [AsyncMock(message=AsyncMock(content="   "))]
        mock_resp.usage = AsyncMock(
            prompt_tokens=10,
            completion_tokens=0,
            total_tokens=10,
        )
        mock_resp.choices[0].finish_reason = "stop"
        mock_openai_client.chat.completions.create.return_value = mock_resp

        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-3.5-turbo")
        schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        }

        with pytest.raises(RuntimeError, match="empty response"):
            await provider.complete_structured("Test", schema)

    @pytest.mark.asyncio
    async def test_chat_completions_path_uses_max_completion_tokens(
        self, mock_openai_client
    ):
        """Chat Completions must use max_completion_tokens."""
        mock_resp = AsyncMock()
        mock_resp.choices = [AsyncMock(message=AsyncMock(content="ok"))]
        mock_resp.usage = AsyncMock(total_tokens=5)
        mock_resp.choices[0].finish_reason = "stop"
        mock_openai_client.chat.completions.create.return_value = mock_resp

        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-3.5-turbo")
        await provider.complete("hi", max_completion_tokens=250)

        call = mock_openai_client.chat.completions.create.call_args[1]
        assert call["max_completion_tokens"] == 250
        assert "max_output_tokens" not in call
