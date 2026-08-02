"""High-value functional tests for OpenAILLMProvider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.interfaces.llm_provider import (
    PROVIDER_MANAGED_OUTPUT,
    LLMResponse,
    OutputLimitCapability,
)
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


@pytest.mark.parametrize(
    ("explicit_url", "environment_url", "resolved_url", "expected_capability"),
    [
        (None, None, None, OutputLimitCapability.SUPPORTED),
        (
            None,
            "https://api.openai.com/v1",
            "https://api.openai.com/v1",
            OutputLimitCapability.SUPPORTED,
        ),
        (
            None,
            "https://gateway.example/v1",
            "https://gateway.example/v1",
            OutputLimitCapability.UNKNOWN,
        ),
        (
            "https://api.openai.com/v1",
            "https://gateway.example/v1",
            "https://api.openai.com/v1",
            OutputLimitCapability.SUPPORTED,
        ),
        (
            "https://gateway.example/v1",
            "https://api.openai.com/v1",
            "https://gateway.example/v1",
            OutputLimitCapability.UNKNOWN,
        ),
    ],
    ids=[
        "sdk-default",
        "official-environment",
        "custom-environment",
        "explicit-official-precedence",
        "explicit-custom-precedence",
    ],
)
def test_output_limit_capability_uses_resolved_endpoint_route(
    explicit_url: str | None,
    environment_url: str | None,
    resolved_url: str | None,
    expected_capability: OutputLimitCapability,
    monkeypatch: pytest.MonkeyPatch,
    clean_environment,
) -> None:
    """Capability classification and SDK construction use one selected route."""
    if environment_url is not None:
        monkeypatch.setenv("OPENAI_BASE_URL", environment_url)

    with patch(
        "chunkhound.providers.llm.openai_compatible_provider.AsyncOpenAI"
    ) as mock_client:
        provider = OpenAILLMProvider(
            api_key="sk-test",
            model="gpt-5",
            base_url=explicit_url,
        )

    assert provider.output_limit_metadata.omission is expected_capability
    if resolved_url is None:
        assert "base_url" not in mock_client.call_args.kwargs
    else:
        assert mock_client.call_args.kwargs["base_url"] == resolved_url


def test_custom_endpoint_ssl_verify_false_creates_insecure_http_client():
    """Explicit ssl_verify=false should only affect custom base_url traffic."""
    with (
        patch(
            "chunkhound.providers.llm.openai_compatible_provider.AsyncOpenAI"
        ) as mock_client,
        patch(
            "chunkhound.providers.llm.openai_compatible_provider.httpx.AsyncClient"
        ) as mock_http_client,
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
    async def test_provider_managed_text_omits_caps_on_responses_and_chat(
        self, mock_openai_client
    ):
        """Official OpenAI omits, rather than serializing enum/None cap values."""
        responses_result = MagicMock()
        responses_result.output = [
            MagicMock(
                type="message", content=[MagicMock(type="output_text", text="ok")]
            )
        ]
        responses_result.usage = None
        responses_result.status = "completed"
        mock_openai_client.responses.create.return_value = responses_result

        chat_result = MagicMock()
        chat_result.choices = [
            MagicMock(message=MagicMock(content="ok"), finish_reason="stop")
        ]
        chat_result.usage = None
        mock_openai_client.chat.completions.create.return_value = chat_result

        responses_provider = OpenAILLMProvider(api_key="sk-test", model="gpt-5")
        chat_provider = OpenAILLMProvider(api_key="sk-test", model="gpt-3.5-turbo")
        for provider in (responses_provider, chat_provider):
            provider.configure_synthesis_output_limit_policy(
                output_limits_enabled=False,
                fallback_tokens=75_000,
            )
            await provider.complete(
                "hello", max_completion_tokens=PROVIDER_MANAGED_OUTPUT
            )

        assert (
            "max_output_tokens"
            not in mock_openai_client.responses.create.call_args.kwargs
        )
        assert (
            "max_completion_tokens"
            not in mock_openai_client.chat.completions.create.call_args.kwargs
        )

    @pytest.mark.asyncio
    async def test_provider_managed_structured_omits_caps_and_preserves_schema(
        self, mock_openai_client
    ):
        """Both OpenAI APIs retain strict schemas while omitting managed caps."""
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        responses_result = MagicMock()
        responses_result.output = [
            MagicMock(
                type="message",
                content=[MagicMock(type="output_text", text='{"answer":"42"}')],
            )
        ]
        responses_result.usage = None
        responses_result.status = "completed"
        mock_openai_client.responses.create.return_value = responses_result

        chat_result = MagicMock()
        chat_result.choices = [
            MagicMock(
                message=MagicMock(content='{"answer":"42"}'),
                finish_reason="stop",
            )
        ]
        chat_result.usage = None
        mock_openai_client.chat.completions.create.return_value = chat_result

        responses_provider = OpenAILLMProvider(api_key="sk-test", model="gpt-5")
        chat_provider = OpenAILLMProvider(api_key="sk-test", model="gpt-3.5-turbo")
        for provider in (responses_provider, chat_provider):
            provider.configure_synthesis_output_limit_policy(
                output_limits_enabled=False,
                fallback_tokens=75_000,
            )
            result = await provider.complete_structured(
                "hello",
                schema,
                max_completion_tokens=PROVIDER_MANAGED_OUTPUT,
            )
            assert result == {"answer": "42"}

        responses_call = mock_openai_client.responses.create.call_args.kwargs
        chat_call = mock_openai_client.chat.completions.create.call_args.kwargs
        assert "max_output_tokens" not in responses_call
        assert responses_call["text"]["format"]["schema"] == schema
        assert "max_completion_tokens" not in chat_call
        assert chat_call["response_format"]["json_schema"]["schema"] == schema

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model", "cap_field"),
        [
            ("gpt-5", "max_output_tokens"),
            ("gpt-3.5-turbo", "max_completion_tokens"),
        ],
        ids=["responses", "chat"],
    )
    async def test_custom_environment_provider_managed_uses_fallback(
        self,
        model: str,
        cap_field: str,
        mock_openai_client,
        monkeypatch: pytest.MonkeyPatch,
        clean_environment,
    ) -> None:
        """A custom environment route sends fallback caps in both API dialects."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

        responses_result = MagicMock()
        responses_result.output = [
            MagicMock(
                type="message", content=[MagicMock(type="output_text", text="ok")]
            )
        ]
        responses_result.usage = None
        responses_result.status = "completed"
        mock_openai_client.responses.create.return_value = responses_result

        chat_result = MagicMock()
        chat_result.choices = [
            MagicMock(message=MagicMock(content="ok"), finish_reason="stop")
        ]
        chat_result.usage = None
        mock_openai_client.chat.completions.create.return_value = chat_result

        provider = OpenAILLMProvider(api_key="sk-test", model=model)
        provider.configure_synthesis_output_limit_policy(
            output_limits_enabled=False,
            fallback_tokens=75_123,
        )

        await provider.complete("hello", max_completion_tokens=PROVIDER_MANAGED_OUTPUT)

        create = (
            mock_openai_client.responses.create
            if cap_field == "max_output_tokens"
            else mock_openai_client.chat.completions.create
        )
        assert provider.output_limit_metadata.omission is OutputLimitCapability.UNKNOWN
        assert create.call_args.kwargs[cap_field] == 75_123

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
    @pytest.mark.parametrize(
        "incomplete_reason",
        ["max_output_tokens", "content_filter", "other_reason"],
        ids=["token-limit", "content-filter", "other"],
    )
    async def test_structured_opt_out_incomplete_response_beats_empty_response(
        self, mock_openai_client, incomplete_reason
    ):
        """Every incomplete Responses status must retain its provider reason."""
        mock_resp = AsyncMock()
        mock_resp.output = [
            AsyncMock(
                type="message",
                content=[AsyncMock(type="output_text", text="")],
            )
        ]
        mock_resp.usage = AsyncMock(input_tokens=123, output_tokens=0, total_tokens=123)
        mock_resp.status = "incomplete"
        mock_resp.incomplete_details = AsyncMock(reason=incomplete_reason)
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
        if incomplete_reason == "max_output_tokens":
            assert "token limit exceeded" in msg
        else:
            assert incomplete_reason in msg
            assert "token limit exceeded" not in msg
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
    async def test_chat_completions_truncation_error_wins_over_empty_response(
        self, mock_openai_client
    ):
        """Chat Completions truncation must beat generic empty-response errors."""
        mock_resp = AsyncMock()
        mock_resp.choices = [AsyncMock(message=AsyncMock(content="   "))]
        mock_resp.usage = AsyncMock(
            prompt_tokens=123,
            completion_tokens=0,
            total_tokens=123,
        )
        mock_resp.choices[0].finish_reason = "length"
        mock_openai_client.chat.completions.create.return_value = mock_resp

        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-3.5-turbo")

        with pytest.raises(RuntimeError, match="token limit exceeded") as exc:
            await provider.complete("Hello")

        assert "empty response" not in str(exc.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "incomplete_reason",
        ["max_output_tokens", "content_filter", "other_reason"],
        ids=["token-limit", "content-filter", "other"],
    )
    async def test_responses_api_incomplete_error_wins_over_empty_response(
        self, mock_openai_client, incomplete_reason
    ):
        """Every incomplete Responses status must retain its provider reason."""
        mock_resp = AsyncMock()
        mock_resp.output = [
            AsyncMock(
                type="message",
                content=[AsyncMock(type="output_text", text="")],
            )
        ]
        mock_resp.usage = AsyncMock(input_tokens=123, output_tokens=0, total_tokens=123)
        mock_resp.status = "incomplete"
        mock_resp.incomplete_details = AsyncMock(reason=incomplete_reason)
        mock_openai_client.responses.create.return_value = mock_resp

        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-5")

        with pytest.raises(RuntimeError) as exc:
            await provider.complete("Hello")

        msg = str(exc.value)
        if incomplete_reason == "max_output_tokens":
            assert "token limit exceeded" in msg
        else:
            assert incomplete_reason in msg
            assert "token limit exceeded" not in msg
        assert "empty response" not in msg

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

    async def test_internal_runtime_error_not_double_wrapped_chat_completions(
        self, mock_openai_client
    ):
        """Chat Completions internal RuntimeError must pass through unwrapped."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = None
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage = MagicMock(
            total_tokens=5, prompt_tokens=3, completion_tokens=2
        )
        mock_openai_client.chat.completions.create.return_value = mock_resp

        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-3.5-turbo")
        with pytest.raises(RuntimeError) as exc:
            await provider.complete("test")

        msg = str(exc.value)
        assert "LLM returned empty response" in msg
        assert "LLM completion failed" not in msg

    @pytest.mark.asyncio
    async def test_internal_runtime_error_not_double_wrapped_responses_api(
        self, mock_openai_client
    ):
        """Responses API internal RuntimeError must pass through unwrapped."""
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
    async def test_internal_runtime_error_not_double_wrapped_complete_structured(
        self, mock_openai_client
    ):
        """Structured internal RuntimeError must pass through unwrapped."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = None
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage = MagicMock(
            total_tokens=5, prompt_tokens=3, completion_tokens=2
        )
        mock_openai_client.chat.completions.create.return_value = mock_resp

        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-3.5-turbo")
        with pytest.raises(RuntimeError) as exc:
            await provider.complete_structured("test", json_schema={"type": "object"})

        msg = str(exc.value)
        assert "LLM structured completion returned empty response" in msg
        assert "LLM structured completion failed" not in msg

    @pytest.mark.asyncio
    async def test_responses_structured_runtime_error_not_double_wrapped(
        self, mock_openai_client
    ):
        """Responses structured RuntimeError must pass through unwrapped."""
        mock_resp = MagicMock()
        mock_resp.output = []  # Empty output triggers the internal RuntimeError.
        mock_resp.status = "completed"
        mock_resp.usage = MagicMock(total_tokens=5, input_tokens=3, output_tokens=2)
        mock_openai_client.responses.create.return_value = mock_resp

        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-4o")
        with pytest.raises(RuntimeError) as exc:
            await provider.complete_structured("test", json_schema={"type": "object"})

        msg = str(exc.value)
        assert "LLM structured output returned empty response" in msg
        assert "LLM structured completion failed" not in msg
