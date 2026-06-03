"""Contract tests for Gemini LLM provider.

Tests verify external constraints and user-facing contracts: error wrapping,
thinking param forwarding, usage tracking, health check shape, and response
parsing. Model names are test-only placeholders — no real model required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from google.genai import types

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.interfaces.llm_provider import LLMResponse
from chunkhound.providers.llm.gemini_llm_provider import GeminiLLMProvider

# ---------------------------------------------------------------------------
# Fixtures – mock at the SDK import boundary, not post-construction
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_genai_client():
    """Patch genai.Client so no real SDK constructor or HTTP calls are made."""
    with patch(
        "chunkhound.providers.llm.gemini_llm_provider.genai.Client"
    ) as mock:
        yield mock


def _make_aclient(mock_client: MagicMock) -> MagicMock:
    """Return the async context manager's inner aclient.

    Assumes the fixture has already been set up.
    """
    return mock_client.return_value.aio.__aenter__.return_value


def _make_resp(
    text: str,
    finish_reason="STOP",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> MagicMock:
    """Build a mock SDK response resembling a real generate_content result."""
    response = MagicMock(spec=["text", "usage_metadata", "candidates"])
    response.text = text
    usage = MagicMock(spec=[
        "prompt_token_count", "candidates_token_count", "total_token_count",
    ])
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = completion_tokens
    usage.total_token_count = prompt_tokens + completion_tokens
    response.usage_metadata = usage
    candidate = MagicMock(spec=["finish_reason"])
    candidate.finish_reason = finish_reason
    response.candidates = [candidate]
    return response


# ---------------------------------------------------------------------------
# Construction & config
# ---------------------------------------------------------------------------


class TestConstruction:
    """Provider construction contracts."""

    def test_api_key_required(self, mock_genai_client):
        """API key must be provided — empty/None must fail."""
        with pytest.raises(ValueError, match="API key required"):
            GeminiLLMProvider(api_key=None)
        with pytest.raises(ValueError, match="API key required"):
            GeminiLLMProvider(api_key="")

    def test_any_model_name_accepted(self, mock_genai_client):
        """Model name is passed through — no validation, no interpretation."""
        for name in (
            "test-model",
            "gemini-2.5-pro",
            "gemini-3.5-flash",
            "future-model-v42",
        ):
            p = GeminiLLMProvider(api_key="test-key", model=name)
            assert p.model == name, f"model name '{name}' not preserved"

    def test_timeout_default(self, mock_genai_client):
        """Default timeout is the project standard."""
        p = GeminiLLMProvider(api_key="test-key")
        assert p.timeout == DEFAULT_LLM_TIMEOUT

    def test_synthesis_concurrency(self, mock_genai_client):
        """Conservative concurrency for Gemini rate limits."""
        p = GeminiLLMProvider(api_key="test-key")
        assert p.get_synthesis_concurrency() == 2


# ---------------------------------------------------------------------------
# Thinking parameter forwarding
# ---------------------------------------------------------------------------


class TestThinkingConfig:
    """Forwarding of thinking params to the SDK — no model-name detection."""

    @pytest.mark.asyncio
    async def test_thinking_level_forwarded_when_set(self, mock_genai_client):
        """When thinking_level is set, it must appear in thinking_config."""
        provider = GeminiLLMProvider(
            api_key="test-key",
            model="test-model",
            thinking_level="high",
        )
        aclient = _make_aclient(mock_genai_client)
        aclient.models.generate_content.return_value = _make_resp("ok")

        await provider.complete("hello")

        call = aclient.models.generate_content.call_args[1]
        tc = call["config"].thinking_config
        assert tc is not None
        assert tc.thinking_level is not None
        assert str(tc.thinking_level) == "ThinkingLevel.HIGH"

    @pytest.mark.asyncio
    async def test_thinking_budget_forwarded_when_set(self, mock_genai_client):
        """When thinking_budget is set, a ThinkingConfig must appear."""
        provider = GeminiLLMProvider(
            api_key="test-key",
            model="test-model",
            thinking_budget=1024,
        )
        aclient = _make_aclient(mock_genai_client)
        aclient.models.generate_content.return_value = _make_resp("ok")

        await provider.complete("hello")

        call = aclient.models.generate_content.call_args[1]
        cfg = call["config"].thinking_config
        assert cfg.thinking_budget == 1024

    @pytest.mark.asyncio
    async def test_no_thinking_params_when_unset(self, mock_genai_client):
        """When neither thinking param is set, nothing is sent."""
        provider = GeminiLLMProvider(
            api_key="test-key",
            model="test-model",
        )
        aclient = _make_aclient(mock_genai_client)
        aclient.models.generate_content.return_value = _make_resp("ok")

        await provider.complete("hello")

        call = aclient.models.generate_content.call_args[1]
        cfg = call["config"]
        assert cfg.thinking_config is None


# ---------------------------------------------------------------------------
# Complete – response parsing & error handling
# ---------------------------------------------------------------------------


class TestComplete:
    """Contracts for the complete() method."""

    @pytest.mark.asyncio
    async def test_returns_llmresponse_with_content(self, mock_genai_client):
        """Core contract: complete() returns LLMResponse with valid text."""
        provider = GeminiLLMProvider(api_key="test-key", model="test-model")
        aclient = _make_aclient(mock_genai_client)
        aclient.models.generate_content.return_value = _make_resp(
            "Hello world", prompt_tokens=10, completion_tokens=20
        )

        result = await provider.complete("hello")

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello world"
        assert result.tokens_used == 30
        assert result.model == "test-model"
        assert result.finish_reason == "STOP"

    @pytest.mark.asyncio
    async def test_empty_response_raises(self, mock_genai_client):
        """Empty text must raise RuntimeError, not silently return empty."""
        provider = GeminiLLMProvider(api_key="test-key", model="test-model")
        aclient = _make_aclient(mock_genai_client)
        aclient.models.generate_content.return_value = _make_resp("")

        with pytest.raises(RuntimeError, match="empty response"):
            await provider.complete("hello")

    @pytest.mark.asyncio
    async def test_safety_finish_reason_raises(self, mock_genai_client):
        """SAFETY finish reason must raise a clear blocked error."""
        provider = GeminiLLMProvider(api_key="test-key", model="test-model")
        aclient = _make_aclient(mock_genai_client)
        aclient.models.generate_content.return_value = _make_resp(
            "blocked", finish_reason=types.FinishReason.SAFETY
        )

        with pytest.raises(RuntimeError, match="blocked"):
            await provider.complete("hello")

    @pytest.mark.asyncio
    async def test_truncation_error_raised(self, mock_genai_client):
        """MAX_TOKENS finish reason must raise truncation error, not empty."""
        provider = GeminiLLMProvider(api_key="test-key", model="test-model")
        aclient = _make_aclient(mock_genai_client)
        aclient.models.generate_content.return_value = _make_resp(
            "", finish_reason=types.FinishReason.MAX_TOKENS
        )

        with pytest.raises(RuntimeError, match="token limit exceeded"):
            await provider.complete("hello")

    @pytest.mark.asyncio
    async def test_sdk_error_wraps_runtime_error(self, mock_genai_client):
        """API errors from the SDK must wrap into a descriptive RuntimeError."""
        provider = GeminiLLMProvider(api_key="test-key", model="test-model")
        aclient = _make_aclient(mock_genai_client)

        # Simulate a Google APIError with code 429
        api_err = MagicMock(spec=["code", "message"])
        api_err.code = 429
        api_err.message = "Rate limit exceeded"
        from google.genai import errors as genai_errors
        aclient.models.generate_content.side_effect = genai_errors.APIError(
            code=429, response_json={"error": {"message": "Rate limit exceeded"}}
        )

        with pytest.raises(RuntimeError) as exc:
            await provider.complete("hello")

        msg = str(exc.value)
        assert "rate limit" in msg.lower()

    @pytest.mark.asyncio
    async def test_internal_runtime_error_not_double_wrapped(
        self, mock_genai_client
    ):
        """RuntimeError raised internally must pass through unwrapped."""
        provider = GeminiLLMProvider(api_key="test-key", model="test-model")
        aclient = _make_aclient(mock_genai_client)
        aclient.models.generate_content.return_value = _make_resp("")

        with pytest.raises(RuntimeError) as exc:
            await provider.complete("test")

        msg = str(exc.value)
        assert "empty response" in msg
        assert "LLM completion failed" not in msg


# ---------------------------------------------------------------------------
# Complete Structured – JSON parsing & error handling
# ---------------------------------------------------------------------------


class TestCompleteStructured:
    """Contracts for the complete_structured() method."""

    @pytest.mark.asyncio
    async def test_returns_parsed_dict(self, mock_genai_client):
        """Structured completion must return a parsed JSON dict."""
        provider = GeminiLLMProvider(api_key="test-key", model="test-model")
        aclient = _make_aclient(mock_genai_client)
        aclient.models.generate_content.return_value = _make_resp(
            '{"answer": "42"}'
        )

        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        result = await provider.complete_structured("hello", schema)

        assert result == {"answer": "42"}

    @pytest.mark.asyncio
    async def test_empty_response_raises(self, mock_genai_client):
        """Empty structured response must raise RuntimeError."""
        provider = GeminiLLMProvider(api_key="test-key", model="test-model")
        aclient = _make_aclient(mock_genai_client)
        aclient.models.generate_content.return_value = _make_resp("")

        schema = {"type": "object", "properties": {}}
        with pytest.raises(RuntimeError, match="empty response"):
            await provider.complete_structured("hello", schema)

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self, mock_genai_client):
        """Invalid JSON from the API must raise a clear error."""
        provider = GeminiLLMProvider(api_key="test-key", model="test-model")
        aclient = _make_aclient(mock_genai_client)
        aclient.models.generate_content.return_value = _make_resp(
            "not valid json at all"
        )

        schema = {"type": "object", "properties": {}}
        with pytest.raises(RuntimeError, match="Invalid JSON"):
            await provider.complete_structured("hello", schema)

    @pytest.mark.asyncio
    async def test_internal_runtime_error_not_double_wrapped(
        self, mock_genai_client
    ):
        """RuntimeError from structured code path must not be double-wrapped."""
        provider = GeminiLLMProvider(api_key="test-key", model="test-model")
        aclient = _make_aclient(mock_genai_client)
        aclient.models.generate_content.return_value = _make_resp("")

        with pytest.raises(RuntimeError) as exc:
            await provider.complete_structured(
                "test", json_schema={"type": "object"}
            )

        msg = str(exc.value)
        assert "empty response" in msg
        assert "LLM structured completion failed" not in msg

    @pytest.mark.asyncio
    async def test_safety_finish_reason_raises(self, mock_genai_client):
        """SAFETY finish reason in structured completion must raise blocked error."""
        provider = GeminiLLMProvider(api_key="test-key", model="test-model")
        aclient = _make_aclient(mock_genai_client)
        aclient.models.generate_content.return_value = _make_resp(
            "blocked", finish_reason=types.FinishReason.SAFETY
        )

        with pytest.raises(RuntimeError, match="blocked"):
            await provider.complete_structured(
                "test", json_schema={"type": "object"}
            )

    @pytest.mark.asyncio
    async def test_truncation_error_raised(self, mock_genai_client):
        """MAX_TOKENS finish reason in structured completion must raise truncation error."""
        provider = GeminiLLMProvider(api_key="test-key", model="test-model")
        aclient = _make_aclient(mock_genai_client)
        aclient.models.generate_content.return_value = _make_resp(
            "", finish_reason=types.FinishReason.MAX_TOKENS
        )

        with pytest.raises(RuntimeError, match="truncat"):
            await provider.complete_structured(
                "test", json_schema={"type": "object"}
            )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Health check response contract."""

    @pytest.mark.asyncio
    async def test_healthy_dict_structure(self, mock_genai_client):
        """A working provider returns the expected shape."""
        provider = GeminiLLMProvider(
            api_key="test-key",
            model="test-model",
            thinking_level="high",
            thinking_budget=None,
        )
        aclient = _make_aclient(mock_genai_client)
        aclient.models.generate_content.return_value = _make_resp("OK")

        result = await provider.health_check()

        assert result["status"] == "healthy"
        assert result["provider"] == "gemini"
        assert result["model"] == "test-model"
        assert result["thinking_level"] == "high"
        assert result["thinking_budget"] is None
        assert "test_response" in result

    @pytest.mark.asyncio
    async def test_unhealthy_on_failure(self, mock_genai_client):
        """On failure the health check must degrade gracefully."""
        provider = GeminiLLMProvider(api_key="test-key", model="test-model")
        aclient = _make_aclient(mock_genai_client)
        aclient.models.generate_content.side_effect = Exception("connection failed")

        result = await provider.health_check()

        assert result["status"] == "unhealthy"
        assert "error" in result


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------


class TestUsageStats:
    """Usage statistics tracking contract."""

    @pytest.mark.asyncio
    async def test_initial_state(self, mock_genai_client):
        """Fresh provider has zero usage."""
        provider = GeminiLLMProvider(api_key="test-key", model="test-model")
        stats = provider.get_usage_stats()
        assert stats["requests_made"] == 0
        assert stats["total_tokens"] == 0
        assert stats["prompt_tokens"] == 0
        assert stats["completion_tokens"] == 0

    @pytest.mark.asyncio
    async def test_tracks_after_completion(self, mock_genai_client):
        """After a completion, usage stats reflect the call."""
        provider = GeminiLLMProvider(api_key="test-key", model="test-model")
        aclient = _make_aclient(mock_genai_client)
        aclient.models.generate_content.return_value = _make_resp(
            "ok", prompt_tokens=15, completion_tokens=25
        )

        await provider.complete("hello")

        stats = provider.get_usage_stats()
        assert stats["requests_made"] == 1
        assert stats["prompt_tokens"] == 15
        assert stats["completion_tokens"] == 25
        assert stats["total_tokens"] == 40
