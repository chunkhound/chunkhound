"""Tests for OpenAI embedding provider rate-limit retry logic.

Covers the three delay-resolution paths in _embed_batch_internal:
  1. retry-after / x-ratelimit-reset-requests response header
  2. "try again in X seconds" body parse
  3. exponential backoff fallback (capped at 120 s)
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from tests.unit.provider_test_helpers import (
    _bare_provider,
    _FakeRateLimitError,
    _ok_response,
)

# ---------------------------------------------------------------------------
# Minimal fake exception hierarchy
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for an HTTP response with a headers dict."""

    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}


# ---------------------------------------------------------------------------
# Tests: header path
# ---------------------------------------------------------------------------


class TestRetryAfterHeader:
    @pytest.mark.asyncio
    async def test_retry_after_header_used_as_delay(self):
        """retry-after header value drives the sleep duration."""
        provider, fake_openai, mod = _bare_provider(retry_attempts=2, retry_delay=1.0)

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeRateLimitError(
                    "429", response=_FakeResponse({"retry-after": "30"})
                )
            return _ok_response()

        provider._client = MagicMock()
        provider._client.embeddings.create = side_effect

        sleep_calls: list[float] = []

        async def fake_sleep(secs):
            sleep_calls.append(secs)

        with patch.object(mod, "openai", fake_openai), patch.object(
            asyncio, "sleep", fake_sleep
        ):
            result = await provider._embed_batch_internal(["hello"])

        assert len(result) == 1
        assert len(sleep_calls) == 1
        assert sleep_calls[0] >= 30.0
        assert sleep_calls[0] <= 35.0  # 30 + max jitter (10 %, capped at 5 s)

    @pytest.mark.asyncio
    async def test_x_ratelimit_reset_requests_header_used_as_fallback(self):
        """x-ratelimit-reset-requests is used when retry-after is absent."""
        provider, fake_openai, mod = _bare_provider(retry_attempts=2, retry_delay=1.0)

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeRateLimitError(
                    "429",
                    response=_FakeResponse({"x-ratelimit-reset-requests": "20"}),
                )
            return _ok_response()

        provider._client = MagicMock()
        provider._client.embeddings.create = side_effect

        sleep_calls: list[float] = []

        async def fake_sleep(secs):
            sleep_calls.append(secs)

        with patch.object(mod, "openai", fake_openai), patch.object(
            asyncio, "sleep", fake_sleep
        ):
            await provider._embed_batch_internal(["hi"])

        assert sleep_calls[0] >= 20.0
        assert sleep_calls[0] <= 25.0

    @pytest.mark.asyncio
    async def test_header_value_capped_at_120s(self):
        """A server returning a huge retry-after is capped at 120 s."""
        provider, fake_openai, mod = _bare_provider(retry_attempts=2, retry_delay=1.0)

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeRateLimitError(
                    "429", response=_FakeResponse({"retry-after": "3600"})
                )
            return _ok_response()

        provider._client = MagicMock()
        provider._client.embeddings.create = side_effect

        sleep_calls: list[float] = []

        async def fake_sleep(secs):
            sleep_calls.append(secs)

        with patch.object(mod, "openai", fake_openai), patch.object(
            asyncio, "sleep", fake_sleep
        ):
            await provider._embed_batch_internal(["hi"])

        assert sleep_calls[0] <= 125.0  # 120 cap + max 5 s jitter


# ---------------------------------------------------------------------------
# Tests: body-parse path
# ---------------------------------------------------------------------------


class TestRetryAfterBodyParse:
    @pytest.mark.asyncio
    async def test_body_parse_seconds_used_when_no_header(self):
        """'try again in X seconds' in the error body drives the sleep duration."""
        provider, fake_openai, mod = _bare_provider(retry_attempts=2, retry_delay=1.0)

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeRateLimitError(
                    "Rate limit exceeded. Please try again in 35 seconds."
                )
            return _ok_response()

        provider._client = MagicMock()
        provider._client.embeddings.create = side_effect

        sleep_calls: list[float] = []

        async def fake_sleep(secs):
            sleep_calls.append(secs)

        with patch.object(mod, "openai", fake_openai), patch.object(
            asyncio, "sleep", fake_sleep
        ):
            await provider._embed_batch_internal(["hello"])

        assert sleep_calls[0] >= 35.0
        assert sleep_calls[0] <= 40.0

    @pytest.mark.asyncio
    async def test_body_parse_fractional_seconds(self):
        """Fractional values like '1.5 seconds' are parsed correctly."""
        provider, fake_openai, mod = _bare_provider(retry_attempts=2, retry_delay=1.0)

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeRateLimitError("try again in 1.5 seconds")
            return _ok_response()

        provider._client = MagicMock()
        provider._client.embeddings.create = side_effect

        sleep_calls: list[float] = []

        async def fake_sleep(secs):
            sleep_calls.append(secs)

        with patch.object(mod, "openai", fake_openai), patch.object(
            asyncio, "sleep", fake_sleep
        ):
            await provider._embed_batch_internal(["x"])

        assert sleep_calls[0] >= 1.5

    @pytest.mark.asyncio
    async def test_body_parse_value_capped_at_120s(self):
        """Parsed body values above 120 s are capped."""
        provider, fake_openai, mod = _bare_provider(retry_attempts=2, retry_delay=1.0)

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeRateLimitError("try again in 9999 seconds")
            return _ok_response()

        provider._client = MagicMock()
        provider._client.embeddings.create = side_effect

        sleep_calls: list[float] = []

        async def fake_sleep(secs):
            sleep_calls.append(secs)

        with patch.object(mod, "openai", fake_openai), patch.object(
            asyncio, "sleep", fake_sleep
        ):
            await provider._embed_batch_internal(["x"])

        assert sleep_calls[0] <= 125.0


# ---------------------------------------------------------------------------
# Tests: exponential backoff fallback
# ---------------------------------------------------------------------------


class TestExponentialBackoffFallback:
    @pytest.mark.asyncio
    async def test_exponential_backoff_on_plain_rate_limit(self):
        """No header, no body hint → exponential backoff (retry_delay * 2^attempt)."""
        provider, fake_openai, mod = _bare_provider(retry_attempts=3, retry_delay=2.0)

        call_count = 0
        sleep_calls: list[float] = []

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _FakeRateLimitError("429 no details")
            return _ok_response()

        provider._client = MagicMock()
        provider._client.embeddings.create = side_effect

        async def fake_sleep(secs):
            sleep_calls.append(secs)

        with patch.object(mod, "openai", fake_openai), patch.object(
            asyncio, "sleep", fake_sleep
        ):
            await provider._embed_batch_internal(["hello"])

        # attempt 0: 2 * 2^0 = 2 s; attempt 1: 2 * 2^1 = 4 s
        assert len(sleep_calls) == 2
        assert sleep_calls[0] >= 2.0
        assert sleep_calls[1] >= 4.0

    @pytest.mark.asyncio
    async def test_exponential_backoff_capped_at_120s(self):
        """Exponential backoff never exceeds the capped 120 s delay + jitter."""
        provider, fake_openai, mod = _bare_provider(
            retry_attempts=2,
            retry_delay=200.0,
        )

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeRateLimitError("429 no details")
            return _ok_response()

        provider._client = MagicMock()
        provider._client.embeddings.create = side_effect

        sleep_calls: list[float] = []

        async def fake_sleep(secs):
            sleep_calls.append(secs)

        with patch.object(mod, "openai", fake_openai), patch.object(
            asyncio, "sleep", fake_sleep
        ):
            await provider._embed_batch_internal(["x"])

        assert sleep_calls[0] <= 125.0  # 120 cap + max 5 s jitter

    @pytest.mark.asyncio
    async def test_rate_limit_raised_after_all_attempts_exhausted(self):
        """RateLimitError propagates once all retry attempts are consumed."""
        provider, fake_openai, mod = _bare_provider(retry_attempts=2, retry_delay=0.0)

        async def always_fail(*args, **kwargs):
            raise _FakeRateLimitError("429")

        provider._client = MagicMock()
        provider._client.embeddings.create = always_fail

        async def fake_sleep(_):
            pass

        with patch.object(mod, "openai", fake_openai), patch.object(
            asyncio, "sleep", fake_sleep
        ):
            with pytest.raises(_FakeRateLimitError):
                await provider._embed_batch_internal(["x"])
