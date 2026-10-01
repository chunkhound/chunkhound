"""Tests for the OpenAI embedding provider's token-limit-error classifier.

Covers the phrase-matching condition in _embed_batch_internal that decides
whether a BadRequestError should route through handle_token_limit_error's
split-and-retry path, or propagate as a hard failure.
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.unit.provider_test_helpers import _bare_provider, _ok_response


class TestTokenLimitClassifier:
    @pytest.mark.asyncio
    async def test_input_length_exceeds_context_length_triggers_split_retry(self):
        """The provider's own reported error phrasing ('input length exceeds
        the context length') was not covered by either existing condition and
        propagated as a hard failure instead of triggering split-and-retry.
        """
        provider, fake_openai, mod = _bare_provider(retry_attempts=2, retry_delay=0.0)

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise fake_openai.BadRequestError(
                    "Error code: 400 - {'error': {'message': 'the input length "
                    "exceeds the context length', 'type': 'api_error', "
                    "'param': None, 'code': None}}"
                )
            return _ok_response()

        provider._client = MagicMock()
        provider._client.embeddings.create = side_effect

        with patch.object(mod, "openai", fake_openai):
            result = await provider._embed_batch_internal(["hello", "world"])

        assert len(result) == 2
        # 1 failed call on the original batch + one successful call per split.
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_unrelated_bad_request_still_propagates(self):
        """A BadRequestError that is not a token-limit error must still raise,
        never silently routed into the split-retry path.
        """
        provider, fake_openai, mod = _bare_provider(retry_attempts=2, retry_delay=0.0)

        async def always_fail(*args, **kwargs):
            raise fake_openai.BadRequestError("invalid api key provided")

        provider._client = MagicMock()
        provider._client.embeddings.create = always_fail

        with patch.object(mod, "openai", fake_openai):
            with pytest.raises(fake_openai.BadRequestError):
                await provider._embed_batch_internal(["hello"])
