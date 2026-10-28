"""Tests for the OpenAI embedding provider's token-limit-error handling.

Covers the phrase-matching condition in _embed_batch_internal that decides
whether a BadRequestError should route through handle_token_limit_error's
split-and-retry path, or propagate as a hard failure, and what a caller gets
for a single text the server rejects although the estimate says it fits.
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.unit.provider_test_helpers import _bare_provider, _ok_response


class TestTokenLimitClassifier:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error_message",
        [
            "Error code: 400 - {'error': {'message': 'the input length "
            "exceeds the context length', 'type': 'api_error', "
            "'param': None, 'code': None}}",
            # llama.cpp server, both of its context-size messages
            "Error code: 400 - {'error': {'code': 400, 'message': 'request "
            "(17066 tokens) exceeds the available context size (8192 tokens), "
            "try increasing it', 'type': 'exceed_context_size_error', "
            "'n_prompt_tokens': 17066, 'n_ctx': 8192}}",
            "Error code: 400 - {'error': {'code': 400, 'message': 'input "
            "(9000 tokens) is larger than the max context size (8192 tokens). "
            "skipping', 'type': 'exceed_context_size_error', "
            "'n_prompt_tokens': 9000, 'n_ctx': 8192}}",
        ],
    )
    async def test_context_length_error_triggers_split_retry(self, error_message):
        """A server's context-length phrasing must trigger split-and-retry,
        not propagate as a hard failure.
        """
        provider, fake_openai, mod = _bare_provider(retry_attempts=2, retry_delay=0.0)

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise fake_openai.BadRequestError(error_message)
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


def _llamacpp_server(fake_openai, max_chars: int, accepted: list[str]):
    """Fake llama.cpp /embeddings: a 400 for any input over *max_chars*."""

    async def create(*args, **kwargs):
        inputs = kwargs["input"]
        for text in inputs:
            if len(text) > max_chars:
                raise fake_openai.BadRequestError(
                    "Error code: 400 - {'error': {'code': 400, 'message': "
                    f"'request ({len(text)} tokens) exceeds the available "
                    "context size (8192 tokens), try increasing it', "
                    "'type': 'exceed_context_size_error'}}"
                )
        accepted.extend(inputs)
        response = MagicMock()
        response.data = [
            MagicMock(index=i, embedding=[0.1] * 1536) for i in range(len(inputs))
        ]
        response.usage = MagicMock(total_tokens=10)
        return response

    return create


class TestSingleTextOverServerContext:
    @pytest.mark.asyncio
    async def test_text_the_estimate_fits_gets_its_first_accepted_piece(self):
        """A text the estimate puts under the limit but the server rejects
        still gets an embedding: that of the first piece the server accepts.
        """
        provider, fake_openai, mod = _bare_provider(retry_attempts=1, retry_delay=0.0)
        text = "".join(f"{i:05d}" for i in range(2000))  # 10000 chars
        assert provider.estimate_tokens(text) < provider.get_model_token_limit()
        accepted: list[str] = []
        provider._client = MagicMock()
        provider._client.embeddings.create = _llamacpp_server(
            fake_openai, max_chars=3000, accepted=accepted
        )

        with patch.object(mod, "openai", fake_openai):
            result = await provider.embed([text])

        assert len(result) == 1
        assert len(accepted) == 1
        assert text.startswith(accepted[0])

    @pytest.mark.asyncio
    async def test_text_rejected_at_every_size_fails(self):
        """Halving stops at a floor: a server that rejects every piece fails
        the text instead of halving it forever.
        """
        provider, fake_openai, mod = _bare_provider(retry_attempts=1, retry_delay=0.0)
        provider._client = MagicMock()
        provider._client.embeddings.create = _llamacpp_server(
            fake_openai, max_chars=0, accepted=[]
        )

        with patch.object(mod, "openai", fake_openai):
            with pytest.raises(ValueError, match="Text too large to embed"):
                await provider.embed(["x" * 10000])
