"""Shared helpers for constructing bare OpenAIEmbeddingProviders in unit tests.

Uses __new__() to bypass __init__() and its side effects (import check,
config validation, Azure detection, etc.). Returns a provider with a
noop _ensure_client — the caller attaches their own _client mock.
"""

from unittest.mock import MagicMock

from chunkhound.providers.embeddings.openai_provider import (
    OPENAI_MODEL_CONFIG,
    OpenAIEmbeddingProvider,
)

# ---------------------------------------------------------------------------
# Fake exception types for isinstance() checks in rate-limit/error paths
# ---------------------------------------------------------------------------


class _FakeRateLimitError(Exception):
    """Stand-in for openai.RateLimitError — the provider matches on isinstance()."""

    def __init__(self, message: str = "429", response=None):
        super().__init__(message)
        self.response = response


# ---------------------------------------------------------------------------
# Bare provider builder
# ---------------------------------------------------------------------------


def _bare_provider(
    retry_attempts: int = 3,
    retry_delay: float = 1.0,
    *,
    model: str = "text-embedding-3-small",
    output_dims: int | None = None,
    client_side_truncation: bool = False,
) -> tuple[OpenAIEmbeddingProvider, MagicMock, type]:
    """Return (provider, fake_openai_module, real_provider_module).

    *fake_openai* has fake exception types set up so tests that exercise
    rate-limit or error paths can use
    ``patch.object(mod, "openai", fake_openai)`` for isinstance() checks.

    The returned provider has _ensure_client replaced with a noop async
    function. Callers must set **provider._client** to their mock before
    calling ``_embed_batch_internal``.
    """
    import chunkhound.providers.embeddings.openai_provider as mod

    fake_openai = MagicMock()
    fake_openai.RateLimitError = _FakeRateLimitError
    fake_openai.BadRequestError = type("BadRequestError", (Exception,), {})
    fake_openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
    fake_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})

    provider = OpenAIEmbeddingProvider.__new__(OpenAIEmbeddingProvider)
    provider._retry_attempts = retry_attempts
    provider._retry_delay = retry_delay
    provider._client = None
    provider._model = model
    provider._base_url = None
    provider._azure_endpoint = None
    provider._azure_deployment = None
    provider._timeout = 30
    provider._output_dims = output_dims
    provider._client_side_truncation = client_side_truncation
    provider._discovered_native_dims = None
    provider._warned_default_dims = False
    provider._model_config = OPENAI_MODEL_CONFIG
    provider._usage_stats = {
        "requests_made": 0,
        "embeddings_generated": 0,
        "tokens_used": 0,
        "errors": 0,
    }

    async def _noop_ensure_client():
        pass

    provider._ensure_client = _noop_ensure_client

    return provider, fake_openai, mod


# ---------------------------------------------------------------------------
# Fake response builder
# ---------------------------------------------------------------------------


def _ok_response(*, dim: int | None = None, embedding: list[float] | None = None):
    """Return a MagicMock embedding response.

    Provide *dim* (dimension count, filled with ``0.1``) or *embedding*
    (explicit vector).  Defaults to 1536-dimensional vectors when neither
    is given.
    """
    result = MagicMock()
    if embedding is not None:
        vec = embedding
    elif dim is not None:
        vec = [0.1] * dim
    else:
        vec = [0.1] * 1536
    result.data = [MagicMock(index=0, embedding=vec)]
    result.usage = MagicMock(total_tokens=10)
    return result
