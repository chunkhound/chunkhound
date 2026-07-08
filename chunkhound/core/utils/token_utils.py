"""Token estimation utilities for accurate provider-specific counting.

This module provides centralized token estimation to ensure consistency
between parser chunking and embedding service batching.
"""

import functools

from loguru import logger

from chunkhound.core.utils.openai_utils import is_official_openai_endpoint

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


@functools.cache
def _encoding_for_model_safe(model: str) -> "tiktoken.Encoding | None":
    """tiktoken.encoding_for_model with all failures → None.

    Negative-caches both unknown models (KeyError) and BPE-download
    failures (e.g. on-prem hosts blocking openaipublic.blob.core.windows.net).
    tiktoken does not cache failed downloads, so without this every call
    retries and stalls on the connect timeout. functools.cache ensures the
    body runs once per unique model, so the warning fires once per model.

    MODULE INVARIANT: tiktoken's encoding constructor functions (e.g.
    cl100k_base() in tiktoken_ext.openai_public) eagerly call
    load_tiktoken_bpe() before Encoding.__init__ runs, so a blocked
    download surfaces from this encoding_for_model() call — NOT from a
    later enc.encode() call. If a future tiktoken version defers BPE
    loading to first encode(), this wrapper will return a non-None
    encoding and the network error will leak past enc.encode(text) in
    _estimate_tokens_openai; revisit when bumping the tiktoken pin.
    """
    if not TIKTOKEN_AVAILABLE:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception as e:
        # Broad on purpose: tiktoken's download path raises urllib.error.URLError,
        # ConnectionError, TimeoutError, OSError, plus client-specific errors
        # that vary across versions. KeyError is the unknown-model case. We also
        # accept that programming-bug exceptions (TypeError on a bad model arg,
        # etc.) collapse to the heuristic — the logged type+message lets an
        # operator distinguish "blocked network" from "bad input" in logs.
        logger.warning(
            "tiktoken failed for model {!r} ({}: {}); using char-count heuristic",
            model, type(e).__name__, e,
        )
        return None


@functools.cache
def _cl100k_base_safe() -> "tiktoken.Encoding | None":
    """Cached cl100k_base lookup; None if the download is blocked.

    functools.cache makes this a one-shot — the warning fires at most once per process.
    """
    if not TIKTOKEN_AVAILABLE:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        # See _encoding_for_model_safe for why this is broad.
        logger.warning(
            "tiktoken cl100k_base download failed ({}: {}); using char-count heuristic",
            type(e).__name__, e,
        )
        return None


# Token estimation ratios (characters per token)
EMBEDDING_CHARS_PER_TOKEN = 3  # Embedding APIs (measured ~3.0 for VoyageAI/OpenAI)
LLM_CHARS_PER_TOKEN = 4  # LLM APIs (conservative estimate)
# Midpoint of EMBEDDING (3) and LLM (4) ratios — used when the target
# provider type is unknown.
DEFAULT_CHARS_PER_TOKEN = 3.5


def estimate_tokens_llm(text: str) -> int:
    """Token estimation for LLM providers (4 chars/token).

    Central implementation - LLM providers should call this
    unless they have a provider-specific tokenizer.
    """
    if not text:
        return 0
    return max(1, len(text) // LLM_CHARS_PER_TOKEN)


def estimate_tokens_chunking(text: str) -> int:
    """Token estimation for chunking decisions (3 chars/token).

    Used by parsers and splitters to enforce chunk size limits before indexing.
    Ratio is calibrated against embedding model token windows (3 chars/token).
    """
    if not text:
        return 0
    return max(1, len(text) // EMBEDDING_CHARS_PER_TOKEN)


def estimate_tokens(
    text: str,
    provider: str | None = None,
    model: str | None = None,
    require_provider: bool = False,
    base_url: str | None = None,
) -> int:
    """Estimate token count for text using provider-specific methods.

    Args:
        text: Text to estimate tokens for
        provider: Provider name (openai, voyageai, etc.).
            If None, gets from registry config.
        model: Model name for provider-specific tokenization.
            If None, gets from registry config.
        require_provider: If True, raises error when no provider configured.
            If False, uses default estimation.
        base_url: Endpoint URL. Used to distinguish official OpenAI from
            OpenAI-compatible proxies (e.g. Qwen via vLLM) where cl100k_base
            would imply false precision. If None, gets from registry config.

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # If no provider passed, get from registry config
    if provider is None:
        from chunkhound.registry import get_registry

        registry = get_registry()
        config = registry.get_config()
        if config and config.embedding:
            provider = config.embedding.provider
            model = config.embedding.model or ""
            base_url = config.embedding.base_url
        elif require_provider:
            raise ValueError("No embedding provider configured")
        else:
            # Fallback to default estimation when provider not required
            return _estimate_tokens_default(text)

    if provider in ("openai", "azure_openai"):
        return _estimate_tokens_openai(
            text, model or "", _uses_openai_tokenizer(provider, base_url)
        )
    elif provider == "voyageai":
        return _estimate_tokens_voyageai(text)
    else:
        return _estimate_tokens_default(text)


def _uses_openai_tokenizer(provider: str, base_url: str | None) -> bool:
    """True iff this endpoint serves models with real OpenAI tokenizers.

    Azure deployments always do (real OpenAI models under arbitrary
    deployment names). Official OpenAI endpoints do. OpenAI-compatible
    proxies (e.g. Qwen via vLLM) do NOT — falling back to cl100k_base
    there would imply false precision.
    """
    if provider == "azure_openai":
        return True
    return provider == "openai" and is_official_openai_endpoint(base_url)


def _estimate_tokens_openai(
    text: str, model: str, cl100k_fallback_ok: bool
) -> int:
    """Use tiktoken for exact OpenAI token counting, with safe fallback.

    cl100k_fallback_ok controls what happens when tiktoken can't resolve
    the model: True (official OpenAI / Azure) uses cl100k_base; False
    (compatible proxy) goes straight to the char heuristic.
    """
    enc = _encoding_for_model_safe(model)
    if enc is None and cl100k_fallback_ok:
        enc = _cl100k_base_safe()
    if enc is not None:
        # disallowed_special=() so a source file containing a special-token
        # literal (e.g. "<|endoftext|>") is counted as ordinary text instead of
        # raising and failing the whole embedding batch. #315
        return len(enc.encode(text, disallowed_special=()))
    # EMBEDDING_CHARS_PER_TOKEN (not LLM_CHARS_PER_TOKEN) because this function
    # is only called from estimate_tokens() for embedding providers; LLM callers
    # use estimate_tokens_llm() directly.
    return max(1, len(text) // EMBEDDING_CHARS_PER_TOKEN)


def _estimate_tokens_voyageai(text: str) -> int:
    """Estimate tokens for VoyageAI using measured ratio.

    Based on actual measurements:
    - 325,138 tokens for 975,414 chars = 3.0 chars/token
    """
    return max(1, len(text) // EMBEDDING_CHARS_PER_TOKEN)


def _estimate_tokens_default(text: str) -> int:
    """Conservative default estimation for unknown providers."""
    return max(1, int(len(text) / DEFAULT_CHARS_PER_TOKEN))


def get_chars_to_tokens_ratio(provider: str, model: str = "") -> float:
    """Get chars-to-tokens ratio for a provider/model combination.

    This is the inverse of token estimation - useful for calculating
    maximum character limits from token limits.
    """
    if provider in ("openai", "azure_openai"):
        # tiktoken is exact, but for ratio calculations use conservative estimate
        return float(EMBEDDING_CHARS_PER_TOKEN)
    elif provider == "voyageai":
        return float(EMBEDDING_CHARS_PER_TOKEN)  # Measured ratio
    else:
        return float(DEFAULT_CHARS_PER_TOKEN)  # Conservative default
