"""Token estimation utilities for accurate provider-specific counting.

This module provides centralized token estimation to ensure consistency
between parser chunking and embedding service batching.
"""

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Token estimation ratios (characters per token)
EMBEDDING_CHARS_PER_TOKEN = 3
LLM_CHARS_PER_TOKEN = 4
DEFAULT_CHARS_PER_TOKEN = 3.5


def estimate_tokens_llm(text: str) -> int:
    """Token estimation for LLM providers (4 chars/token).

    Central implementation - LLM providers should call this
    unless they have a provider-specific tokenizer.
    """
    if not text:
        return 0
    return max(1, len(text) // LLM_CHARS_PER_TOKEN)


def estimate_tokens_embedding(text: str) -> int:
    """Token estimation for embedding providers (3 chars/token).

    Central implementation for chunking - used by parsers to estimate
    chunk sizes for embedding APIs.
    """
    if not text:
        return 0
    return max(1, len(text) // EMBEDDING_CHARS_PER_TOKEN)


def estimate_tokens(
    text: str,
    provider: str | None = None,
    model: str | None = None,
    require_provider: bool = False,
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
        elif require_provider:
            raise ValueError("No embedding provider configured")
        else:
            # Fallback to default estimation when provider not required
            return _estimate_tokens_default(text)

    if provider == "openai" and TIKTOKEN_AVAILABLE:
        return _estimate_tokens_openai(text, model or "")
    elif provider == "voyageai":
        return _estimate_tokens_voyageai(text)
    else:
        return _estimate_tokens_default(text)


def _estimate_tokens_openai(text: str, model: str) -> int:
    """Use tiktoken for exact OpenAI token counting."""
    if not TIKTOKEN_AVAILABLE:
        # Fallback to conservative estimation
        return max(1, int(len(text) / EMBEDDING_CHARS_PER_TOKEN))

    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Model not found, use default encoding
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 default
        return len(encoding.encode(text))


def _estimate_tokens_voyageai(text: str) -> int:
    """Estimate tokens for VoyageAI using measured ratio.

    Based on actual measurements:
    - 325,138 tokens for 975,414 chars = 3.0 chars/token
    """
    return max(1, int(len(text) / EMBEDDING_CHARS_PER_TOKEN))


def _estimate_tokens_default(text: str) -> int:
    """Conservative default estimation for unknown providers."""
    return max(1, int(len(text) / DEFAULT_CHARS_PER_TOKEN))


def get_chars_to_tokens_ratio(provider: str, model: str = "") -> float:
    """Get chars-to-tokens ratio for a provider/model combination.

    This is the inverse of token estimation - useful for calculating
    maximum character limits from token limits.
    """
    if provider == "openai":
        # tiktoken is exact, but for ratio calculations use conservative estimate
        return float(EMBEDDING_CHARS_PER_TOKEN)
    elif provider == "voyageai":
        return float(EMBEDDING_CHARS_PER_TOKEN)  # Measured ratio
    else:
        return float(DEFAULT_CHARS_PER_TOKEN)  # Conservative default
