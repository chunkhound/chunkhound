"""Shared utilities for embedding providers."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from chunkhound.core.exceptions.embedding import (
    EmbeddingConfigurationError,
    EmbeddingDimensionError,
    EmbeddingProviderError,
)
from chunkhound.core.utils.token_utils import LLM_CHARS_PER_TOKEN


def chunk_text_by_words(text: str, max_tokens: int) -> list[str]:
    """Split text into chunks by approximate token count using word splitting."""
    words = text.split()
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for word in words:
        word_tokens = len(word) // LLM_CHARS_PER_TOKEN + 1
        if current_tokens + word_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_tokens += word_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def validate_text_input(texts: list[str]) -> list[str]:
    """Validate and preprocess texts before embedding."""
    if not texts:
        return []

    validated: list[str] = []
    for text in texts:
        if not isinstance(text, str):
            raise ValueError(f"Text must be string, got {type(text)}")
        if not text.strip():
            continue  # Skip empty texts
        validated.append(text.strip())

    return validated


def require_non_blank_queries(texts: Sequence[str]) -> None:
    """Raise ValueError if any search query is blank (it has no embedding)."""
    if any(not t.strip() for t in texts):
        raise ValueError("Search query must not be blank")


async def embed_preserving_positions(
    texts: list[str],
    embed_non_blank: Callable[[list[str]], Awaitable[list[list[float]]]],
    fallback_dims: int | None = None,
) -> list[list[float]]:
    """Embed ``texts`` returning exactly one vector per input, in order.

    Blank/whitespace-only texts are never sent to the API (providers reject or
    silently drop them). Their slots get a zero vector whose length is taken
    from the vectors the API actually returned, so custom or unknown models
    work regardless of what ``provider.dims`` claims.

    Zero vectors have undefined direction under cosine similarity, so callers
    must not persist them: indexing paths filter blank chunks before embedding
    and query paths reject blank queries. This padding exists only to keep the
    one-vector-per-input contract. If every text is blank no request is made, so
    the length comes from ``fallback_dims`` (best guess, e.g. ``provider.dims``);
    without it this raises ``ValueError``.
    """
    non_blank = validate_text_input(texts)
    if not non_blank:
        if fallback_dims is None:
            raise ValueError("Cannot embed: all input texts are blank")
        return [[0.0] * fallback_dims for _ in texts]
    embedded = await embed_non_blank(non_blank)
    if len(embedded) != len(non_blank):
        raise EmbeddingProviderError(
            f"Embedding API returned {len(embedded)} vectors for "
            f"{len(non_blank)} non-blank texts"
        )
    zero = [0.0] * len(embedded[0])
    results = iter(embedded)
    return [next(results) if text.strip() else list(zero) for text in texts]


def ensure_vector_count(vectors: Sequence[Any], texts: Sequence[Any]) -> None:
    """Raise a clear provider error if a provider returned the wrong count."""
    if len(vectors) != len(texts):
        raise EmbeddingProviderError(
            f"Embedding provider returned {len(vectors)} vectors for "
            f"{len(texts)} texts"
        )


def get_usage_stats_dict(
    requests_made: int, tokens_used: int, embeddings_generated: int
) -> dict[str, Any]:
    """Get standardized usage statistics dictionary."""
    return {
        "requests_made": requests_made,
        "tokens_used": tokens_used,
        "embeddings_generated": embeddings_generated,
    }


def get_dimensions_for_model(
    model: str, dimensions_map: dict[str, int], default_dims: int = 1536
) -> int:
    """Get embedding dimensions for a model with fallback to default."""
    return dimensions_map.get(model, default_dims)


def l2_normalize(vector: list[float]) -> list[float]:
    """L2-normalize vector to unit length for cosine similarity.

    Args:
        vector: Input vector to normalize

    Returns:
        Unit-length vector (or original if zero-magnitude)
    """
    magnitude = sum(x * x for x in vector) ** 0.5
    if magnitude > 0:
        return [x / magnitude for x in vector]
    return vector


def apply_client_side_truncation(
    embeddings: list[list[float]], output_dims: int
) -> list[list[float]]:
    """Truncate embeddings to output_dims and L2-normalize.

    Use when the API doesn't support a server-side dimensions parameter.
    The API returns full-dimension vectors; this truncates and re-normalizes.

    Args:
        embeddings: Full-dimension embedding vectors from API
        output_dims: Positive validated target dimension after truncation

    Returns:
        Truncated and L2-normalized embedding vectors.

    Raises:
        EmbeddingDimensionError: If a vector is shorter than output_dims.

    Contract:
        Callers must validate output_dims before reaching this helper.
    """
    for v in embeddings:
        if len(v) < output_dims:
            raise EmbeddingDimensionError(
                f"Vector dimension {len(v)} < requested output_dims {output_dims}"
            )
    return [l2_normalize(v[:output_dims]) for v in embeddings]


def build_dimension_request_param(
    output_dims: int | None,
    client_side_truncation: bool,
) -> int | None:
    """Return the dimension param value for the API request, or None.

    When server-side truncation is active (output_dims set and NOT
    client-side), the API receives a dimension parameter and returns
    already-truncated vectors.  When client-side truncation is active
    or no output_dims is set, the API gets no dimension parameter.

    The caller maps the return value to the provider-specific API
    param name (OpenAI: "dimensions", VoyageAI: "output_dimension").
    """
    if output_dims is not None and not client_side_truncation:
        return output_dims
    return None


def build_runtime_supported_dimensions(
    native_dims: int | None,
    client_side_truncation: bool,
) -> Sequence[int]:
    """Return supported dims for runtime-discovered models/endpoints.

    When the client truncates, all dimensions up to native are valid
    (``range(1, native_dims + 1)``).  When the server truncates or no
    truncation is active, only the native dimension is valid.

    Runtime-discovered models assume ``min_dims=1`` (every dimension
    from 1 to native is reachable via client-side truncation).
    """
    if native_dims is None:
        return []
    if client_side_truncation:
        return range(1, native_dims + 1)
    return [native_dims]


def validate_positive_output_dims(
    output_dims: int | None,
    *,
    model: str | None = None,
) -> int | None:
    """Validate output_dims is a positive integer, or return None if unset.

    Shared validation for type and range, used by both providers at
    init-time and embed-time.

    Args:
        output_dims: The value to validate.
        model: Optional model name for error messages.

    Returns:
        Validated positive int, or None if output_dims is None.

    Raises:
        EmbeddingConfigurationError: If output_dims is set but not a
            positive integer.
    """
    if output_dims is None:
        return None
    # bool is a subclass of int in Python — reject explicitly
    if isinstance(output_dims, bool) or not isinstance(output_dims, int) or output_dims <= 0:
        prefix = f"Model '{model}' uses " if model else ""
        raise EmbeddingConfigurationError(
            f"{prefix}output_dims={output_dims!r}, but "
            "output_dims must be a positive integer."
        )
    return output_dims


def validate_runtime_output_dims_config(
    output_dims: int | None,
    client_side_truncation: bool,
    *,
    model: str | None = None,
    context: str | None = None,
) -> int | None:
    """Validate runtime output_dims config, guarding against missing
    dims for client-side truncation.

    Wraps ``validate_positive_output_dims`` then checks the
    ``client_side_truncation`` invariant: if server-side truncation is
    disabled (client-side active), output_dims must be set.

    Args:
        output_dims: The value to validate.
        client_side_truncation: Whether client-side truncation is active.
        model: Optional model name for error messages.
        context: Optional context string for the error message, e.g.
            ``"runtime truncation"`` to clarify the embed-time path.

    Returns:
        Validated positive int, or None if output_dims is None.

    Raises:
        EmbeddingConfigurationError: If output_dims is None but
            client_side_truncation is True, or if output_dims is not
            a positive integer.
    """
    dims = validate_positive_output_dims(output_dims, model=model)
    if dims is None and client_side_truncation:
        if context:
            raise EmbeddingConfigurationError(
                f"Model '{model}' uses client_side_truncation=True but "
                f"output_dims is not set. Set output_dims before using {context}."
            )
        raise EmbeddingConfigurationError(
            f"Model '{model}' uses client_side_truncation=True but "
            "output_dims is not set. output_dims must be a positive integer."
        )
    return dims


def validate_embedding_dims(
    actual_dims: int,
    expected_dims: int,
    *,
    model: str | None = None,
) -> None:
    """Validate embedding dimensions match expected value.

    Raises:
        EmbeddingDimensionError: If dimensions don't match.
    """
    if actual_dims != expected_dims:
        msg = (
            f"Embedding dimension mismatch: got {actual_dims}, expected {expected_dims}"
        )
        if model:
            msg += f" (model={model})"
        raise EmbeddingDimensionError(msg)
