"""Shared utilities for embedding providers."""

from typing import Any

from chunkhound.core.exceptions.embedding import (
    EmbeddingConfigurationError,
    EmbeddingDimensionError,
)
from chunkhound.core.utils.token_utils import LLM_CHARS_PER_TOKEN


def chunk_text_by_words(text: str, max_tokens: int) -> list[str]:
    """Split text into chunks by approximate token count using word splitting."""
    words = text.split()
    chunks = []
    current_chunk = []
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

    validated = []
    for text in texts:
        if not isinstance(text, str):
            raise ValueError(f"Text must be string, got {type(text)}")
        if not text.strip():
            continue  # Skip empty texts
        validated.append(text.strip())

    return validated


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


def mean_pool_embeddings(embeddings: list[list[float]]) -> list[float]:
    """Average multiple embeddings into one L2-normalized vector.

    Used when an oversized text is split into chunks that each produce their own
    embedding. The mean-pooled result represents the full text as a single vector.

    Args:
        embeddings: List of equal-dimension embedding vectors.

    Returns:
        Single L2-normalized embedding vector.

    Raises:
        ValueError: If embeddings list is empty.
    """
    if not embeddings:
        raise ValueError("Cannot mean-pool an empty list of embeddings")
    if len(embeddings) == 1:
        return embeddings[0]
    dim = len(embeddings[0])
    if any(len(e) != dim for e in embeddings[1:]):
        dims = [len(e) for e in embeddings]
        raise EmbeddingDimensionError(
            f"All embeddings must have equal dimensions, got {dims}"
        )
    pooled = [sum(col) / len(embeddings) for col in zip(*embeddings)]
    return l2_normalize(pooled)


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


def validate_embedding_dims(
    actual_dims: int,
    expected_dims: int,
    *,
    model: str | None = None,
) -> None:
    """Validate embedding dimensions match expected value (INV-1).

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
