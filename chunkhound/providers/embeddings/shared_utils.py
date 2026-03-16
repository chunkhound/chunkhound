"""Shared utilities for embedding providers."""

from typing import Any

from chunkhound.core.exceptions.embedding import EmbeddingDimensionError
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
        raise ValueError(f"All embeddings must have equal dimensions, got {dims}")
    pooled = [sum(col) / len(embeddings) for col in zip(*embeddings)]
    return l2_normalize(pooled)


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
