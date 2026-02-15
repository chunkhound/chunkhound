"""Shared utilities for embedding providers."""

from typing import Any


def estimate_tokens_rough(text: str) -> float:
    """Rough token estimation (1.3 tokens per word approximation)."""
    return len(text.split()) * 1.3


def chunk_text_by_words(text: str, max_tokens: int) -> list[str]:
    """Split text into chunks by approximate token count using word splitting."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        word_tokens = len(word) // 4 + 1  # Rough token estimate
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
