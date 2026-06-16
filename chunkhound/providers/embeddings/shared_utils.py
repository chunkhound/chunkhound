"""Shared utilities for embedding providers."""

import math
from typing import Any

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
    """Normalize a vector to unit length (L2 norm = 1).

    Required after client-side truncation to preserve cosine similarity semantics.
    """
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return vector
    return [x / norm for x in vector]


def apply_client_side_truncation(embeddings: list[list[float]], output_dims: int) -> list[list[float]]:
    """Truncate embeddings to output_dims and L2-normalize each vector.

    Use when the provider returns full-size vectors and the caller wants
    Matryoshka-style dimension reduction without a server-side API parameter.
    """
    return [l2_normalize(v[:output_dims]) for v in embeddings]
