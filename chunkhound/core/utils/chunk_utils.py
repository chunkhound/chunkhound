"""Chunk utility functions for cross-layer use.

This module provides utilities for working with chunk dictionaries
that are used across multiple layers (database, embedding, research).
"""


def get_chunk_id(chunk: dict) -> int | str | None:
    """Extract chunk ID from a chunk dictionary.

    Chunks may have their ID in either 'chunk_id' or 'id' field depending
    on the source (database vs search results).

    Args:
        chunk: Chunk dictionary

    Returns:
        Chunk ID (int from DB or str) or None if not found
    """
    return chunk.get("chunk_id") or chunk.get("id")
