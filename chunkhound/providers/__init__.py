"""Providers package for ChunkHound -
concrete implementations of abstract interfaces."""

from .database import DuckDBProvider
from .embeddings import OpenAIEmbeddingProvider

__all__ = [
    # Database providers
    "DuckDBProvider",
    # Embedding providers
    "OpenAIEmbeddingProvider",
]
