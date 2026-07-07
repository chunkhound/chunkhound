"""Interfaces package for ChunkHound -
abstract protocols for provider implementations."""

from .database_provider import DatabaseProvider
from .embedding_provider import EmbeddingProvider
from .language_parser import LanguageParser

__all__ = [
    "DatabaseProvider",
    "EmbeddingProvider",
    "LanguageParser",
]
