"""ChunkHound Core Package - Domain models, types, and exceptions.

This package contains the core domain models and types that form the foundation
of the ChunkHound architecture. These models are independent of infrastructure
concerns and provide a clean separation between business
logic and implementation details.

Modules:
    models: Domain models for File, Chunk, and Embedding entities
    types: Common type definitions and aliases
    exceptions: Core exception classes for error handling
"""

from chunkhound.core.exceptions import (
    ChunkHoundError,
    EmbeddingError,
    ModelError,
    ParsingError,
    ValidationError,
)
from chunkhound.core.models import Chunk, Embedding, EmbeddingResult, File
from chunkhound.core.types import ChunkType, Language, ModelName, ProviderName
from chunkhound.version import __version__

__all__ = [
    # Domain Models
    "File",
    "Chunk",
    "Embedding",
    "EmbeddingResult",
    # Types
    "ChunkType",
    "Language",
    "ProviderName",
    "ModelName",
    # Exceptions
    "ChunkHoundError",
    "ValidationError",
    "ModelError",
    "EmbeddingError",
    "ParsingError",
    # Version
    "__version__",
]
