"""ChunkHound Core Exceptions Package - Core exception classes for error handling.

This package contains the exception hierarchy for the ChunkHound system. These
exceptions provide clear error categorization and enable proper error handling
throughout the application.

The exception hierarchy is designed to:
- Provide specific exception types for different error categories
- Maintain backward compatibility with existing error handling
- Enable proper error propagation and debugging
- Support structured error messages and context
"""

from chunkhound.core.exceptions.core import (
    ChunkHoundError,
    ConfigurationError,
    DatabaseError,
    DiskUsageLimitExceededError,
    EmbeddingError,
    ModelError,
    ParsingError,
    ProviderError,
    ValidationError,
)

__all__ = [
    # Base exception
    "ChunkHoundError",
    # Domain-specific exceptions
    "ValidationError",
    "ModelError",
    "EmbeddingError",
    "ParsingError",
    "DatabaseError",
    "ConfigurationError",
    "DiskUsageLimitExceededError",
    "ProviderError",
]
