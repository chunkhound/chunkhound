"""Embedding provider exceptions for protocol-level errors.

These exceptions extend ChunkHoundError for consistent error handling with
context tracking. They are specific to embedding provider operations:
dimension validation, batch processing, and matryoshka configuration.

Note: EmbeddingProviderError is distinct from core.EmbeddingError which
covers higher-level embedding operations (generate, store, retrieve).
"""

from chunkhound.core.exceptions.core import ChunkHoundError


class EmbeddingProviderError(ChunkHoundError):
    """Base exception for embedding provider errors.

    Inherits from ChunkHoundError for consistent error handling with
    context tracking and structured error messages.
    """

    pass


class EmbeddingDimensionError(EmbeddingProviderError):
    """Raised when embedding dimension doesn't match expected value.

    This occurs when:
    - API returns embeddings with unexpected dimension
    - provider.dims doesn't match actual embedding length
    """

    pass


class EmbeddingBatchError(EmbeddingProviderError):
    """Raised when embedding batch has inconsistent dimensions.

    This occurs when:
    - Embeddings in a single batch have different dimensions
    - Batch dimension doesn't match expected index dimension
    """

    pass


class EmbeddingConfigurationError(EmbeddingProviderError):
    """Raised for embedding configuration errors.

    This occurs when:
    - Unknown model specified for official API
    - output_dims not in model's supported_dimensions
    - Model doesn't support matryoshka but output_dims specified
    """

    pass
