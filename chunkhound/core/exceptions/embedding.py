"""Embedding interface exceptions - simple exceptions for protocol-level errors.

These exceptions are distinct from core.EmbeddingError which provides rich context
tracking. These are lightweight exceptions for configuration and dimension validation
at the embedding provider interface level.

Note: EmbeddingProviderError is the base to avoid collision with core.EmbeddingError.
ConfigurationError here is specific to embedding config, distinct from
core.ConfigurationError.
"""


class EmbeddingProviderError(Exception):
    """Base exception for embedding provider errors."""

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


class ConfigurationError(EmbeddingProviderError):
    """Raised for embedding configuration errors.

    This occurs when:
    - Unknown model specified for official API
    - output_dims not in model's supported_dimensions
    - Model doesn't support matryoshka but output_dims specified
    """

    pass


# Backward compatibility alias
EmbeddingError = EmbeddingProviderError
