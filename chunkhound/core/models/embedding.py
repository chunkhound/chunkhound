"""ChunkHound Embedding Domain Model - Represents vector embeddings in the system.

This module contains the Embedding domain model which represents a vector embedding
that has been generated for a code chunk. The Embedding model encapsulates embedding
metadata, vector data, and provides methods for working
with embeddings in a type-safe manner.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from chunkhound.core.exceptions import ModelError, ValidationError
from chunkhound.core.types import (
    ChunkId,
    Dimensions,
    Distance,
    EmbeddingVector,
    ModelName,
    ProviderName,
)


@dataclass(frozen=True)
class Embedding:
    """Domain model representing a vector embedding for a code chunk.

    This immutable model encapsulates all information about a vector embedding
    that has been generated for a semantic code chunk, including the vector data,
    provider information, and metadata.

    Attributes:
        chunk_id: Reference to the chunk this embedding represents
        provider: Name of the embedding provider (e.g., "openai")
        model: Model name used to generate the embedding
        dims: Number of dimensions in the embedding vector
        vector: The actual embedding vector
        created_at: When the embedding was generated
    """

    chunk_id: ChunkId
    provider: ProviderName
    model: ModelName
    dims: Dimensions
    vector: EmbeddingVector
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        """Validate embedding model after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate embedding model attributes."""
        # Provider validation
        if not self.provider or not self.provider.strip():
            raise ValidationError("provider", self.provider, "Provider cannot be empty")

        # Model validation
        if not self.model or not self.model.strip():
            raise ValidationError("model", self.model, "Model cannot be empty")

        # Dimensions validation
        if self.dims <= 0:
            raise ValidationError("dims", self.dims, "Dimensions must be positive")

        # Vector validation
        if not self.vector:
            raise ValidationError("vector", self.vector, "Vector cannot be empty")

        if len(self.vector) != self.dims:
            raise ValidationError(
                "vector_length",
                len(self.vector),
                f"Vector length ({len(self.vector)}) "
                f"must match dimensions ({self.dims})",
            )

        # Check for invalid values in vector
        for i, value in enumerate(self.vector):
            if not isinstance(value, (int, float)):
                raise ValidationError(
                    f"vector[{i}]",
                    value,
                    f"Vector values must be numeric, got {type(value)}",
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Embedding":
        """Create an Embedding model from a dictionary.

        This method provides backward compatibility with existing code that
        uses dictionary representations of embeddings.

        Args:
            data: Dictionary containing embedding data

        Returns:
            Embedding model created from dictionary data

        Raises:
            ValidationError: If required fields are missing or invalid
        """
        try:
            # Extract required fields
            chunk_id = data.get("chunk_id")
            if chunk_id is None:
                raise ValidationError("chunk_id", chunk_id, "Chunk ID is required")

            provider = data.get("provider")
            if not provider:
                raise ValidationError("provider", provider, "Provider is required")

            model = data.get("model")
            if not model:
                raise ValidationError("model", model, "Model is required")

            dims = data.get("dims")
            if dims is None:
                raise ValidationError("dims", dims, "Dimensions are required")

            vector = data.get("vector")
            if not vector:
                raise ValidationError("vector", vector, "Vector is required")

            # Handle optional fields
            created_at = data.get("created_at")
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)

            return cls(
                chunk_id=ChunkId(chunk_id),
                provider=ProviderName(provider),
                model=ModelName(model),
                dims=Dimensions(int(dims)),
                vector=vector,
                created_at=created_at,
            )

        except (ValueError, TypeError) as e:
            raise ValidationError("data", data, f"Invalid data format: {e}")

    def to_dict(self) -> dict[str, Any]:
        """Convert Embedding model to dictionary.

        This method provides backward compatibility with existing code that
        expects dictionary representations of embeddings.

        Returns:
            Dictionary representation of the embedding
        """
        result = {
            "chunk_id": self.chunk_id,
            "provider": self.provider,
            "model": self.model,
            "dims": self.dims,
            "vector": self.vector,
        }

        if self.created_at is not None:
            result["created_at"] = self.created_at.isoformat()

        return result

    @property
    def vector_size(self) -> int:
        """Get the size of the embedding vector."""
        return len(self.vector)

    @property
    def provider_model_key(self) -> str:
        """Get a unique key combining provider and model."""
        return f"{self.provider}/{self.model}"

    def dot_product(self, other: "Embedding") -> float:
        """Compute dot product with another embedding.

        Args:
            other: Another embedding to compute dot product with

        Returns:
            Dot product value

        Raises:
            ModelError: If embeddings have different dimensions
        """
        if self.dims != other.dims:
            raise ModelError(
                "Embedding",
                "dot_product",
                f"Dimension mismatch: {self.dims} vs {other.dims}",
            )

        return sum(a * b for a, b in zip(self.vector, other.vector))

    def cosine_similarity(self, other: "Embedding") -> Distance:
        """Compute cosine similarity with another embedding.

        Args:
            other: Another embedding to compute similarity with

        Returns:
            Cosine similarity value (-1 to 1)

        Raises:
            ModelError: If embeddings have different dimensions
        """
        if self.dims != other.dims:
            raise ModelError(
                "Embedding",
                "cosine_similarity",
                f"Dimension mismatch: {self.dims} vs {other.dims}",
            )

        # Compute dot product
        dot_prod = self.dot_product(other)

        # Compute magnitudes
        mag_self = sum(x * x for x in self.vector) ** 0.5
        mag_other = sum(x * x for x in other.vector) ** 0.5

        # Avoid division by zero
        if mag_self == 0 or mag_other == 0:
            return Distance(0.0)

        return Distance(dot_prod / (mag_self * mag_other))

    def euclidean_distance(self, other: "Embedding") -> Distance:
        """Compute Euclidean distance with another embedding.

        Args:
            other: Another embedding to compute distance with

        Returns:
            Euclidean distance value

        Raises:
            ModelError: If embeddings have different dimensions
        """
        if self.dims != other.dims:
            raise ModelError(
                "Embedding",
                "euclidean_distance",
                f"Dimension mismatch: {self.dims} vs {other.dims}",
            )

        squared_diff = sum((a - b) ** 2 for a, b in zip(self.vector, other.vector))
        return Distance(squared_diff**0.5)

    def magnitude(self) -> float:
        """Compute the magnitude (L2 norm) of the embedding vector."""
        return float(sum(x * x for x in self.vector) ** 0.5)

    def normalize(self) -> "Embedding":
        """Create a normalized version of this embedding.

        Returns:
            New Embedding instance with normalized vector
        """
        magnitude = self.magnitude()
        if magnitude == 0:
            # Return unchanged if zero vector
            return self

        normalized_vector = [x / magnitude for x in self.vector]

        return Embedding(
            chunk_id=self.chunk_id,
            provider=self.provider,
            model=self.model,
            dims=self.dims,
            vector=normalized_vector,
            created_at=self.created_at,
        )

    def is_compatible_with(self, other: "Embedding") -> bool:
        """Check if this embedding is compatible with another for similarity operations.

        Args:
            other: Another embedding to check compatibility with

        Returns:
            True if embeddings can be compared
        """
        return (
            self.provider == other.provider
            and self.model == other.model
            and self.dims == other.dims
        )

    def __str__(self) -> str:
        """Return string representation of the embedding."""
        return (
            f"Embedding(chunk_id={self.chunk_id}, "
            f"{self.provider_model_key}, "
            f"dims={self.dims})"
        )

    def __repr__(self) -> str:
        """Return detailed string representation of the embedding."""
        return (
            f"Embedding(chunk_id={self.chunk_id}, provider='{self.provider}', "
            f"model='{self.model}', dims={self.dims}, vector_size={self.vector_size})"
        )


@dataclass(frozen=True)
class EmbeddingResult:
    """Result from an embedding generation operation.

    This model represents the result of generating embeddings for one or more texts,
    including the embedding vectors, metadata, and usage information.

    Attributes:
        embeddings: List of generated embedding vectors
        model: Model name used to generate embeddings
        provider: Provider name that generated embeddings
        dims: Number of dimensions in each embedding
        total_tokens: Total tokens processed (if available)
    """

    embeddings: list[EmbeddingVector]
    model: ModelName
    provider: ProviderName
    dims: Dimensions
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        """Validate embedding result after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate embedding result attributes."""
        # Embeddings validation
        if not self.embeddings:
            raise ValidationError(
                "embeddings", self.embeddings, "Embeddings list cannot be empty"
            )

        # Provider validation
        if not self.provider or not self.provider.strip():
            raise ValidationError("provider", self.provider, "Provider cannot be empty")

        # Model validation
        if not self.model or not self.model.strip():
            raise ValidationError("model", self.model, "Model cannot be empty")

        # Dimensions validation
        if self.dims <= 0:
            raise ValidationError("dims", self.dims, "Dimensions must be positive")

        # Validate each embedding vector
        for i, embedding in enumerate(self.embeddings):
            if not embedding:
                raise ValidationError(
                    f"embeddings[{i}]", embedding, "Embedding vector cannot be empty"
                )

            if len(embedding) != self.dims:
                raise ValidationError(
                    f"embeddings[{i}]",
                    len(embedding),
                    f"Embedding vector length "
                    f"({len(embedding)}) must match "
                    f"dimensions ({self.dims})",
                )

        # Token count validation
        if self.total_tokens is not None and self.total_tokens < 0:
            raise ValidationError(
                "total_tokens", self.total_tokens, "Token count cannot be negative"
            )

    @property
    def count(self) -> int:
        """Get the number of embeddings in this result."""
        return len(self.embeddings)

    @property
    def provider_model_key(self) -> str:
        """Get a unique key combining provider and model."""
        return f"{self.provider}/{self.model}"

    def to_embeddings(self, chunk_ids: list[ChunkId]) -> list[Embedding]:
        """Convert result to a list of Embedding models.

        Args:
            chunk_ids: List of chunk IDs to associate with embeddings

        Returns:
            List of Embedding models

        Raises:
            ValidationError: If chunk_ids length doesn't match embeddings length
        """
        if len(chunk_ids) != len(self.embeddings):
            raise ValidationError(
                "chunk_ids",
                len(chunk_ids),
                f"Chunk IDs count ({len(chunk_ids)}) "
                f"must match embeddings count "
                f"({len(self.embeddings)})",
            )

        timestamp = datetime.utcnow()

        return [
            Embedding(
                chunk_id=chunk_id,
                provider=self.provider,
                model=self.model,
                dims=self.dims,
                vector=vector,
                created_at=timestamp,
            )
            for chunk_id, vector in zip(chunk_ids, self.embeddings)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert EmbeddingResult to dictionary.

        Returns:
            Dictionary representation of the embedding result
        """
        result = {
            "embeddings": self.embeddings,
            "model": self.model,
            "provider": self.provider,
            "dims": self.dims,
        }

        if self.total_tokens is not None:
            result["total_tokens"] = self.total_tokens

        return result

    def __str__(self) -> str:
        """Return string representation of the embedding result."""
        return (
            f"EmbeddingResult({self.count} embeddings, "
            f"{self.provider_model_key}, "
            f"dims={self.dims})"
        )

    def __repr__(self) -> str:
        """Return detailed string representation of the embedding result."""
        return (
            f"EmbeddingResult(count={self.count}, provider='{self.provider}', "
            f"model='{self.model}', dims={self.dims}, total_tokens={self.total_tokens})"
        )
