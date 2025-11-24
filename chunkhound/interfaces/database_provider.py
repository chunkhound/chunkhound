"""DatabaseProvider protocol for ChunkHound - abstract interface for database implementations."""

from pathlib import Path
from typing import Any, Protocol

from chunkhound.core.models import Chunk, Embedding, File


class DatabaseProvider(Protocol):
    """Abstract protocol for database providers.

    Defines the interface that all database implementations must follow.
    This enables pluggable database backends (DuckDB, PostgreSQL, SQLite, etc.)
    """

    @property
    def db_path(self) -> Path | str:
        """Database connection path or identifier."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if database connection is active."""
        ...

    # Connection Management
    def connect(self) -> None:
        """Establish database connection and initialize schema."""
        ...

    def disconnect(self) -> None:
        """Close database connection and cleanup resources."""
        ...

    # Schema Management
    def create_schema(self) -> None:
        """Create database schema for files, chunks, and embeddings."""
        ...

    def create_indexes(self) -> None:
        """Create database indexes for performance optimization."""
        ...

    def create_vector_index(
        self, provider: str, model: str, dims: int, metric: str = "cosine"
    ) -> None:
        """Create vector index for specific provider/model/dims combination."""
        ...

    def drop_vector_index(
        self, provider: str, model: str, dims: int, metric: str = "cosine"
    ) -> str:
        """Drop vector index for specific provider/model/dims combination."""
        ...

    # File Operations
    def insert_file(self, file: File) -> int:
        """Insert file record and return file ID."""
        ...

    def get_file_by_path(
        self, path: str, as_model: bool = False
    ) -> dict[str, Any] | File | None:
        """Get file record by path."""
        ...

    def get_file_by_id(
        self, file_id: int, as_model: bool = False
    ) -> dict[str, Any] | File | None:
        """Get file record by ID."""
        ...

    def update_file(self, file_id: int, **kwargs: Any) -> None:
        """Update file record with new values."""
        ...

    def delete_file_completely(self, file_path: str) -> bool:
        """Delete a file and all its chunks/embeddings completely."""
        ...

    # Chunk Operations
    def insert_chunk(self, chunk: Chunk) -> int:
        """Insert chunk record and return chunk ID."""
        ...

    def insert_chunks_batch(self, chunks: list[Chunk]) -> list[int]:
        """Insert multiple chunks in batch and return chunk IDs."""
        ...

    def get_chunk_by_id(
        self, chunk_id: int, as_model: bool = False
    ) -> dict[str, Any] | Chunk | None:
        """Get chunk record by ID."""
        ...

    def get_chunks_by_file_id(
        self, file_id: int, as_model: bool = False
    ) -> list[dict[str, Any] | Chunk]:
        """Get all chunks for a specific file."""
        ...

    def delete_file_chunks(self, file_id: int) -> None:
        """Delete all chunks for a file."""
        ...

    def delete_chunk(self, chunk_id: int) -> None:
        """Delete a single chunk by ID."""
        ...

    def update_chunk(self, chunk_id: int, **kwargs: Any) -> None:
        """Update chunk record with new values."""
        ...

    # Embedding Operations
    def insert_embedding(self, embedding: Embedding) -> int:
        """Insert embedding record and return embedding ID."""
        ...

    def insert_embeddings_batch(
        self,
        embeddings_data: list[dict],
        batch_size: int | None = None,
        connection: Any = None,
    ) -> int:
        """Insert multiple embedding vectors with optimization.

        Args:
            embeddings_data: List of embedding data dictionaries
            batch_size: Optional batch size for database operations (uses provider default if None)
            connection: Optional database connection to use (for transaction contexts)
        """
        ...

    def get_embedding_by_chunk_id(
        self, chunk_id: int, provider: str, model: str
    ) -> Embedding | None:
        """Get embedding for specific chunk, provider, and model."""
        ...

    def get_existing_embeddings(
        self, chunk_ids: list[int], provider: str, model: str
    ) -> set[int]:
        """Get set of chunk IDs that already have embeddings for given provider/model."""
        ...

    def delete_embeddings_by_chunk_id(self, chunk_id: int) -> None:
        """Delete all embeddings for a specific chunk."""
        ...

    def get_all_chunks_with_metadata(self) -> list[dict[str, Any]]:
        """Get all chunks with their metadata including file paths (provider-agnostic)."""
        ...

    # Search Operations
    def search_semantic(
        self,
        query_embedding: list[float],
        provider: str,
        model: str,
        page_size: int = 10,
        offset: int = 0,
        threshold: float | None = None,
        path_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform semantic vector search.

        Args:
            query_embedding: Query embedding vector
            provider: Embedding provider name
            model: Embedding model name
            page_size: Number of results per page
            offset: Starting position for pagination
            threshold: Optional similarity threshold
            path_filter: Optional relative path to limit search scope (e.g., 'src/', 'tests/')

        Returns:
            Tuple of (results, pagination_metadata)
        """
        ...

    def find_similar_chunks(
        self,
        chunk_id: int,
        provider: str,
        model: str,
        limit: int = 10,
        threshold: float | None = None,
        path_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find chunks similar to the given chunk using its embedding.

        Args:
            chunk_id: ID of the chunk to find similar chunks for
            provider: Embedding provider name
            model: Embedding model name
            limit: Maximum number of results to return
            threshold: Optional similarity threshold
            path_filter: Optional relative path to limit search scope

        Returns:
            List of similar chunks with scores and metadata
        """
        ...

    def search_by_embedding(
        self,
        query_embedding: list[float],
        provider: str,
        model: str,
        limit: int = 10,
        threshold: float | None = None,
        path_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find chunks similar to the given embedding vector.

        Args:
            query_embedding: The embedding vector to search with
            provider: Embedding provider name
            model: Embedding model name
            limit: Maximum number of results to return
            threshold: Optional similarity threshold
            path_filter: Optional relative path to limit search scope

        Returns:
            List of similar chunks with scores and metadata
        """
        ...

    def search_regex(
        self,
        pattern: str,
        page_size: int = 10,
        offset: int = 0,
        path_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform regex search on code content.

        Args:
            pattern: Regular expression pattern to search for
            page_size: Number of results per page
            offset: Starting position for pagination
            path_filter: Optional relative path to limit search scope (e.g., 'src/', 'tests/')

        Returns:
            Tuple of (results, pagination_metadata)
        """
        ...

    def search_text(
        self, query: str, page_size: int = 10, offset: int = 0
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform full-text search on code content.

        Returns:
            Tuple of (results, pagination_metadata)
        """
        ...

    # Statistics and Monitoring
    def get_stats(self) -> dict[str, int]:
        """Get database statistics (file count, chunk count, etc.)."""
        ...

    def get_file_stats(self, file_id: int) -> dict[str, Any]:
        """Get statistics for a specific file."""
        ...

    def get_provider_stats(self, provider: str, model: str) -> dict[str, Any]:
        """Get statistics for a specific embedding provider/model."""
        ...

    # Transaction and Bulk Operations
    def execute_query(
        self, query: str, params: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return results."""
        ...

    def begin_transaction(self) -> None:
        """Begin a database transaction."""
        ...

    def commit_transaction(self) -> None:
        """Commit the current transaction."""
        ...

    def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        ...

    # File Processing Integration
    async def process_file(
        self, file_path: Path, skip_embeddings: bool = False
    ) -> dict[str, Any]:
        """Process a file end-to-end: parse, chunk, and store in database."""
        ...

    async def process_directory(
        self,
        directory: Path,
        patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Process all supported files in a directory."""
        ...

    # Health and Diagnostics
    def optimize_tables(self) -> None:
        """Optimize tables by compacting fragments and rebuilding indexes (provider-specific)."""
        ...

    def health_check(self) -> dict[str, Any]:
        """Perform health check and return status information."""
        ...

    def get_connection_info(self) -> dict[str, Any]:
        """Get information about the database connection."""
        ...
