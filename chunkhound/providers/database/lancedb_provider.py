"""LanceDB provider implementation for ChunkHound - concrete database provider using LanceDB."""

import os
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from chunkhound.core.models import Chunk, Embedding, File
from chunkhound.core.types.common import ChunkType, Language

# Import existing components that will be used by the provider
from chunkhound.embeddings import EmbeddingManager
from chunkhound.providers.database.serial_database_provider import (
    SerialDatabaseProvider,
)
from chunkhound.utils.chunk_hashing import generate_chunk_id

# Type hinting only
if TYPE_CHECKING:
    from chunkhound.core.config.database_config import DatabaseConfig


# PyArrow schemas - avoiding LanceModel to prevent enum issues
def get_files_schema() -> pa.Schema:
    """Get PyArrow schema for files table."""
    return pa.schema(
        [
            ("id", pa.int64()),
            ("path", pa.string()),
            ("size", pa.int64()),
            ("modified_time", pa.float64()),
            ("content_hash", pa.string()),
            ("indexed_time", pa.float64()),
            ("language", pa.string()),
            ("encoding", pa.string()),
            ("line_count", pa.int64()),
        ]
    )


def get_chunks_schema(embedding_dims: int | None = None) -> pa.Schema:
    """Get PyArrow schema for chunks table.

    Args:
        embedding_dims: Number of dimensions for embedding vectors.
                       If None, uses variable-size list (which doesn't support vector search)
    """
    # Define embedding field based on whether we have fixed dimensions
    if embedding_dims is not None:
        embedding_field = pa.list_(pa.float32(), embedding_dims)  # Fixed-size list
    else:
        embedding_field = pa.list_(pa.float32())  # Variable-size list

    return pa.schema(
        [
            ("id", pa.int64()),
            ("file_id", pa.int64()),
            ("content", pa.string()),
            ("start_line", pa.int64()),
            ("end_line", pa.int64()),
            ("chunk_type", pa.string()),
            ("language", pa.string()),
            ("name", pa.string()),
            ("embedding", embedding_field),
            ("provider", pa.string()),
            ("model", pa.string()),
            ("created_time", pa.float64()),
        ]
    )


def _has_valid_embedding(x: Any) -> bool:
    """Check if embedding is valid (not None, not empty, not all zeros).

    Handles both list and numpy array embeddings. Zero-vector detection
    provides defense-in-depth for legacy placeholder vectors.
    """
    if not hasattr(x, "__len__"):
        return False
    if x is None or not isinstance(x, (list, np.ndarray)) or len(x) == 0:
        return False
    # Check not all zeros (legacy placeholder detection)
    if isinstance(x, np.ndarray):
        return np.any(x != 0)
    return any(v != 0 for v in x)


def _deduplicate_by_id(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate LanceDB results by 'id' field, preserving order.

    LanceDB queries may return duplicates across table fragments until
    compaction runs. This helper ensures each chunk_id appears once.

    Args:
        results: List of result dictionaries with 'id' field

    Returns:
        Deduplicated list (first occurrence wins, preserves order)
    """
    if not results:
        return results

    seen: set[int] = set()
    unique: list[dict[str, Any]] = []

    for result in results:
        chunk_id = result.get("id")
        if chunk_id is not None and chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(result)

    return unique


class LanceDBProvider(SerialDatabaseProvider):
    """LanceDB implementation using serial executor pattern.

    Optimized for performance: Uses native LanceDB queries instead of pandas DataFrame filtering
    wherever possible to reduce memory usage and improve query performance for large datasets.
    All major query operations (_executor_get_existing_embeddings, _executor_search_regex,
    _executor_get_stats, execute_query, _executor_get_chunk_ids_without_embeddings_paginated)
    have been optimized to avoid loading entire tables into memory.
    """

    def __init__(
        self,
        db_path: Path | str,
        base_directory: Path,
        embedding_manager: EmbeddingManager | None = None,
        config: "DatabaseConfig | None" = None,
    ):
        """Initialize LanceDB provider.

        Args:
            db_path: Path to LanceDB database directory (already includes .lancedb suffix from DatabaseConfig.get_db_path())
            base_directory: Base directory for path normalization
            embedding_manager: Optional embedding manager for vector generation
            config: Database configuration for provider-specific settings
        """
        # Database path expected from DatabaseConfig.get_db_path() with .lancedb suffix
        # Ensure it's absolute to avoid LanceDB internal path resolution issues
        absolute_db_path = Path(db_path).absolute()

        # Initialize base class
        super().__init__(absolute_db_path, base_directory, embedding_manager, config)

        self.index_type = config.lancedb_index_type if config else None
        self._fragment_threshold = (
            config.lancedb_optimize_fragment_threshold if config else 100
        )
        self.connection: Any | None = (
            None  # For backward compatibility only - do not use directly
        )

        # Table references
        self._files_table = None
        self._chunks_table = None

        # Query result caches (cleared when data is modified)
        self._file_path_cache = {}
        self._chunk_id_cache = {}

        # Performance monitoring
        self._query_performance = {}

    def _create_connection(self) -> Any:
        """Create and return a LanceDB connection.

        This method is called from within the executor thread to create
        a thread-local connection.

        Returns:
            LanceDB connection object
        """
        import lancedb
        import logging as python_logging

        # Enable LanceDB internal logging for debugging timeouts
        lancedb_logger = python_logging.getLogger('lancedb')
        lancedb_logger.setLevel(python_logging.DEBUG)

        # Also enable lower-level Lance logging if available
        lance_logger = python_logging.getLogger('lance')
        lance_logger.setLevel(python_logging.DEBUG)

        abs_db_path = self._db_path

        # Save CWD (thread-safe in executor)
        original_cwd = os.getcwd()
        try:
            os.chdir(abs_db_path.parent)
            conn = lancedb.connect(abs_db_path.name)
            return conn
        finally:
            os.chdir(original_cwd)

    def _get_schema_sql(self) -> list[str] | None:
        """LanceDB doesn't use SQL - return None."""
        return None

    def _executor_connect(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for connect - runs in DB thread.

        Note: The connection is already created by _create_connection,
        so this method ensures schema and indexes are created.
        """
        try:
            # Store connection reference for backward compatibility
            self.connection = conn

            # Create schema and indexes in executor thread
            self._executor_create_schema(conn, state)
            self._executor_create_indexes(conn, state)

            logger.info(f"Connected to LanceDB at {self._db_path}")
        except Exception as e:
            logger.error(f"Error in LanceDB connect: {e}")
            raise

    def _executor_disconnect(
        self, conn: Any, state: dict[str, Any], skip_checkpoint: bool
    ) -> None:
        """Executor method for disconnect - runs in DB thread."""
        try:
            # Clear connection and table references
            self.connection = None
            self._files_table = None
            self._chunks_table = None

            # Connection will be closed by base class
            logger.info("Disconnected from LanceDB")
        except Exception as e:
            logger.error(f"Error in LanceDB disconnect: {e}")
            raise

    def _get_embedding_dimensions_safe(self) -> int | None:
        """Safely retrieve embedding dimensions from embedding manager.

        Returns:
            Embedding dimensions if available, None otherwise.

        Notes:
            Handles all edge cases gracefully:
            - No embedding_manager configured
            - No default provider registered
            - Provider missing .dims attribute
            - Provider.dims raises exception
        """
        if self.embedding_manager is None:
            logger.debug("No embedding_manager configured - using variable-size schema")
            return None

        try:
            provider = self.embedding_manager.get_default_provider()
            if provider is None:
                logger.debug("No default embedding provider - using variable-size schema")
                return None

            dims = provider.dims
            if not isinstance(dims, int) or dims <= 0:
                logger.warning(
                    f"Invalid embedding dimensions: {dims} - using variable-size schema"
                )
                return None

            logger.debug(
                f"Detected embedding dimensions: {dims} "
                f"(provider={provider.name}, model={provider.model})"
            )
            return dims

        except AttributeError:
            logger.debug("Embedding provider has no 'dims' attribute - using variable-size schema")
            return None
        except Exception as e:
            logger.warning(f"Error detecting embedding dimensions: {e} - using variable-size schema")
            return None

    def create_schema(self) -> None:
        """Create database schema for files, chunks, and embeddings."""
        return self._execute_in_db_thread_sync("create_schema")

    def _executor_create_schema(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for create_schema - runs in DB thread."""
        # Create files table if it doesn't exist
        try:
            self._files_table = conn.open_table("files")
        except Exception:
            # Table doesn't exist, create it
            # Create table using PyArrow schema
            self._files_table = conn.create_table("files", schema=get_files_schema())
            logger.info("Created files table")

        # Create chunks table if it doesn't exist
        try:
            self._chunks_table = conn.open_table("chunks")
            logger.debug("Opened existing chunks table")
        except Exception:
            # Table doesn't exist, create it
            # Try to get embedding dimensions to avoid migration later
            embedding_dims = self._get_embedding_dimensions_safe()

            if embedding_dims is not None:
                logger.info(
                    f"Creating chunks table with fixed-size embedding schema "
                    f"({embedding_dims} dimensions) - no migration will be needed"
                )
            else:
                logger.info(
                    "Creating chunks table with variable-size embedding schema - "
                    "table will be migrated when first embeddings are inserted"
                )

            self._chunks_table = conn.create_table(
                "chunks", schema=get_chunks_schema(embedding_dims)
            )
            logger.info("Created chunks table")

    def create_indexes(self) -> None:
        """Create database indexes for performance optimization."""
        return self._execute_in_db_thread_sync("create_indexes")

    def _executor_create_indexes(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for create_indexes - runs in DB thread."""
        # Validate LanceDB version for ivf_rq index type
        if self.index_type == "ivf_rq":
            import lancedb
            from packaging.version import Version

            lancedb_version = getattr(lancedb, "__version__", "0.0.0")
            if Version(lancedb_version) < Version("0.25.3"):
                raise ValueError(
                    f"ivf_rq index type requires LanceDB 0.25.3+, but found {lancedb_version}. "
                    f"Please upgrade: pip install 'lancedb>=0.25.3'"
                )

        # Create scalar index on chunks.id for merge_insert performance
        # merge_insert performs join on id column, index provides O(log n) vs O(n) lookup
        if self._chunks_table:
            try:
                # Check if index already exists
                indices = self._chunks_table.list_indices()
                has_id_index = any(
                    idx.columns == ["id"] or "id" in idx.columns
                    for idx in indices
                )

                if not has_id_index:
                    logger.info("Creating scalar index on chunks.id for merge_insert performance")
                    self._chunks_table.create_scalar_index("id")
                    logger.info("Scalar index on chunks.id created successfully")
                else:
                    logger.debug("Scalar index on chunks.id already exists")
            except Exception as e:
                # Non-fatal: merge_insert works without index, just slower
                logger.warning(
                    f"Could not create scalar index on chunks.id: {e}. "
                    f"This is non-fatal but may slow down merge_insert operations. "
                    f"Check LanceDB version supports create_scalar_index()."
                )

    def create_vector_index(
        self, provider: str, model: str, dims: int, metric: str = "cosine"
    ) -> None:
        """Create vector index for specific provider/model/dims combination."""
        return self._execute_in_db_thread_sync(
            "create_vector_index", provider, model, dims, metric
        )

    def _executor_create_vector_index(
        self,
        conn: Any,
        state: dict[str, Any],
        provider: str,
        model: str,
        dims: int,
        metric: str = "cosine",
    ) -> None:
        """Executor method for create_vector_index - runs in DB thread."""
        if not self._chunks_table:
            return

        try:
            # Check if index already exists by attempting a simple search
            try:
                test_vector = [0.0] * dims
                self._chunks_table.search(
                    test_vector, vector_column_name="embedding"
                ).limit(1).to_list()
                logger.debug(f"Vector index already exists for {provider}/{model}")
                return
            except Exception:
                # Index doesn't exist, create it
                pass

            # Verify sufficient data exists for IVF PQ training
            total_embeddings = len(
                self._executor_get_existing_embeddings(conn, state, [], provider, model)
            )
            if total_embeddings < 1000:
                logger.debug(
                    f"Skipping index creation for {provider}/{model}: insufficient data ({total_embeddings} < 1000)"
                )
                return

            # Create vector index (wait_timeout not supported in LanceDB OSS)
            if self.index_type == "ivf_hnsw_sq":
                self._chunks_table.create_index(
                    vector_column_name="embedding",
                    index_type="IVF_HNSW_SQ",
                    metric=metric,
                )
            elif self.index_type == "ivf_rq":
                self._chunks_table.create_index(
                    vector_column_name="embedding",
                    index_type="IVF_RQ",
                    metric=metric,
                )
            else:
                # Default to auto-configured index with explicit vector column
                self._chunks_table.create_index(
                    vector_column_name="embedding", metric=metric
                )
            logger.debug(
                f"Created vector index for {provider}/{model} with metric={metric}"
            )
        except Exception as e:
            logger.debug(f"Failed to create vector index for {provider}/{model}: {e}")

    def drop_vector_index(
        self, provider: str, model: str, dims: int, metric: str = "cosine"
    ) -> str:
        """Drop vector index for specific provider/model/dims combination."""
        # LanceDB handles index management automatically
        return "Index management handled automatically by LanceDB"

    def _generate_chunk_id_safe(self, chunk: Chunk) -> int:
        """Generate chunk ID with fallback to hash-based ID.

        Returns chunk.id if present, otherwise generates deterministic
        hash-based ID from file_id, content, and chunk type.

        Args:
            chunk: Chunk object to generate ID for

        Returns:
            Chunk ID (existing or generated)
        """
        return chunk.id or generate_chunk_id(
            chunk.file_id,
            chunk.code or "",
            concept=str(
                chunk.chunk_type.value
                if hasattr(chunk.chunk_type, "value")
                else chunk.chunk_type
            ),
        )

    # File Operations
    def insert_file(self, file: File) -> int:
        """Insert file record and return file ID."""
        return self._execute_in_db_thread_sync("insert_file", file)

    def _executor_insert_file(
        self, conn: Any, state: dict[str, Any], file: File
    ) -> int:
        """Executor method for insert_file - runs in DB thread."""
        if not self._files_table:
            self._executor_create_schema(conn, state)

        # Store path as-is (now relative with forward slashes from IndexingCoordinator)
        normalized_path = file.path

        # Prepare file data
        file_data = {
            "id": file.id or int(time.time() * 1000000),
            "path": normalized_path,
            "size": file.size_bytes,
            "modified_time": file.mtime,
            "content_hash": getattr(file, "content_hash", None) or "",
            "indexed_time": time.time(),
            "language": str(
                file.language.value
                if hasattr(file.language, "value")
                else file.language
            ),
            "encoding": "utf-8",
            "line_count": 0,
        }

        # Use merge_insert for atomic upsert based on path
        # This eliminates the TOCTOU race condition by making the
        # check-and-insert/update operation atomic at the database level
        self._files_table.merge_insert(
            "path"
        ).when_matched_update_all().when_not_matched_insert_all().execute([file_data])

        # Get the file ID (either newly inserted or existing)
        # We need to query back because merge_insert doesn't return the ID
        result = (
            self._files_table.search().where(f"path = '{normalized_path}'").to_list()
        )
        if result:
            return result[0]["id"]
        else:
            # This should not happen, but handle gracefully
            logger.error(
                f"Failed to retrieve file ID after merge_insert for path: {normalized_path}"
            )
            return file_data["id"]

    def get_file_by_path(
        self, path: str, as_model: bool = False
    ) -> dict[str, Any] | File | None:
        """Get file record by path."""
        # Use caching for frequent file lookups
        cache_key = f"file_path:{path}:{as_model}"
        cached_result = getattr(self, '_file_path_cache', {}).get(cache_key)
        if cached_result is not None:
            return cached_result

        result = self._execute_in_db_thread_sync("get_file_by_path", path, as_model)

        # Cache the result (simple dict cache, not LRU for thread safety)
        if not hasattr(self, '_file_path_cache'):
            self._file_path_cache = {}
        self._file_path_cache[cache_key] = result

        return result

    def _executor_get_file_by_path(
        self, conn: Any, state: dict[str, Any], path: str, as_model: bool = False
    ) -> dict[str, Any] | File | None:
        """Executor method for get_file_by_path - runs in DB thread."""
        if not self._files_table:
            return None

        try:
            # Normalize path to handle both absolute and relative paths
            from chunkhound.core.utils import normalize_path_for_lookup

            base_dir = state.get("base_directory")
            normalized_path = normalize_path_for_lookup(path, base_dir)
            results = (
                self._files_table.search()
                .where(f"path = '{normalized_path}'")
                .to_list()
            )
            if not results:
                return None

            result = results[0]
            if as_model:
                return File(
                    id=result["id"],
                    path=result["path"],
                    size_bytes=result["size"],
                    mtime=result["modified_time"],
                    language=Language(result["language"]),
                )
            return result
        except Exception as e:
            logger.error(f"Error getting file by path: {e}")
            return None

    def get_file_by_id(
        self, file_id: int, as_model: bool = False
    ) -> dict[str, Any] | File | None:
        """Get file record by ID."""
        return self._execute_in_db_thread_sync("get_file_by_id", file_id, as_model)

    def _executor_get_file_by_id(
        self, conn: Any, state: dict[str, Any], file_id: int, as_model: bool = False
    ) -> dict[str, Any] | File | None:
        """Executor method for get_file_by_id - runs in DB thread."""
        if not self._files_table:
            return None

        try:
            results = self._files_table.search().where(f"id = {file_id}").to_list()
            if not results:
                return None

            result = results[0]
            if as_model:
                return File(
                    id=result["id"],
                    path=result["path"],
                    size_bytes=result["size"],
                    mtime=result["modified_time"],
                    language=Language(result["language"]),
                )
            return result
        except Exception as e:
            logger.error(f"Error getting file by ID: {e}")
            return None

    def _clear_query_caches(self) -> None:
        """Clear query result caches when data is modified."""
        self._file_path_cache.clear()
        self._chunk_id_cache.clear()

    def update_file(
        self,
        file_id: int,
        size_bytes: int | None = None,
        mtime: float | None = None,
        content_hash: str | None = None,
        **kwargs,
    ) -> None:
        """Update file record with new values."""
        result = self._execute_in_db_thread_sync(
            "update_file", file_id, size_bytes, mtime, content_hash
        )
        # Clear caches since file data changed
        self._clear_query_caches()
        return result

    def _executor_update_file(
        self,
        conn: Any,
        state: dict[str, Any],
        file_id: int,
        size_bytes: int | None = None,
        mtime: float | None = None,
        content_hash: str | None = None,
        **kwargs,
    ) -> None:
        """Executor method for update_file - runs in DB thread."""
        if not self._files_table:
            return

        try:
            # Get existing file record
            existing_file = self._executor_get_file_by_id(conn, state, file_id, False)
            if not existing_file:
                return

            # Update the relevant fields
            updated_file = dict(existing_file)
            if size_bytes is not None:
                updated_file["size"] = size_bytes
            if mtime is not None:
                updated_file["modified_time"] = mtime
            if content_hash is not None:
                updated_file["content_hash"] = content_hash
            updated_file["indexed_time"] = time.time()

            # LanceDB doesn't support in-place updates, so we use merge_insert
            # This updates the record by matching on the 'id' field
            self._files_table.merge_insert("id").when_matched_update_all().execute(
                [updated_file]
            )

        except Exception as e:
            logger.error(f"Error updating file {file_id}: {e}")

    def delete_file_completely(self, file_path: str) -> bool:
        """Delete a file and all its chunks/embeddings completely."""
        return self._execute_in_db_thread_sync("delete_file_completely", file_path)

    def _executor_delete_file_completely(
        self, conn: Any, state: dict[str, Any], file_path: str
    ) -> bool:
        """Executor method for delete_file_completely - runs in DB thread."""
        try:
            # Get file record in the executor thread
            file_record = self._executor_get_file_by_path(conn, state, file_path, False)
            if not file_record:
                return False

            file_id = file_record["id"]

            # Delete chunks first
            if self._chunks_table:
                self._chunks_table.delete(f"file_id = {file_id}")

            # Delete file record
            if self._files_table:
                self._files_table.delete(f"id = {file_id}")

            return True
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False

    # Chunk Operations
    def insert_chunk(self, chunk: Chunk) -> int:
        """Insert chunk record and return chunk ID."""
        return self._execute_in_db_thread_sync("insert_chunk", chunk)

    def _executor_insert_chunk(
        self, conn: Any, state: dict[str, Any], chunk: Chunk
    ) -> int:
        """Executor method for insert_chunk - runs in DB thread."""
        if not self._chunks_table:
            self._executor_create_schema(conn, state)

        chunk_data = {
            "id": self._generate_chunk_id_safe(chunk),
            "file_id": chunk.file_id,
            "content": chunk.code or "",
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "chunk_type": str(
                chunk.chunk_type.value
                if hasattr(chunk.chunk_type, "value")
                else chunk.chunk_type
            ),
            "language": str(
                chunk.language.value
                if hasattr(chunk.language, "value")
                else chunk.language
            ),
            "name": chunk.symbol or "",
            "embedding": None,
            "provider": "",
            "model": "",
            "created_time": time.time(),
        }

        # Use PyArrow Table directly to avoid LanceDB DataFrame schema alignment bug
        # Convert single item to proper format for pa.table
        chunk_data_list = [chunk_data]
        chunk_table = pa.Table.from_pylist(chunk_data_list, schema=get_chunks_schema())

        # Use merge_insert for atomic upsert with conflict-free semantics
        # Handles idempotency (same file indexed multiple times) and concurrent writes
        # (initial scan + file watcher, multiple processes, multiple file events)
        # LanceDB's MVCC ensures conflicts are resolved via automatic retries
        (
            self._chunks_table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(chunk_table)
        )
        return chunk_data["id"]

    def insert_chunks_batch(self, chunks: list[Chunk]) -> list[int]:
        """Insert multiple chunks in batch using optimized DataFrame operations."""
        return self._execute_in_db_thread_sync("insert_chunks_batch", chunks)

    def _executor_insert_chunks_batch(
        self, conn: Any, state: dict[str, Any], chunks: list[Chunk]
    ) -> list[int]:
        """Executor method for insert_chunks_batch - runs in DB thread."""
        if not chunks:
            return []

        if not self._chunks_table:
            self._executor_create_schema(conn, state)

        # PERFORMANCE PROFILING: Track total batch operation time
        batch_start_time = time.time()

        # Process in optimal batch sizes (LanceDB best practice: 1000+ items)
        batch_size = 1000
        all_chunk_ids = []

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]

            # PERFORMANCE PROFILING: Track individual sub-batch time
            sub_batch_start = time.time()

            chunk_data_list = []
            chunk_ids = []

            for chunk in batch_chunks:
                chunk_id = self._generate_chunk_id_safe(chunk)
                chunk_ids.append(chunk_id)

                chunk_data = {
                    "id": chunk_id,
                    "file_id": chunk.file_id,
                    "content": chunk.code or "",
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": str(
                        chunk.chunk_type.value
                        if hasattr(chunk.chunk_type, "value")
                        else chunk.chunk_type
                    ),
                    "language": str(
                        chunk.language.value
                        if hasattr(chunk.language, "value")
                        else chunk.language
                    ),
                    "name": chunk.symbol or "",
                    "embedding": None,
                    "provider": "",
                    "model": "",
                    "created_time": time.time(),
                }
                chunk_data_list.append(chunk_data)

            # PERFORMANCE PROFILING: Track table creation time
            table_create_start = time.time()

            # Use PyArrow Table directly to avoid LanceDB DataFrame schema alignment bug
            chunks_table = pa.Table.from_pylist(
                chunk_data_list, schema=get_chunks_schema()
            )

            table_create_time = time.time() - table_create_start

            # PERFORMANCE PROFILING: Track merge_insert time
            merge_start = time.time()

            # Use merge_insert for atomic upsert with conflict-free semantics
            # Handles idempotency (same file indexed multiple times) and concurrent writes
            # (initial scan + file watcher, multiple processes, multiple file events)
            # LanceDB's MVCC ensures conflicts are resolved via automatic retries
            (
                self._chunks_table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(chunks_table)
            )

            merge_time = time.time() - merge_start
            sub_batch_total = time.time() - sub_batch_start

            all_chunk_ids.extend(chunk_ids)

            logger.debug(
                f"Bulk inserted batch of {len(batch_chunks)} chunks in {sub_batch_total:.3f}s "
                f"(table_create: {table_create_time:.3f}s, merge_insert: {merge_time:.3f}s)"
            )

        total_batch_time = time.time() - batch_start_time
        logger.debug(f"Completed bulk insert of {len(chunks)} chunks in {total_batch_time:.3f}s total")
        return all_chunk_ids

    def get_chunk_by_id(
        self, chunk_id: int, as_model: bool = False
    ) -> dict[str, Any] | Chunk | None:
        """Get chunk record by ID."""
        # Use caching for frequent chunk lookups
        cache_key = f"chunk_id:{chunk_id}:{as_model}"
        cached_result = getattr(self, '_chunk_id_cache', {}).get(cache_key)
        if cached_result is not None:
            return cached_result

        result = self._execute_in_db_thread_sync("get_chunk_by_id", chunk_id, as_model)

        # Cache the result
        if not hasattr(self, '_chunk_id_cache'):
            self._chunk_id_cache = {}
        self._chunk_id_cache[cache_key] = result

        return result

    def _executor_get_chunk_by_id(
        self, conn: Any, state: dict[str, Any], chunk_id: int, as_model: bool = False
    ) -> dict[str, Any] | Chunk | None:
        """Executor method for get_chunk_by_id - runs in DB thread."""
        if not self._chunks_table:
            return None

        try:
            results = self._chunks_table.search().where(f"id = {chunk_id}").to_list()
            if not results:
                return None

            result = results[0]
            if as_model:
                return Chunk(
                    id=result["id"],
                    file_id=result["file_id"],
                    code=result["content"],
                    start_line=result["start_line"],
                    end_line=result["end_line"],
                    chunk_type=ChunkType(result["chunk_type"]),
                    language=Language(result["language"]),
                    symbol=result["name"],
                )
            return result
        except Exception as e:
            logger.error(f"Error getting chunk by ID: {e}")
            return None

    def get_chunks_by_file_id(
        self, file_id: int, as_model: bool = False
    ) -> list[dict[str, Any] | Chunk]:
        """Get all chunks for a specific file."""
        return self._execute_in_db_thread_sync(
            "get_chunks_by_file_id", file_id, as_model
        )

    def _executor_get_chunks_by_file_id(
        self, conn: Any, state: dict[str, Any], file_id: int, as_model: bool = False
    ) -> list[dict[str, Any] | Chunk]:
        """Executor method for get_chunks_by_file_id - runs in DB thread."""
        if not self._chunks_table:
            return []

        try:
            results = (
                self._chunks_table.search().where(f"file_id = {file_id}").to_list()
            )
            # Deduplicate across fragments
            results = _deduplicate_by_id(results)

            if as_model:
                return [
                    Chunk(
                        id=result["id"],
                        file_id=result["file_id"],
                        code=result["content"],
                        start_line=result["start_line"],
                        end_line=result["end_line"],
                        chunk_type=ChunkType(result["chunk_type"]),
                        language=Language(result["language"]),
                        symbol=result["name"],
                    )
                    for result in results
                ]
            return results
        except Exception as e:
            logger.error(f"Error getting chunks by file ID: {e}")
            return []

    def delete_file_chunks(self, file_id: int) -> None:
        """Delete all chunks for a file."""
        return self._execute_in_db_thread_sync("delete_file_chunks", file_id)

    def _executor_delete_file_chunks(
        self, conn: Any, state: dict[str, Any], file_id: int
    ) -> None:
        """Executor method for delete_file_chunks - runs in DB thread."""
        if self._chunks_table:
            try:
                self._chunks_table.delete(f"file_id = {file_id}")
            except Exception as e:
                logger.error(f"Error deleting chunks for file {file_id}: {e}")

    def delete_chunk(self, chunk_id: int) -> None:
        """Delete a single chunk by ID."""
        return self._execute_in_db_thread_sync("delete_chunk", chunk_id)

    def _executor_delete_chunk(
        self, conn: Any, state: dict[str, Any], chunk_id: int
    ) -> None:
        """Executor method for delete_chunk - runs in DB thread."""
        if self._chunks_table:
            try:
                self._chunks_table.delete(f"id = {chunk_id}")
            except Exception as e:
                logger.error(f"Error deleting chunk {chunk_id}: {e}")

    def update_chunk(self, chunk_id: int, **kwargs) -> None:
        """Update chunk record with new values."""
        # LanceDB doesn't support in-place updates, need to implement via delete/insert
        pass

    # Embedding Operations
    def insert_embedding(self, embedding: Embedding) -> int:
        """Insert embedding record and return embedding ID."""
        # In LanceDB, embeddings are stored directly in the chunks table
        # This is a no-op since we use insert_embeddings_batch for efficiency
        return embedding.id or 0

    def insert_embeddings_batch(
        self,
        embeddings_data: list[dict],
        batch_size: int | None = None,
        connection=None,
    ) -> int:
        """Insert multiple embedding vectors efficiently using merge_insert."""
        return self._execute_in_db_thread_sync(
            "insert_embeddings_batch", embeddings_data, batch_size
        )

    def _executor_insert_embeddings_batch(
        self,
        conn: Any,
        state: dict[str, Any],
        embeddings_data: list[dict],
        batch_size: int | None = None,
    ) -> int:
        """Executor method for insert_embeddings_batch - runs in DB thread."""
        if not embeddings_data or not self._chunks_table:
            return 0

        try:
            # Determine embedding dimensions from the first embedding
            first_embedding = embeddings_data[0].get(
                "embedding", embeddings_data[0].get("vector")
            )
            if not first_embedding:
                logger.error("No embedding data found in first record")
                return 0

            embedding_dims = len(first_embedding)
            provider = embeddings_data[0]["provider"]
            model = embeddings_data[0]["model"]

            # Check if embedding columns exist in schema and if they have the correct type
            current_schema = self._chunks_table.schema
            embedding_field = None
            for field in current_schema:
                if field.name == "embedding":
                    embedding_field = field
                    break

            # Check if we need to recreate the table due to schema mismatch
            needs_recreation = False
            if embedding_field:
                # Check if it's a fixed-size list with correct dimensions
                if not pa.types.is_fixed_size_list(embedding_field.type):
                    logger.info(
                        "Embedding column exists but is variable-size list - need to recreate table with fixed-size list"
                    )
                    needs_recreation = True
                elif (
                    hasattr(embedding_field.type, "list_size")
                    and embedding_field.type.list_size != embedding_dims
                ):
                    logger.info(
                        f"Embedding column exists but has wrong dimensions ({embedding_field.type.list_size} vs {embedding_dims}) - need to recreate table"
                    )
                    needs_recreation = True

            if needs_recreation:
                # Need to recreate table with proper fixed-size schema
                existing_data_df = self._chunks_table.to_pandas()
                logger.info(
                    f"Migrating chunks table to fixed-size embedding schema:\n"
                    f"  Reason: Schema created before embedding dimensions were known\n"
                    f"  Required dimensions: {embedding_dims} (provider={provider}, model={model})\n"
                    f"  Chunks to migrate: {len(existing_data_df):,}\n"
                    f"  This is a ONE-TIME operation - future embedding insertions will be fast\n"
                    f"  To avoid this in future: Ensure embedding provider configured before database creation"
                )

                # Drop the old table
                conn.drop_table("chunks")

                # Create new table with proper schema
                new_schema = get_chunks_schema(embedding_dims)
                self._chunks_table = conn.create_table("chunks", schema=new_schema)
                logger.info("Created new chunks table with fixed-size embedding schema")

                # Re-insert existing data (without embeddings - they'll be added below)
                if len(existing_data_df) > 0:
                    # Prepare data for reinsertion
                    chunks_to_restore = []
                    for _, row in existing_data_df.iterrows():
                        chunk_data = {
                            "id": row["id"],
                            "file_id": row["file_id"],
                            "content": row["content"],
                            "start_line": row["start_line"],
                            "end_line": row["end_line"],
                            "chunk_type": row["chunk_type"],
                            "language": row["language"],
                            "name": row["name"],
                            "embedding": None,  # No placeholder - needs embedding generation
                            "provider": "",
                            "model": "",
                            "created_time": row.get("created_time", time.time()),
                        }
                        chunks_to_restore.append(chunk_data)

                    # Insert in batches
                    restore_batch_size = 1000
                    for i in range(0, len(chunks_to_restore), restore_batch_size):
                        batch = chunks_to_restore[i : i + restore_batch_size]
                        restore_table = pa.Table.from_pylist(batch, schema=new_schema)
                        self._chunks_table.add(restore_table, mode="append")

                    logger.info(
                        f"Restored {len(chunks_to_restore)} chunks to new table"
                    )

            elif not embedding_field:
                # Add embedding columns to the table if they don't exist
                logger.debug("Adding embedding columns to chunks table")
                # Create a proper fixed-size list type for the embedding column
                embedding_type = pa.list_(pa.float32(), embedding_dims)
                self._chunks_table.add_columns(
                    {
                        "embedding": f"arrow_cast(NULL, '{embedding_type}')",
                        "provider": "arrow_cast(NULL, 'string')",
                        "model": "arrow_cast(NULL, 'string')",
                    }
                )

            # Determine optimal batch size if not provided
            if batch_size is None:
                # Use smaller batches to reduce reading overhead and prevent timeouts
                batch_size = min(2000, len(embeddings_data))

            total_updated = 0

            # Process in batches for better memory management
            # Use read-modify-write pattern: LanceDB's when_matched_update_all()
            # requires ALL columns in source data to match target schema
            for i in range(0, len(embeddings_data), batch_size):
                batch = embeddings_data[i : i + batch_size]

                # Build lookup of chunk_id -> embedding data
                embedding_lookup = {}
                for e in batch:
                    embedding = e.get("embedding", e.get("vector"))
                    # Ensure embedding is a list
                    if hasattr(embedding, "tolist"):
                        embedding = embedding.tolist()
                    elif not isinstance(embedding, list):
                        embedding = list(embedding)
                    embedding_lookup[e["chunk_id"]] = {
                        "embedding": embedding,
                        "provider": e["provider"],
                        "model": e["model"],
                    }

                # Read existing rows for these chunk IDs
                # NOTE: Using Lance SQL filter instead of .search() because .search()
                # may not reliably find rows with NULL embedding columns (vector search semantics)
                chunk_ids = list(embedding_lookup.keys())
                chunk_ids_str = ','.join(map(str, chunk_ids))

                # OPTIMIZATION: Use native LanceDB queries instead of pandas filtering
                # This avoids loading data into pandas DataFrames and reduces memory usage
                try:
                    # Primary: Use LanceDB's native Lance filter (efficient for large tables)
                    existing_df = (
                        self._chunks_table.to_lance()
                        .to_table(filter=f"id IN ({chunk_ids_str})")
                        .to_pandas()
                    )
                except Exception as lance_err:
                    # OPTIMIZATION: Use direct LanceDB search queries instead of pandas pagination
                    # This is more efficient than loading data into pandas for filtering
                    logger.warning(
                        f"Lance SQL filter unavailable, using optimized search fallback. "
                        f"Error: {lance_err}"
                    )

                    # Use direct search queries instead of pandas filtering
                    existing_rows = []
                    chunk_ids_set = set(chunk_ids)

                    # For smaller batches, use direct queries
                    if len(chunk_ids) <= 1000:
                        try:
                            # Use LanceDB's native search with IN clause
                            results = self._chunks_table.search().where(f"id IN ({chunk_ids_str})").to_list()
                            existing_rows = results
                        except Exception:
                            # Fallback to individual queries for very large IN clauses
                            for chunk_id in chunk_ids:
                                try:
                                    result = self._chunks_table.search().where(f"id = {chunk_id}").to_list()
                                    if result:
                                        existing_rows.extend(result)
                                except Exception:
                                    continue
                    else:
                        # For larger sets, use streaming approach with smaller batches
                        batch_size = 500
                        for i in range(0, len(chunk_ids), batch_size):
                            batch_chunk_ids = chunk_ids[i:i + batch_size]
                            batch_ids_str = ','.join(map(str, batch_chunk_ids))
                            try:
                                batch_results = self._chunks_table.search().where(f"id IN ({batch_ids_str})").to_list()
                                existing_rows.extend(batch_results)
                            except Exception:
                                # Individual queries as last resort
                                for chunk_id in batch_chunk_ids:
                                    try:
                                        result = self._chunks_table.search().where(f"id = {chunk_id}").to_list()
                                        if result:
                                            existing_rows.extend(result)
                                    except Exception:
                                        continue

                    # Convert to DataFrame for consistent downstream handling
                    existing_df = pd.DataFrame(existing_rows) if existing_rows else pd.DataFrame()

                # Diagnostic logging
                logger.debug(f"Looking for {len(chunk_ids)} chunk IDs, found {len(existing_df)} existing chunks")
                if len(existing_df) == 0 and len(chunk_ids) > 0:
                    total_rows = self._chunks_table.count_rows()
                    logger.warning(
                        f"Embedding update: search returned 0 results but table has {total_rows} rows. "
                        f"This indicates a LanceDB query issue. Using paginated fallback. "
                        f"Chunk IDs requested: {chunk_ids[:5]}{'...' if len(chunk_ids) > 5 else ''}"
                    )

                # Merge embedding data into existing rows (full row data required)
                merge_data = []
                for _, row in existing_df.iterrows():
                    chunk_id = row["id"]
                    if chunk_id in embedding_lookup:
                        emb_data = embedding_lookup[chunk_id]
                        merge_data.append(
                            {
                                "id": row["id"],
                                "file_id": row["file_id"],
                                "content": row["content"],
                                "start_line": row["start_line"],
                                "end_line": row["end_line"],
                                "chunk_type": row["chunk_type"],
                                "language": row["language"],
                                "name": row["name"],
                                "embedding": emb_data["embedding"],
                                "provider": emb_data["provider"],
                                "model": emb_data["model"],
                                "created_time": row["created_time"],
                            }
                        )

                # merge_insert with PyArrow table to avoid nullable field mismatches
                # (see LanceDB GitHub issue #2366)
                if merge_data:
                    merge_table = pa.Table.from_pylist(
                        merge_data, schema=get_chunks_schema(embedding_dims)
                    )
                    (
                        self._chunks_table.merge_insert("id")
                        .when_matched_update_all()
                        .execute(merge_table)
                    )

                total_updated += len(merge_data)

                if len(embeddings_data) > batch_size:
                    logger.debug(
                        f"Processed {total_updated}/{len(embeddings_data)} embeddings"
                    )

            # Create vector index if we have enough embeddings
            total_rows = self._chunks_table.count_rows()
            if total_rows >= 256:  # LanceDB minimum for index creation
                try:
                    # Check if we need to create an index
                    # LanceDB will handle this efficiently if index already exists
                    self._executor_create_vector_index(
                        conn, state, provider, model, embedding_dims
                    )
                except Exception as e:
                    # This is expected if the table was created with variable-size list schema
                    # The index will work once the table is recreated with fixed-size schema
                    logger.debug(
                        f"Vector index creation deferred (expected with initial schema): {e}"
                    )

            logger.debug(
                f"Successfully updated {total_updated} embeddings using merge_insert"
            )

            return total_updated

        except Exception as e:
            logger.error(f"Error in bulk embedding insert: {e}")
            raise

    def get_embedding_by_chunk_id(
        self, chunk_id: int, provider: str, model: str
    ) -> Embedding | None:
        """Get embedding for specific chunk, provider, and model."""
        chunk = self.get_chunk_by_id(chunk_id)
        if not chunk or not chunk.get("embedding"):
            return None

        created_time = chunk.get("created_time", time.time())
        created_at = datetime.fromtimestamp(created_time) if created_time else None

        return Embedding(
            chunk_id=chunk_id,
            provider=chunk.get("provider", provider),
            model=chunk.get("model", model),
            dims=len(chunk["embedding"]),
            vector=chunk["embedding"],
            created_at=created_at,
        )

    def get_existing_embeddings(
        self, chunk_ids: list[int], provider: str, model: str
    ) -> set[int]:
        """Get set of chunk IDs that already have embeddings for given provider/model."""
        return self._execute_in_db_thread_sync(
            "get_existing_embeddings", chunk_ids, provider, model
        )

    def _executor_get_existing_embeddings(
        self,
        conn: Any,
        state: dict[str, Any],
        chunk_ids: list[int],
        provider: str,
        model: str,
    ) -> set[int]:
        """Executor method for get_existing_embeddings - runs in DB thread.

        Optimized to avoid loading entire table into memory for large datasets.
        Uses native LanceDB queries instead of pandas filtering for better performance.
        """
        if not self._chunks_table:
            return set()

        try:
            existing_chunk_ids = set()

            if not chunk_ids:
                # No specific chunk_ids provided - scan all chunks for embeddings
                # Use native LanceDB queries instead of pandas filtering
                try:
                    # Use LanceDB's native search with WHERE clause for provider/model filtering
                    # This avoids loading data into pandas DataFrames entirely
                    results = self._chunks_table.search().where(
                        f"provider = '{provider}' AND model = '{model}' AND embedding IS NOT NULL"
                    ).to_list()

                    # Filter results to only include valid embeddings (zero-vector check)
                    for result in results:
                        if _has_valid_embedding(result.get("embedding")):
                            existing_chunk_ids.add(result["id"])

                except Exception as e:
                    logger.warning(f"Native LanceDB query failed, using fallback: {e}")
                    # Fallback to paginated pandas approach if native queries fail
                    total_rows = self._chunks_table.count_rows()
                    page_size = 5000

                    for offset in range(0, total_rows, page_size):
                        try:
                            page_df = self._chunks_table.to_pandas(offset=offset, limit=page_size)
                            if page_df.empty:
                                break

                            # Filter for valid embeddings with matching provider/model
                            embeddings_mask = page_df["embedding"].apply(_has_valid_embedding)
                            matching_df = page_df[
                                embeddings_mask
                                & (page_df["provider"] == provider)
                                & (page_df["model"] == model)
                            ]

                            existing_chunk_ids.update(matching_df["id"].tolist())

                        except TypeError:
                            # LanceDB may not support offset/limit in to_pandas()
                            logger.warning("LanceDB to_pandas() doesn't support pagination, loading full table")
                            try:
                                all_chunks_df = self._chunks_table.to_pandas()
                                embeddings_mask = all_chunks_df["embedding"].apply(_has_valid_embedding)
                                existing_embeddings_df = all_chunks_df[
                                    embeddings_mask
                                    & (all_chunks_df["provider"] == provider)
                                    & (all_chunks_df["model"] == model)
                                ]
                                existing_chunk_ids.update(existing_embeddings_df["id"].tolist())
                            except Exception as fallback_error:
                                logger.error(f"Fallback loading also failed: {fallback_error}")
                                return set()
                            break
            else:
                # Specific chunk_ids provided - use targeted query
                # For smaller lists, we can query directly
                if len(chunk_ids) <= 1000:
                    chunk_ids_str = ','.join(map(str, chunk_ids))
                    try:
                        # Use LanceDB's native search with combined WHERE clause
                        # This filters at the database level instead of loading and filtering in pandas
                        results = self._chunks_table.search().where(
                            f"id IN ({chunk_ids_str}) AND provider = '{provider}' AND model = '{model}' AND embedding IS NOT NULL"
                        ).to_list()

                        # Filter results to only include valid embeddings (zero-vector check)
                        for result in results:
                            if _has_valid_embedding(result.get("embedding")):
                                existing_chunk_ids.add(result["id"])

                    except Exception:
                        # Fallback to individual queries if IN clause fails
                        for chunk_id in chunk_ids:
                            try:
                                result = self._chunks_table.search().where(
                                    f"id = {chunk_id} AND provider = '{provider}' AND model = '{model}' AND embedding IS NOT NULL"
                                ).to_list()
                                if result and _has_valid_embedding(result[0].get("embedding")):
                                    existing_chunk_ids.add(chunk_id)
                            except Exception:
                                continue
                else:
                    # For larger lists, use batched native queries instead of pandas pagination
                    batch_size = 500
                    for i in range(0, len(chunk_ids), batch_size):
                        batch_chunk_ids = chunk_ids[i:i + batch_size]
                        chunk_ids_str = ','.join(map(str, batch_chunk_ids))

                        try:
                            # Use native LanceDB query for each batch
                            batch_results = self._chunks_table.search().where(
                                f"id IN ({chunk_ids_str}) AND provider = '{provider}' AND model = '{model}' AND embedding IS NOT NULL"
                            ).to_list()

                            # Filter results to only include valid embeddings
                            for result in batch_results:
                                if _has_valid_embedding(result.get("embedding")):
                                    existing_chunk_ids.add(result["id"])

                        except Exception:
                            # Fallback to individual queries for this batch
                            for chunk_id in batch_chunk_ids:
                                try:
                                    result = self._chunks_table.search().where(
                                        f"id = {chunk_id} AND provider = '{provider}' AND model = '{model}' AND embedding IS NOT NULL"
                                    ).to_list()
                                    if result and _has_valid_embedding(result[0].get("embedding")):
                                        existing_chunk_ids.add(chunk_id)
                                except Exception:
                                    continue

            return existing_chunk_ids

        except Exception as e:
            logger.error(f"Error getting existing embeddings: {e}")
            return set()

    def delete_embeddings_by_chunk_id(self, chunk_id: int) -> None:
        """Delete all embeddings for a specific chunk."""
        # In LanceDB, this would involve updating the chunk to remove embedding data
        pass

    def get_all_chunks_with_metadata(self) -> list[dict[str, Any]]:
        """Get all chunks with their metadata including file paths (provider-agnostic)."""
        return self._execute_in_db_thread_sync("get_all_chunks_with_metadata")

    def _executor_get_all_chunks_with_metadata(
        self, conn: Any, state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Executor method for get_all_chunks_with_metadata - runs in DB thread.

        Optimized to avoid loading entire tables into memory for large datasets.
        Uses native LanceDB queries instead of pandas DataFrames for better performance.
        """
        if not self._chunks_table or not self._files_table:
            return []

        try:
            # For large datasets, use streaming approach to avoid memory issues
            total_chunks = self._chunks_table.count_rows()
            page_size = 5000  # Process in smaller chunks to prevent memory issues
            result = []

            # Build file_id to path mapping incrementally to avoid loading all files at once
            file_paths = {}
            processed_file_ids = set()

            for offset in range(0, total_chunks, page_size):
                try:
                    # Use native LanceDB queries instead of pandas loading
                    # Get chunks using search (no offset/limit support, but we can limit results)
                    chunks_query = self._chunks_table.search().limit(page_size)
                    if offset > 0:
                        # For pagination, we'll need to skip results manually
                        # This is less efficient but avoids pandas DataFrame loading
                        all_chunks_page = self._chunks_table.search().limit(offset + page_size).to_list()
                        chunks_page = all_chunks_page[offset:offset + page_size]
                    else:
                        chunks_page = chunks_query.to_list()

                    if not chunks_page:
                        break

                    # Get unique file_ids from this page that we haven't processed yet
                    page_file_ids = set(chunk["file_id"] for chunk in chunks_page)
                    new_file_ids = page_file_ids - processed_file_ids

                    # Load file paths for new file_ids in batches using native queries
                    if new_file_ids:
                        file_batch_size = 5000
                        new_file_ids_list = list(new_file_ids)
                        for file_offset in range(0, len(new_file_ids_list), file_batch_size):
                            file_batch = new_file_ids_list[file_offset:file_offset + file_batch_size]
                            file_ids_str = ','.join(map(str, file_batch))

                            try:
                                # Use native LanceDB filter for file lookup
                                file_results = (
                                    self._files_table.to_lance()
                                    .to_table(filter=f"id IN ({file_ids_str})")
                                    .to_pandas()
                                )
                            except Exception:
                                # Fallback to search-based filtering
                                file_results = self._files_table.search().where(
                                    f"id IN ({file_ids_str})"
                                ).to_pandas()

                            # Update file paths mapping
                            for _, file_row in file_results.iterrows():
                                file_paths[file_row["id"]] = file_row["path"]

                        processed_file_ids.update(new_file_ids)

                    # Build results for this page using native data structures
                    for chunk in chunks_page:
                        result.append(
                            {
                                "id": chunk["id"],
                                "file_id": chunk["file_id"],
                                "file_path": file_paths.get(
                                    chunk["file_id"], ""
                                ),  # Keep stored format
                                "content": chunk["content"],
                                "start_line": chunk["start_line"],
                                "end_line": chunk["end_line"],
                                "chunk_type": chunk["chunk_type"],
                                "language": chunk["language"],
                                "name": chunk["name"],
                            }
                        )

                except Exception as page_error:
                    # If native queries fail, fall back to pandas approach
                    logger.warning(f"Native query approach failed: {page_error}, using pandas fallback")
                    try:
                        # Load a page of chunks using pandas
                        chunks_page = self._chunks_table.to_pandas(offset=offset, limit=page_size)
                        if chunks_page.empty:
                            break

                        # Get unique file_ids from this page that we haven't processed yet
                        page_file_ids = set(chunks_page["file_id"].unique())
                        new_file_ids = page_file_ids - processed_file_ids

                        # Load file paths for new file_ids in batches
                        if new_file_ids:
                            file_batch_size = 5000
                            new_file_ids_list = list(new_file_ids)
                            for file_offset in range(0, len(new_file_ids_list), file_batch_size):
                                file_batch = new_file_ids_list[file_offset:file_offset + file_batch_size]
                                file_ids_str = ','.join(map(str, file_batch))

                                try:
                                    # Try LanceDB's native filter first
                                    file_results = (
                                        self._files_table.to_lance()
                                        .to_table(filter=f"id IN ({file_ids_str})")
                                        .to_pandas()
                                    )
                                except Exception:
                                    # Fallback to search-based filtering
                                    file_results = self._files_table.search().where(
                                        f"id IN ({file_ids_str})"
                                    ).to_pandas()

                                # Update file paths mapping
                                for _, file_row in file_results.iterrows():
                                    file_paths[file_row["id"]] = file_row["path"]

                            processed_file_ids.update(new_file_ids)

                        # Build results for this page
                        for _, chunk in chunks_page.iterrows():
                            result.append(
                                {
                                    "id": chunk["id"],
                                    "file_id": chunk["file_id"],
                                    "file_path": file_paths.get(
                                        chunk["file_id"], ""
                                    ),  # Keep stored format
                                    "content": chunk["content"],
                                    "start_line": chunk["start_line"],
                                    "end_line": chunk["end_line"],
                                    "chunk_type": chunk["chunk_type"],
                                    "language": chunk["language"],
                                    "name": chunk["name"],
                                }
                            )

                    except Exception as fallback_error:
                        logger.error(f"Pandas fallback also failed: {fallback_error}")
                        return []
                    break

            return result

        except Exception as e:
            logger.error(f"Error getting chunks with metadata: {e}")
            return []

    # Search Operations (delegate to base class which uses executor)
    def _executor_search_semantic(
        self,
        conn: Any,
        state: dict[str, Any],
        query_embedding: list[float],
        provider: str,
        model: str,
        page_size: int = 10,
        offset: int = 0,
        threshold: float | None = None,
        path_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Executor method for search_semantic - runs in DB thread."""
        if self._chunks_table is None:
            raise RuntimeError("Chunks table not initialized")


        # Validate embeddings exist for this provider/model
        try:
            chunks_count = self._chunks_table.count_rows()
            if chunks_count == 0:
                return [], {
                    "offset": offset,
                    "page_size": 0,
                    "has_more": False,
                    "total": 0,
                }

            # Check if any chunks have embeddings for this provider/model
            try:
                sample_chunks = self._chunks_table.head(
                    min(100, chunks_count)
                ).to_pandas()
                # Handle embeddings that are lists - also exclude zero vectors
                embeddings_mask = sample_chunks["embedding"].apply(_has_valid_embedding)
            except Exception as data_error:
                logger.error(
                    f"LanceDB data corruption detected during semantic search: {data_error}"
                )
                return [], {
                    "offset": offset,
                    "page_size": 0,
                    "has_more": False,
                    "total": 0,
                }
            embeddings_exist = (
                embeddings_mask
                & (sample_chunks["provider"] == provider)
                & (sample_chunks["model"] == model)
            ).any()

            if not embeddings_exist:
                logger.warning(
                    f"No embeddings found for provider={provider}, model={model}"
                )
                return [], {
                    "offset": offset,
                    "page_size": 0,
                    "has_more": False,
                    "total": 0,
                }

            # Perform vector search with explicit vector column name
            query = self._chunks_table.search(
                query_embedding, vector_column_name="embedding"
            )
            query = query.where(
                f"provider = '{provider}' AND model = '{model}' AND embedding IS NOT NULL"
            )
            query = query.limit(page_size + offset)

            if threshold:
                query = query.where(f"_distance <= {threshold}")

            if path_filter:
                # Join with files table to filter by path
                pass  # Would need more complex query joining with files table

            results = query.to_list()

            # Deduplicate across fragments (safety net for fragment-induced duplicates)
            results = _deduplicate_by_id(results)

            # Apply offset manually since LanceDB doesn't have native offset
            paginated_results = results[offset : offset + page_size]

            # Format results to match DuckDB output and exclude raw embeddings
            formatted_results = []
            for result in paginated_results:
                # Get file path from files table
                file_path = ""
                if self._files_table and "file_id" in result:
                    try:
                        file_results = (
                            self._files_table.search()
                            .where(f"id = {result['file_id']}")
                            .to_list()
                        )
                        if file_results:
                            file_path = file_results[0].get("path", "")
                    except Exception:
                        pass

                # Convert _distance to similarity (1 - distance for cosine)
                similarity = (
                    1.0 - result.get("_distance", 0.0) if "_distance" in result else 1.0
                )

                # Format the result to match DuckDB's output
                formatted_result = {
                    "chunk_id": result["id"],
                    "symbol": result.get("name", ""),
                    "content": result.get("content", ""),
                    "chunk_type": result.get("chunk_type", ""),
                    "start_line": result.get("start_line", 0),
                    "end_line": result.get("end_line", 0),
                    "file_path": file_path,  # Keep stored format
                    "language": result.get("language", ""),
                    "similarity": similarity,
                }
                formatted_results.append(formatted_result)

            pagination = {
                "offset": offset,
                "page_size": len(paginated_results),
                "has_more": len(results) > offset + page_size,
                "total": len(results),
            }

            return formatted_results, pagination

        except Exception as e:
            logger.error(
                f"Error in semantic search with provider={provider}, model={model}: {e}"
            )
            # Re-raise the error instead of silently returning empty results
            raise RuntimeError(f"Semantic search failed: {e}") from e

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
            threshold: Optional similarity threshold (0-1, where 1 is most similar)
            path_filter: Optional relative path to limit search scope

        Returns:
            List of similar chunks with scores and metadata
        """
        return self._execute_in_db_thread_sync(
            "find_similar_chunks",
            chunk_id,
            provider,
            model,
            limit,
            threshold,
            path_filter,
        )

    def _executor_find_similar_chunks(
        self,
        conn: Any,
        state: dict[str, Any],
        chunk_id: int,
        provider: str,
        model: str,
        limit: int,
        threshold: float | None,
        path_filter: str | None,
    ) -> list[dict[str, Any]]:
        """Executor method for find_similar_chunks - runs in DB thread.

        LanceDB-specific implementation:
        - Embeddings stored inline in chunks table (no separate embeddings table)
        - Uses native .search() API with vector_column_name parameter
        - Path filtering deferred (not yet implemented in LanceDB)
        """
        if self._chunks_table is None:
            raise RuntimeError("Chunks table not initialized")

        try:
            # PHASE 1: Retrieve target chunk's embedding
            # In LanceDB, embeddings are stored directly in chunks table
            target_results = (
                self._chunks_table.search()
                .where(
                    f"id = {chunk_id} AND provider = '{provider}' "
                    f"AND model = '{model}'"
                )
                .limit(1)
                .to_list()
            )

            if not target_results:
                logger.warning(
                    f"No embedding found for chunk_id={chunk_id}, "
                    f"provider='{provider}', model='{model}'"
                )
                return []

            target_chunk = target_results[0]
            target_embedding = target_chunk.get("embedding")

            # Validate embedding exists and is valid
            if not _has_valid_embedding(target_embedding):
                logger.warning(
                    f"Chunk {chunk_id} has no valid embedding for "
                    f"provider={provider}, model={model}"
                )
                return []

            # PHASE 2: Vector search for similar chunks
            query = self._chunks_table.search(
                target_embedding, vector_column_name="embedding"
            )
            query = query.where(
                f"provider = '{provider}' AND model = '{model}' "
                f"AND embedding IS NOT NULL AND id != {chunk_id}"
            )

            # Note: Cannot filter by _distance in WHERE clause - it only exists in results
            # We'll filter after getting results if threshold is specified
            # Request more results than needed if using threshold
            fetch_limit = limit * 3 if threshold is not None else limit
            query = query.limit(fetch_limit)

            # TODO(#107): Path filtering not yet implemented in LanceDB
            # See https://github.com/chunkhound/chunkhound/issues/107
            if path_filter:
                logger.warning(
                    "Path filtering not yet implemented for LanceDB "
                    "find_similar_chunks"
                )

            results = query.to_list()

            # PHASE 3: Format results with file paths and apply threshold
            formatted_results = []
            for result in results:
                # Convert distance to similarity score (cosine: similarity = 1 - distance)
                distance = result.get("_distance", 0.0)
                similarity = 1.0 - distance

                # Apply threshold filter if specified
                if threshold is not None and similarity < threshold:
                    continue

                # Get file path from files table (reuse pattern from search_semantic)
                file_path = ""
                if self._files_table and "file_id" in result:
                    try:
                        file_results = (
                            self._files_table.search()
                            .where(f"id = {result['file_id']}")
                            .limit(1)
                            .to_list()
                        )
                        if file_results:
                            file_path = file_results[0].get("path", "")
                    except Exception as e:
                        logger.debug(f"Failed to fetch file path: {e}")

                formatted_results.append({
                    "chunk_id": result["id"],
                    "name": result.get("name", ""),
                    "content": result.get("content", ""),
                    "chunk_type": result.get("chunk_type", ""),
                    "start_line": result.get("start_line", 0),
                    "end_line": result.get("end_line", 0),
                    "file_path": file_path,
                    "language": result.get("language", ""),
                    "score": similarity,  # Match DuckDB convention
                })

                # Stop once we have enough results that meet the threshold
                if len(formatted_results) >= limit:
                    break

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to find similar chunks: {e}")
            return []

    def _executor_search_regex(
        self,
        conn: Any,
        state: dict[str, Any],
        pattern: str,
        page_size: int,
        offset: int,
        path_filter: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Executor method for search_regex - runs in DB thread.

        Optimized for large datasets: uses native LanceDB LIKE/regex queries instead of pandas filtering.
        """
        if not self._chunks_table or not self._files_table:
            return [], {"offset": offset, "page_size": 0, "has_more": False, "total": 0}

        try:
            # Optimize query based on pattern complexity
            escaped_pattern = pattern.replace("'", "''")

            # Use LIKE for simple patterns (much faster than regex)
            if self._is_simple_pattern(pattern):
                search_type = "like"
            else:
                search_type = "regex"

            logger.debug(f"Using {search_type} search for pattern: {pattern}")

            # Build file filter if path filtering is needed
            file_filter_ids = None
            if path_filter:
                escaped_path = path_filter.replace("'", "''")
                try:
                    file_results = self._files_table.search().where(
                        f"path LIKE '{escaped_path}%'"
                    ).to_list()
                    file_filter_ids = {r["id"] for r in file_results}
                    logger.debug(f"Path filter matched {len(file_filter_ids)} files")
                except Exception as path_error:
                    logger.warning(f"Path filtering failed: {path_error}")

            # Use native LanceDB queries instead of pandas filtering
            try:
                # Build the WHERE clause for content search
                if search_type == "like":
                    content_condition = f"content LIKE '%{escaped_pattern}%'"
                else:
                    content_condition = f"regexp_match(content, '{escaped_pattern}')"

                # Add file filter if needed
                if file_filter_ids:
                    file_ids_str = ','.join(map(str, file_filter_ids))
                    full_condition = f"({content_condition}) AND file_id IN ({file_ids_str})"
                else:
                    full_condition = content_condition

                # Execute the search - fetch all results for accurate total count
                # Note: This may use more memory for broad searches, but ensures correct pagination
                results = self._chunks_table.search().where(full_condition).to_list()

                # Deduplicate results (fragments may cause duplicates)
                results = _deduplicate_by_id(results)

                # Safeguard against excessive memory usage for broad searches
                MAX_REGEX_RESULTS = 10000
                if len(results) > MAX_REGEX_RESULTS:
                    logger.warning(
                        f"Regex search returned {len(results)} results, truncating to {MAX_REGEX_RESULTS} "
                        f"for memory safety. Total count is approximate."
                    )
                    results = results[:MAX_REGEX_RESULTS]

                # Apply offset manually
                paginated_results = results[offset : offset + page_size]

                # Convert to expected format
                formatted_results = []
                for result in paginated_results:
                    formatted_result = {
                        "id": result["id"],
                        "file_id": result["file_id"],
                        "name": result.get("name", ""),
                        "content": result.get("content", ""),
                        "chunk_type": result.get("chunk_type", ""),
                        "start_line": result.get("start_line", 0),
                        "end_line": result.get("end_line", 0),
                        "language": result.get("language", ""),
                    }
                    formatted_results.append(formatted_result)

                # Get file paths for results (batch operation for efficiency)
                formatted = self._add_file_paths_to_results(formatted_results)

                pagination = {
                    "offset": offset,
                    "page_size": len(formatted),
                    "has_more": len(results) > offset + page_size,
                    "total": len(results),
                }

                return formatted, pagination

            except Exception as native_error:
                # Fallback to pandas-based approach if native queries fail
                logger.warning(f"Native LanceDB search failed: {native_error}, using pandas fallback")

                # For large datasets, collect results with early termination
                total_rows = self._chunks_table.count_rows()
                scan_page_size = min(10000, max(page_size * 2, 1000))
                collected_results = []

                # Scan chunks in pages to avoid loading everything
                for scan_offset in range(0, total_rows, scan_page_size):
                    try:
                        # Load a batch of chunks
                        batch_df = self._chunks_table.to_pandas(offset=scan_offset, limit=scan_page_size)
                        if batch_df.empty:
                            break

                        # Apply content search filter
                        if search_type == "like":
                            content_mask = batch_df["content"].str.contains(escaped_pattern, regex=False, na=False)
                        else:
                            content_mask = batch_df["content"].str.contains(escaped_pattern, regex=True, na=False)

                        # Apply path filter if specified
                        if file_filter_ids is not None:
                            path_mask = batch_df["file_id"].isin(file_filter_ids)
                            combined_mask = content_mask & path_mask
                        else:
                            combined_mask = content_mask

                        matching_chunks = batch_df[combined_mask]

                        # Convert to result format
                        for _, chunk in matching_chunks.iterrows():
                            result = {
                                "id": chunk["id"],
                                "file_id": chunk["file_id"],
                                "name": chunk.get("name", ""),
                                "content": chunk.get("content", ""),
                                "chunk_type": chunk.get("chunk_type", ""),
                                "start_line": chunk.get("start_line", 0),
                                "end_line": chunk.get("end_line", 0),
                                "language": chunk.get("language", ""),
                            }
                            collected_results.append(result)

                        # Early termination if we have enough results for pagination
                        if len(collected_results) >= offset + page_size * 2:
                            break

                    except TypeError:
                        # LanceDB doesn't support offset/limit in to_pandas()
                        logger.warning("Using memory-intensive fallback for regex search")
                        try:
                            all_chunks_df = self._chunks_table.to_pandas()

                            # Apply content search
                            if search_type == "like":
                                content_mask = all_chunks_df["content"].str.contains(escaped_pattern, regex=False, na=False)
                            else:
                                content_mask = all_chunks_df["content"].str.contains(escaped_pattern, regex=True, na=False)

                            # Apply path filter
                            if file_filter_ids is not None:
                                path_mask = all_chunks_df["file_id"].isin(file_filter_ids)
                                combined_mask = content_mask & path_mask
                            else:
                                combined_mask = content_mask

                            matching_df = all_chunks_df[combined_mask]

                            # Convert to results
                            collected_results = []
                            for _, chunk in matching_df.iterrows():
                                result = {
                                    "id": chunk["id"],
                                    "file_id": chunk["file_id"],
                                    "name": chunk.get("name", ""),
                                    "content": chunk.get("content", ""),
                                    "chunk_type": chunk.get("chunk_type", ""),
                                    "start_line": chunk.get("start_line", 0),
                                    "end_line": chunk.get("end_line", 0),
                                    "language": chunk.get("language", ""),
                                }
                                collected_results.append(result)

                        except Exception as fallback_error:
                            logger.error(f"Fallback regex search failed: {fallback_error}")
                            raise RuntimeError(f"Regex search failed: {fallback_error}") from fallback_error
                        break

                # Deduplicate results (fragments may cause duplicates)
                collected_results = _deduplicate_by_id(collected_results)
                total_found = len(collected_results)

                # Apply pagination
                paginated = collected_results[offset : offset + page_size]

                # Get file paths for paginated results (batch operation for efficiency)
                formatted = self._add_file_paths_to_results(paginated)

                pagination = {
                    "offset": offset,
                    "page_size": len(paginated),
                    "has_more": total_found > offset + page_size,
                    "total": total_found,
                }

                return formatted, pagination

        except Exception as e:
            logger.error(f"Error in regex search: {e}")
            raise RuntimeError(f"Regex search failed: {e}") from e

    def _is_simple_pattern(self, pattern: str) -> bool:
        """Check if pattern is simple enough to use LIKE instead of regex.

        Simple patterns are those without regex special characters.
        LIKE is much faster than regex for simple substring searches.
        """
        # Regex special characters that make a pattern complex
        regex_chars = ['^', '$', '.', '*', '+', '?', '|', '(', ')', '[', ']', '{', '}', '\\']

        # If pattern contains regex special chars, it's complex
        return not any(char in pattern for char in regex_chars)

    def _add_file_paths_to_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add file paths to search results efficiently."""
        if not results:
            return results

        # Collect unique file_ids
        file_ids = list(set(r["file_id"] for r in results))
        file_paths = {}

        # Batch load file paths
        batch_size = 1000
        for i in range(0, len(file_ids), batch_size):
            batch_file_ids = file_ids[i : i + batch_size]
            ids_str = ','.join(map(str, batch_file_ids))

            try:
                # Try LanceDB native filter
                file_batch = (
                    self._files_table.to_lance()
                    .to_table(filter=f"id IN ({ids_str})")
                    .to_pandas()
                )
            except Exception:
                # Fallback to search
                file_batch = self._files_table.search().where(f"id IN ({ids_str})").to_pandas()

            # Build path mapping
            for _, file_row in file_batch.iterrows():
                file_paths[file_row["id"]] = file_row["path"]

        # Add paths to results
        for result in results:
            result["file_path"] = file_paths.get(result["file_id"], "")
            # Reformat to match expected output
            result["chunk_id"] = result.pop("id")
            result["symbol"] = result.pop("name")

        return results

    def search_fuzzy(
        self,
        query: str,
        page_size: int = 10,
        offset: int = 0,
        path_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform fuzzy text search using LanceDB's text capabilities."""
        return self._execute_in_db_thread_sync(
            "search_fuzzy", query, page_size, offset, path_filter
        )

    def _executor_search_fuzzy(
        self,
        conn: Any,
        state: dict[str, Any],
        query: str,
        page_size: int = 10,
        offset: int = 0,
        path_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Executor method for search_fuzzy - runs in DB thread."""
        if not self._chunks_table:
            return [], {"offset": offset, "page_size": 0, "has_more": False, "total": 0}

        try:
            # Use LanceDB's full-text search capabilities
            results = (
                self._chunks_table.search()
                .where(f"content LIKE '%{query}%'")
                .limit(page_size + offset)
                .to_list()
            )

            # Apply offset manually
            paginated_results = results[offset : offset + page_size]

            # Format results to match DuckDB output and exclude raw embeddings
            formatted_results = []
            for result in paginated_results:
                # Get file path from files table
                file_path = ""
                if self._files_table and "file_id" in result:
                    try:
                        file_results = (
                            self._files_table.search()
                            .where(f"id = {result['file_id']}")
                            .to_list()
                        )
                        if file_results:
                            file_path = file_results[0].get("path", "")
                    except Exception:
                        pass

                # Format the result to match DuckDB's output (no similarity for fuzzy search)
                formatted_result = {
                    "chunk_id": result["id"],
                    "symbol": result.get("name", ""),
                    "content": result.get("content", ""),
                    "chunk_type": result.get("chunk_type", ""),
                    "start_line": result.get("start_line", 0),
                    "end_line": result.get("end_line", 0),
                    "file_path": file_path,  # Keep stored format
                    "language": result.get("language", ""),
                }
                formatted_results.append(formatted_result)

            pagination = {
                "offset": offset,
                "page_size": len(paginated_results),
                "has_more": len(results) > offset + page_size,
                "total": len(results),
            }

            return formatted_results, pagination

        except Exception as e:
            logger.error(f"Error in fuzzy search: {e}")
            return [], {"offset": offset, "page_size": 0, "has_more": False, "total": 0}

    def _executor_search_text(
        self,
        conn: Any,
        state: dict[str, Any],
        query: str,
        page_size: int = 10,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Executor method for search_text - runs in DB thread."""
        return self._executor_search_fuzzy(conn, state, query, page_size, offset, None)

    # Statistics and Monitoring
    def get_stats(self) -> dict[str, int]:
        """Get database statistics (file count, chunk count, etc.)."""
        return self._execute_in_db_thread_sync("get_stats")

    def _executor_get_stats(self, conn: Any, state: dict[str, Any]) -> dict[str, int]:
        """Executor method for get_stats - runs in DB thread.

        Optimized to avoid loading entire tables into memory for large datasets.
        Uses native LanceDB queries instead of pandas DataFrames for better performance.
        """
        stats = {"files": 0, "chunks": 0, "embeddings": 0, "size_mb": 0}

        try:
            if self._files_table:
                try:
                    # Use native LanceDB count_rows() instead of loading entire table into pandas
                    stats["files"] = self._files_table.count_rows()
                except Exception as data_error:
                    logger.warning(
                        f"Failed to get files stats due to data corruption: {data_error}"
                    )
                    stats["files"] = 0

            if self._chunks_table:
                try:
                    # Use native LanceDB count_rows() for total chunks
                    stats["chunks"] = self._chunks_table.count_rows()

                    # Count valid embeddings using native WHERE query instead of pandas filtering
                    # This avoids loading the entire chunks table into memory
                    try:
                        # Use LanceDB's native search with WHERE clause to count valid embeddings
                        # We can't directly count with WHERE in LanceDB, so we get the results and count
                        embedding_results = self._chunks_table.search().where(
                            "embedding IS NOT NULL"
                        ).to_list()

                        # Filter for valid embeddings (non-empty, non-zero vectors)
                        valid_embeddings = [
                            result for result in embedding_results
                            if _has_valid_embedding(result.get("embedding"))
                        ]
                        stats["embeddings"] = len(valid_embeddings)

                    except Exception as embedding_error:
                        logger.warning(f"Failed to count embeddings with native query: {embedding_error}")
                        # Fallback to pandas approach if native queries fail
                        try:
                            chunks_df = self._chunks_table.to_pandas()
                            embeddings_mask = chunks_df["embedding"].apply(_has_valid_embedding)
                            stats["embeddings"] = len(chunks_df[embeddings_mask])
                        except Exception as fallback_error:
                            logger.warning(f"Pandas fallback also failed: {fallback_error}")
                            stats["embeddings"] = 0

                except Exception as data_error:
                    logger.warning(
                        f"Failed to get chunks stats due to data corruption: {data_error}"
                    )
                    # Try to get count using count_rows() which is more robust
                    try:
                        stats["chunks"] = self._chunks_table.count_rows()
                    except Exception:
                        stats["chunks"] = 0
                    stats["embeddings"] = 0

            # Calculate size (approximate)
            if self._db_path.exists():
                total_size = sum(
                    f.stat().st_size for f in self._db_path.rglob("*") if f.is_file()
                )
                stats["size_mb"] = total_size / (1024 * 1024)

        except Exception as e:
            logger.error(f"Error getting stats: {e}")

        return stats

    def get_file_stats(self, file_id: int) -> dict[str, Any]:
        """Get statistics for a specific file."""
        return self._execute_in_db_thread_sync("get_file_stats", file_id)

    def _executor_get_file_stats(
        self, conn: Any, state: dict[str, Any], file_id: int
    ) -> dict[str, Any]:
        """Executor method for get_file_stats - runs in DB thread."""
        chunks = self._executor_get_chunks_by_file_id(conn, state, file_id, False)
        return {
            "file_id": file_id,
            "chunk_count": len(chunks),
            "embedding_count": sum(
                1
                for chunk in chunks
                if chunk.get("embedding") is not None
                and isinstance(chunk.get("embedding"), (list, np.ndarray))
                and len(chunk.get("embedding", [])) > 0
            ),
        }

    def get_provider_stats(self, provider: str, model: str) -> dict[str, Any]:
        """Get statistics for a specific embedding provider/model."""
        return self._execute_in_db_thread_sync("get_provider_stats", provider, model)

    def _executor_get_provider_stats(
        self, conn: Any, state: dict[str, Any], provider: str, model: str
    ) -> dict[str, Any]:
        """Executor method for get_provider_stats - runs in DB thread."""
        if not self._chunks_table:
            return {"provider": provider, "model": model, "embedding_count": 0}

        try:
            results = (
                self._chunks_table.search()
                .where(
                    f"provider = '{provider}' AND model = '{model}' AND embedding IS NOT NULL"
                )
                .to_list()
            )

            return {
                "provider": provider,
                "model": model,
                "embedding_count": len(results),
            }
        except Exception as e:
            logger.error(f"Error getting provider stats: {e}")
            return {"provider": provider, "model": model, "embedding_count": 0}

    # Transaction and Bulk Operations
    def execute_query(
        self, query: str, params: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a limited subset of read queries for coordinator helpers.

        LanceDB has no SQL interface; this adapter recognizes a small set of
        patterns used by higher layers (e.g., change detection in the indexing
        coordinator) and serves equivalent results via the native API.

        Optimized to avoid loading entire tables into memory for large datasets.
        Uses native LanceDB queries instead of pandas DataFrames for better performance.

        Supported forms:
        - SELECT path, size, modified_time, content_hash FROM files
        - SELECT path, size, modified_time FROM files
        """
        try:
            if not self._files_table:
                return []

            q = (query or "").strip().lower().replace("\n", " ")
            if q.startswith("select") and " from files" in q:
                # Determine requested columns
                cols: list[str] = []
                try:
                    select_part = q.split("from", 1)[0]
                    select_part = select_part.replace("select", "").strip()
                    cols = [c.strip() for c in select_part.split(",") if c.strip()]
                except Exception:
                    cols = ["path", "size", "modified_time", "content_hash"]

                # Fetch all rows via native LanceDB API instead of pandas
                # This avoids loading the entire table into memory as a DataFrame
                try:
                    # Use LanceDB's native to_list() method for direct dictionary results
                    # This is much more memory-efficient than to_pandas() + iterrows()
                    all_records = self._files_table.to_list()

                    # Filter to requested columns without pandas DataFrame overhead
                    rows: list[dict[str, Any]] = []
                    for rec in all_records:
                        out: dict[str, Any] = {}
                        for c in cols:
                            if c in rec:
                                out[c] = rec[c]
                            else:
                                # Provide None for missing optional columns
                                out[c] = None
                        rows.append(out)
                    return rows
                except Exception:
                    return []

            # Unsupported pattern → no-op (coordinator will fall back)
            return []
        except Exception:
            return []

    # File Processing Integration (inherited from base class)
    async def process_file_incremental(self, file_path: Path) -> dict[str, Any]:
        """Process a file with incremental parsing and differential chunking."""
        if not self._services_initialized:
            self._initialize_shared_instances()

        # Call process_file with embeddings enabled for real-time indexing
        # This ensures embeddings are generated immediately for modified files
        return await self._indexing_coordinator.process_file(
            file_path, skip_embeddings=False
        )

    # Health and Diagnostics
    def get_fragment_count(self) -> dict[str, int]:
        """Get current fragment counts for chunks and files tables.

        Returns:
            Dictionary with fragment counts: {"chunks": 551, "files": 12}
        """
        return self._execute_in_db_thread_sync("get_fragment_count")

    def _executor_get_fragment_count(
        self, conn: Any, state: dict[str, Any]
    ) -> dict[str, int]:
        """Executor method for get_fragment_count - runs in DB thread."""
        result = {}

        if self._chunks_table:
            try:
                stats = self._chunks_table.stats()
                # stats is a dict, access fragment info directly
                if isinstance(stats, dict) and "fragment_stats" in stats:
                    fragment_stats = stats["fragment_stats"]
                    if hasattr(fragment_stats, "num_fragments"):
                        result["chunks"] = fragment_stats.num_fragments
                    elif isinstance(fragment_stats, dict) and "num_fragments" in fragment_stats:
                        result["chunks"] = fragment_stats["num_fragments"]
                    else:
                        result["chunks"] = 0
                else:
                    result["chunks"] = 0
            except Exception as e:
                logger.debug(f"Could not get chunks fragment count: {e}")
                result["chunks"] = 0

        if self._files_table:
            try:
                stats = self._files_table.stats()
                # stats is a dict, access fragment info directly
                if isinstance(stats, dict) and "fragment_stats" in stats:
                    fragment_stats = stats["fragment_stats"]
                    if hasattr(fragment_stats, "num_fragments"):
                        result["files"] = fragment_stats.num_fragments
                    elif isinstance(fragment_stats, dict) and "num_fragments" in fragment_stats:
                        result["files"] = fragment_stats["num_fragments"]
                    else:
                        result["files"] = 0
                else:
                    result["files"] = 0
            except Exception as e:
                logger.debug(f"Could not get files fragment count: {e}")
                result["files"] = 0

        return result

    def should_optimize(self, operation: str = "") -> bool:
        """Check if optimization is warranted based on fragment count vs threshold.

        Args:
            operation: Optional operation name for logging (e.g., "post-chunking")

        Returns:
            True if fragment count exceeds threshold, False otherwise
        """
        try:
            counts = self.get_fragment_count()
            chunks_fragments = counts.get("chunks", 0)
            if chunks_fragments < self._fragment_threshold:
                op_desc = f" {operation}" if operation else ""
                logger.debug(
                    f"Skipping{op_desc} optimization: {chunks_fragments} fragments "
                    f"< threshold {self._fragment_threshold}"
                )
                return False
            return True
        except Exception as e:
            logger.debug(f"Could not check fragment count, will optimize: {e}")
            return True

    def should_optimize_during_indexing(self) -> bool:
        """Check if optimization should run during indexing to prevent fragmentation.

        LanceDB optimization during indexing is now always enabled for optimal performance.

        Returns:
            True - optimization always runs during indexing for LanceDB
        """
        try:
            counts = self.get_fragment_count()
            chunks_fragments = counts.get("chunks", 0)
            should_optimize = chunks_fragments >= 25  # Default threshold for optimization
            logger.debug(f"Fragment check: chunks={chunks_fragments}, threshold=25, should_optimize={should_optimize}")
            return should_optimize
        except Exception as e:
            logger.debug(f"Could not check fragment count for indexing optimization: {e}")
            return False

    def optimize_tables(self) -> None:
        """Optimize tables by compacting fragments and rebuilding indexes."""
        # Use higher timeout for optimization operations (5 minutes vs default 30s)
        import os
        original_timeout = os.environ.get("CHUNKHOUND_DB_EXECUTE_TIMEOUT")
        try:
            os.environ["CHUNKHOUND_DB_EXECUTE_TIMEOUT"] = "300"  # 5 minutes
            return self._execute_in_db_thread_sync("optimize_tables")
        finally:
            # Restore original timeout
            if original_timeout is not None:
                os.environ["CHUNKHOUND_DB_EXECUTE_TIMEOUT"] = original_timeout
            elif "CHUNKHOUND_DB_EXECUTE_TIMEOUT" in os.environ:
                del os.environ["CHUNKHOUND_DB_EXECUTE_TIMEOUT"]

    def _executor_optimize_tables(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for optimize_tables - runs in DB thread."""
        from datetime import timedelta

        try:
            if self._chunks_table:
                logger.debug("Optimizing chunks table - compacting fragments...")
                # Use minimal cleanup window (1 minute) to focus on fragment consolidation
                # rather than time-based cleanup. The goal is compaction, not age-based deletion.
                stats = self._chunks_table.optimize(
                    cleanup_older_than=timedelta(minutes=1), delete_unverified=True
                )
                if stats is not None:
                    logger.debug(
                        f"Chunks table cleanup freed {stats.bytes_removed / 1024 / 1024:.2f} MB"
                    )
                logger.debug("Chunks table optimization complete")

            if self._files_table:
                logger.debug("Optimizing files table - compacting fragments...")
                stats = self._files_table.optimize(
                    cleanup_older_than=timedelta(minutes=1), delete_unverified=True
                )
                if stats is not None:
                    logger.debug(
                        f"Files table cleanup freed {stats.bytes_removed / 1024 / 1024:.2f} MB"
                    )
                logger.debug("Files table optimization complete")

        except Exception as e:
            logger.warning(f"Failed to optimize tables: {e}")

    def health_check(self) -> dict[str, Any]:
        """Perform health check and return status information."""
        return self._execute_in_db_thread_sync("health_check")

    def _executor_get_chunk_ids_without_embeddings_paginated(
        self,
        conn: Any,
        state: dict[str, Any],
        provider: str,
        model: str,
        exclude_patterns: list[str] | None,
        limit: int,
        offset: int,
    ) -> list[int]:
        """Executor method for get_chunk_ids_without_embeddings_paginated - runs in DB thread.

        Optimized for large datasets: uses native LanceDB queries instead of pandas filtering.
        This prevents memory issues and improves performance for large chunk datasets.
        """
        if not self._chunks_table:
            return []

        try:
            if not exclude_patterns:
                # Simple case: no path filtering needed
                # Use native LanceDB queries to find chunks needing embeddings
                collected_chunk_ids = []

                logger.debug(f"Finding chunks needing embeddings for provider='{provider}', model='{model}'")

                # Build WHERE condition for chunks that need embeddings
                # A chunk needs embeddings if:
                # 1. embedding IS NULL, OR
                # 2. embedding is empty, OR
                # 3. embedding is invalid (all zeros), OR
                # 4. provider/model mismatch

                # For LanceDB, we can efficiently query for NULL embeddings and provider/model mismatches
                # For empty/invalid embeddings, we need to fetch and check

                # First, get chunks with NULL embeddings or wrong provider/model
                try:
                    null_embedding_results = self._chunks_table.search().where(
                        f"(embedding IS NULL OR provider != '{provider}' OR model != '{model}')"
                    ).to_list()

                    # Extract IDs and filter for actual needs
                    for result in null_embedding_results:
                        embedding = result.get("embedding")
                        current_provider = result.get("provider")
                        current_model = result.get("model")

                        # Check if this chunk actually needs embedding
                        needs_embedding = False
                        if embedding is None:
                            needs_embedding = True
                        elif not _has_valid_embedding(embedding):
                            needs_embedding = True
                        elif current_provider != provider or current_model != model:
                            needs_embedding = True

                        if needs_embedding:
                            collected_chunk_ids.append(result["id"])

                except Exception as native_error:
                    logger.warning(f"Native LanceDB query failed: {native_error}, using pandas fallback")
                    # Fallback to pandas approach if native queries fail
                    try:
                        all_chunks_df = self._chunks_table.to_pandas()
                        logger.debug(f"Fallback: loaded all {len(all_chunks_df)} chunks")

                        def _fallback_needs_embedding_check(row):
                            embedding = row["embedding"]
                            if embedding is None or (hasattr(embedding, '__len__') and len(embedding) == 0):
                                return True
                            if not _has_valid_embedding(embedding):
                                return True
                            if row["provider"] != provider or row["model"] != model:
                                return True
                            return False

                        needs_embedding_mask = all_chunks_df.apply(_fallback_needs_embedding_check, axis=1)
                        needs_embedding_df = all_chunks_df[needs_embedding_mask]
                        collected_chunk_ids = needs_embedding_df["id"].tolist()
                        logger.debug(f"Fallback: found {len(collected_chunk_ids)} chunks needing embeddings")
                    except Exception as fallback_error:
                        logger.error(f"Fallback loading failed: {fallback_error}")
                        return []

                # Apply pagination to collected results
                paginated_ids = collected_chunk_ids[offset : offset + limit]
                logger.debug(f"Returning {len(paginated_ids)} chunk IDs (offset={offset}, limit={limit}, total found={len(collected_chunk_ids)})")
                return paginated_ids

            else:
                # Complex case: path filtering required
                # This requires joining with files table, which LanceDB doesn't support efficiently
                # We'll collect candidate chunks first, then filter by path
                candidate_ids = self._executor_get_chunk_ids_without_embeddings_paginated(
                    conn, state, provider, model, None, limit * 5, offset  # Get more candidates
                )

                if not candidate_ids:
                    return []

                # Filter by exclude patterns using native queries
                filtered_ids = []
                batch_size = 1000

                for i in range(0, len(candidate_ids), batch_size):
                    batch_ids = candidate_ids[i : i + batch_size]
                    ids_str = ','.join(map(str, batch_ids))

                    try:
                        # Get file paths for this batch using native LanceDB filter
                        file_results = (
                            self._files_table.to_lance()
                            .to_table(filter=f"id IN ({ids_str})")
                            .to_pandas()
                        )
                    except Exception:
                        # Fallback to search
                        file_results = self._files_table.search().where(
                            f"id IN ({ids_str})"
                        ).to_pandas()

                    # Filter out excluded paths
                    for _, file_row in file_results.iterrows():
                        file_path = file_row["path"]
                        should_exclude = False

                        if exclude_patterns and file_path:
                            for pattern in exclude_patterns:
                                if pattern in file_path:
                                    should_exclude = True
                                    break

                        if not should_exclude:
                            # Find corresponding chunk_id from candidate_ids
                            chunk_id = file_row["id"]
                            if chunk_id in batch_ids:
                                filtered_ids.append(chunk_id)

                    # Stop if we have enough results
                    if len(filtered_ids) >= limit:
                        break

                return filtered_ids[:limit]

        except Exception as e:
            logger.error(f"Failed to get chunk IDs without embeddings: {e}")
            return []

    def _executor_health_check(
        self, conn: Any, state: dict[str, Any]
    ) -> dict[str, Any]:
        """Executor method for health_check - runs in DB thread."""
        health_status = {
            "status": "healthy" if self.is_connected else "disconnected",
            "provider": "lancedb",
            "database_path": str(self._db_path),
            "tables": {
                "files": self._files_table is not None,
                "chunks": self._chunks_table is not None,
            },
        }

        # Check for data corruption
        if self.is_connected and self._chunks_table:
            try:
                # Try to read a small sample to detect corruption
                self._chunks_table.head(10).to_pandas()
                health_status["data_integrity"] = "ok"
            except Exception as e:
                health_status["status"] = "corrupted"
                health_status["data_integrity"] = f"corruption detected: {e}"
                health_status["recovery_suggestion"] = (
                    "Run optimize_tables() or recreate database"
                )

        return health_status

    def get_chunk_ids_without_embeddings_paginated(
        self,
        provider: str,
        model: str,
        exclude_patterns: list[str] | None = None,
        limit: int = 10000,
        offset: int = 0,
    ) -> list[int]:
        """Get chunk IDs that don't have embeddings for the specified provider/model with pagination."""
        return self._execute_in_db_thread_sync(
            "get_chunk_ids_without_embeddings_paginated",
            provider,
            model,
            exclude_patterns,
            limit,
            offset,
        )

    def get_connection_info(self) -> dict[str, Any]:
        """Get information about the database connection."""
        return {
            "provider": "lancedb",
            "database_path": str(self._db_path),
            "connected": self.is_connected,
            "index_type": self.index_type,
            "performance_stats": self._query_performance,
        }

    def _record_query_performance(self, operation: str, duration: float, record_count: int = 0) -> None:
        """Record performance metrics for query operations."""
        if operation not in self._query_performance:
            self._query_performance[operation] = {
                "calls": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "max_time": 0.0,
                "total_records": 0,
            }

        stats = self._query_performance[operation]
        stats["calls"] += 1
        stats["total_time"] += duration
        stats["max_time"] = max(stats["max_time"], duration)
        stats["avg_time"] = stats["total_time"] / stats["calls"]
        stats["total_records"] += record_count

        # Log slow queries
        if duration > 5.0:  # Log queries taking more than 5 seconds
            logger.warning(
                f"Slow {operation}: {duration:.2f}s, {record_count} records"
            )
