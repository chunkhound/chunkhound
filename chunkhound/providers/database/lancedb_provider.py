"""LanceDB provider implementation for ChunkHound - concrete database provider using LanceDB."""

import os
import time
from datetime import datetime, timedelta
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


def _validate_integer_ids(ids: list[Any], operation: str) -> list[int]:
    """Validate and convert IDs to integers for safe SQL construction.

    Defense-in-depth against type confusion when building SQL IN clauses.
    All chunk/file IDs in ChunkHound are 64-bit integers.

    Args:
        ids: List of values that should be integers
        operation: Name of calling operation for error messages

    Returns:
        List of validated integer IDs

    Raises:
        ValueError: If any ID cannot be converted to integer
    """
    if not ids:
        return []

    validated = []
    for i, id_val in enumerate(ids):
        try:
            int_id = int(id_val)
            validated.append(int_id)
        except (TypeError, ValueError) as e:
            type_name = type(id_val).__name__
            raise ValueError(
                f"{operation}: Invalid ID at index {i}: {id_val!r} ({type_name})"
            ) from e

    return validated


class LanceDBProvider(SerialDatabaseProvider):
    """LanceDB implementation using serial executor pattern."""

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

    def _create_connection(self) -> Any:
        """Create and return a LanceDB connection.

        This method is called from within the executor thread to create
        a thread-local connection.

        Returns:
            LanceDB connection object
        """
        import lancedb

        abs_db_path = self._db_path

        # Save CWD (thread-safe in executor)
        original_cwd = os.getcwd()
        try:
            os.chdir(abs_db_path.parent)
            # Enable 3-second eventual consistency for read operations.
            # This ensures cached table handles refresh to see recent deletes within 3s,
            # aligning with ChunkHound's event processing architecture:
            # - Debounce delay: 500ms
            # - Event dedup window: 2s
            # - Typical processing: 500ms-1.5s
            # - Total pipeline: ~3s from file change to search availability
            # Without this, cached handles read stale table versions indefinitely.
            conn = lancedb.connect(
                abs_db_path.name,
                read_consistency_interval=timedelta(seconds=3)
            )
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
        return self._execute_in_db_thread_sync("get_file_by_path", path, as_model)

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

    def update_file(
        self,
        file_id: int,
        size_bytes: int | None = None,
        mtime: float | None = None,
        content_hash: str | None = None,
        **kwargs,
    ) -> None:
        """Update file record with new values."""
        return self._execute_in_db_thread_sync(
            "update_file", file_id, size_bytes, mtime, content_hash
        )

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

        # Process in optimal batch sizes (LanceDB best practice: 1000+ items)
        batch_size = 1000
        all_chunk_ids = []

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
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

            # Use PyArrow Table directly to avoid LanceDB DataFrame schema alignment bug
            chunks_table = pa.Table.from_pylist(
                chunk_data_list, schema=get_chunks_schema()
            )

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
            all_chunk_ids.extend(chunk_ids)

            logger.debug(f"Bulk inserted batch of {len(batch_chunks)} chunks")

        logger.debug(f"Completed bulk insert of {len(chunks)} chunks in batches")
        return all_chunk_ids

    def get_chunk_by_id(
        self, chunk_id: int, as_model: bool = False
    ) -> dict[str, Any] | Chunk | None:
        """Get chunk record by ID."""
        return self._execute_in_db_thread_sync("get_chunk_by_id", chunk_id, as_model)

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

    def delete_chunks_batch(self, chunk_ids: list[int]) -> None:
        """Delete multiple chunks by ID efficiently."""
        self._execute_in_db_thread_sync("delete_chunks_batch", chunk_ids)

    def _executor_delete_chunks_batch(
        self, conn: Any, state: dict[str, Any], chunk_ids: list[int]
    ) -> None:
        """Executor method for delete_chunks_batch - runs in DB thread."""
        if not chunk_ids or not self._chunks_table:
            return
        try:
            # Validate IDs for defense-in-depth before SQL construction
            validated_ids = _validate_integer_ids(chunk_ids, "delete_chunks_batch")
            ids_str = ",".join(str(id) for id in validated_ids)
            self._chunks_table.delete(f"id IN ({ids_str})")

            # Trigger compaction after batch delete if fragment threshold exceeded
            # Uses existing configuration (lancedb_optimize_fragment_threshold)
            if self.should_optimize("post-delete"):
                self._executor_optimize_tables(conn, state)
        except Exception as e:
            logger.error(f"Error deleting chunks batch: {e}")

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
                # LanceDB recommends 200k-500k batch sizes for optimal performance
                # Use 100k as a balance between throughput (10x improvement) and memory
                batch_size = min(100000, len(embeddings_data))

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
                # Validate IDs for defense-in-depth before SQL construction
                validated_chunk_ids = _validate_integer_ids(
                    chunk_ids, "bulk_embedding_insert"
                )
                chunk_ids_str = ",".join(map(str, validated_chunk_ids))

                try:
                    # Primary: Use LanceDB's native Lance filter (efficient for large tables)
                    existing_df = (
                        self._chunks_table.to_lance()
                        .to_table(filter=f"id IN ({chunk_ids_str})")
                        .to_pandas()
                    )
                except Exception as lance_err:
                    # Fallback: Paginated pandas filtering (memory-safe for large tables)
                    total_rows = self._chunks_table.count_rows()

                    logger.warning(
                        f"Lance SQL filter unavailable, using paginated fallback for {total_rows:,} rows. "
                        f"Error: {lance_err}"
                    )

                    # Paginate to avoid loading entire table into memory
                    page_size = 10_000
                    existing_rows = []
                    chunk_ids_set = set(chunk_ids)  # For faster lookup

                    for offset in range(0, total_rows, page_size):
                        # Load batch of rows
                        try:
                            batch_df = self._chunks_table.to_pandas(offset=offset, limit=page_size)
                        except TypeError:
                            # LanceDB may not support offset/limit in to_pandas()
                            # Fall back to loading all and slicing (less efficient but works)
                            if offset == 0:
                                logger.debug("LanceDB to_pandas() doesn't support pagination, loading full table")
                                all_chunks_df = self._chunks_table.to_pandas()
                                batch_df = all_chunks_df[all_chunks_df['id'].isin(chunk_ids)]
                                existing_rows.extend(batch_df.to_dict('records'))
                                break
                            else:
                                break

                        # Filter to requested chunk IDs
                        matching = batch_df[batch_df['id'].isin(chunk_ids_set)]
                        if len(matching) > 0:
                            existing_rows.extend(matching.to_dict('records'))

                        # Early termination if we found all requested chunks
                        if len(existing_rows) >= len(chunk_ids):
                            break

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
                # Use to_dict('records') instead of iterrows() for 10-100x speedup
                ids_to_update = existing_df["id"].isin(embedding_lookup.keys())
                rows_to_merge = existing_df[ids_to_update].to_dict("records")

                merge_data = []
                for row in rows_to_merge:
                    emb_data = embedding_lookup[row["id"]]
                    row["embedding"] = emb_data["embedding"]
                    row["provider"] = emb_data["provider"]
                    row["model"] = emb_data["model"]
                    merge_data.append(row)

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
        """Executor method for get_existing_embeddings - runs in DB thread."""
        if not self._chunks_table:
            return set()

        try:
            # In LanceDB, we store embeddings directly in the chunks table
            # A chunk has embeddings if the embedding field is not null AND
            # the provider/model match what we're looking for
            chunks_count = self._chunks_table.count_rows()
            try:
                all_chunks_df = self._chunks_table.head(chunks_count).to_pandas()
            except Exception as data_error:
                logger.error(
                    f"LanceDB data corruption detected in chunks table: {data_error}"
                )
                logger.info("Attempting table recovery by recreating indexes...")
                # Try to recover by optimizing the table
                try:
                    self._chunks_table.optimize()
                    all_chunks_df = self._chunks_table.head(chunks_count).to_pandas()
                except Exception as recovery_error:
                    logger.error(f"Failed to recover chunks table: {recovery_error}")
                    return set()

            # Handle embeddings that are lists - pandas notna() might not work correctly with lists
            # Also check embedding is not all zeros (defense-in-depth for legacy placeholder vectors)
            embeddings_mask = all_chunks_df["embedding"].apply(_has_valid_embedding)

            # If no specific chunk_ids provided, check all chunks
            if not chunk_ids:
                # Find all chunks that have embeddings for this provider/model
                existing_embeddings_df = all_chunks_df[
                    embeddings_mask
                    & (all_chunks_df["provider"] == provider)
                    & (all_chunks_df["model"] == model)
                ]
            else:
                # Filter to only the requested chunk IDs
                filtered_df = all_chunks_df[all_chunks_df["id"].isin(chunk_ids)]
                filtered_embeddings_mask = filtered_df.index.isin(
                    all_chunks_df[embeddings_mask].index
                )

                # Find chunks that have embeddings for this provider/model
                existing_embeddings_df = filtered_df[
                    filtered_embeddings_mask
                    & (filtered_df["provider"] == provider)
                    & (filtered_df["model"] == model)
                ]

            return set(existing_embeddings_df["id"].tolist())
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
        """Executor method for get_all_chunks_with_metadata - runs in DB thread."""
        if not self._chunks_table or not self._files_table:
            return []

        try:
            # Get all chunks using LanceDB native API (workaround for to_pandas() bug)
            chunks_count = self._chunks_table.count_rows()
            try:
                chunks_df = self._chunks_table.head(chunks_count).to_pandas()
            except Exception as data_error:
                logger.error(
                    f"LanceDB data corruption detected in chunks table: {data_error}"
                )
                logger.info("Attempting table recovery by recreating indexes...")
                try:
                    self._chunks_table.optimize()
                    chunks_df = self._chunks_table.head(chunks_count).to_pandas()
                except Exception as recovery_error:
                    logger.error(f"Failed to recover chunks table: {recovery_error}")
                    return []

            # Get all files for path lookup
            files_count = self._files_table.count_rows()
            try:
                files_df = self._files_table.head(files_count).to_pandas()
            except Exception as data_error:
                logger.error(
                    f"LanceDB data corruption detected in files table: {data_error}"
                )
                try:
                    self._files_table.optimize()
                    files_df = self._files_table.head(files_count).to_pandas()
                except Exception as recovery_error:
                    logger.error(f"Failed to recover files table: {recovery_error}")
                    return []

            # Create file_id to path mapping
            file_paths = dict(zip(files_df["id"], files_df["path"]))

            # Build result with file paths
            result = []
            for _, chunk in chunks_df.iterrows():
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
        """Executor method for search_regex - runs in DB thread."""
        if not self._chunks_table or not self._files_table:
            return [], {"offset": offset, "page_size": 0, "has_more": False, "total": 0}

        try:
            # Build WHERE clause using regexp_match (DataFusion SQL function)
            # Escape single quotes in pattern to prevent SQL injection
            escaped_pattern = pattern.replace("'", "''")
            where_clause = f"regexp_match(content, '{escaped_pattern}')"

            # Get all matching chunks
            # Note: .search().where() without vector may return duplicates across fragments
            results = self._chunks_table.search().where(where_clause).to_list()

            # Deduplicate across fragments (critical fix for fragmentation bug)
            results = _deduplicate_by_id(results)

            # Apply path filter if provided
            if path_filter:
                # Get file IDs matching path filter
                escaped_path = path_filter.replace("'", "''")
                file_results = self._files_table.search().where(
                    f"path LIKE '{escaped_path}%'"
                ).to_list()
                valid_file_ids = {r["id"] for r in file_results}
                results = [r for r in results if r["file_id"] in valid_file_ids]

            total_count = len(results)

            # Apply pagination
            paginated = results[offset : offset + page_size]

            # Format results with file paths
            formatted = []
            for result in paginated:
                file_path = ""
                if self._files_table:
                    file_results = self._files_table.search().where(
                        f"id = {result['file_id']}"
                    ).to_list()
                    if file_results:
                        file_path = file_results[0].get("path", "")

                formatted.append({
                    "chunk_id": result["id"],
                    "symbol": result.get("name", ""),
                    "content": result.get("content", ""),
                    "chunk_type": result.get("chunk_type", ""),
                    "start_line": result.get("start_line", 0),
                    "end_line": result.get("end_line", 0),
                    "file_path": file_path,
                    "language": result.get("language", ""),
                })

            pagination = {
                "offset": offset,
                "page_size": len(paginated),
                "has_more": total_count > offset + page_size,
                "total": total_count,
            }

            return formatted, pagination

        except Exception as e:
            logger.error(f"Error in regex search: {e}")
            raise RuntimeError(f"Regex search failed: {e}") from e

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
        """Executor method for get_stats - runs in DB thread."""
        stats = {"files": 0, "chunks": 0, "embeddings": 0, "size_mb": 0}

        try:
            if self._files_table:
                try:
                    stats["files"] = len(self._files_table.to_pandas())
                except Exception as data_error:
                    logger.warning(
                        f"Failed to get files stats due to data corruption: {data_error}"
                    )
                    stats["files"] = 0

            if self._chunks_table:
                try:
                    chunks_df = self._chunks_table.to_pandas()
                    stats["chunks"] = len(chunks_df)
                    # Handle embeddings that are lists - also exclude zero vectors
                    embeddings_mask = chunks_df["embedding"].apply(_has_valid_embedding)
                    stats["embeddings"] = len(chunks_df[embeddings_mask])
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

                # Fetch all rows via native API
                try:
                    total = int(self._files_table.count_rows())
                except Exception:
                    total = 0
                rows: list[dict[str, Any]] = []
                try:
                    if total > 0:
                        df = self._files_table.head(total).to_pandas()
                    else:
                        # Fallback for engines that don't support count_rows
                        df = self._files_table.to_pandas()
                    # Normalize frame into list of dicts with requested columns
                    for _, rec in df.iterrows():
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
                result["chunks"] = stats.fragment_stats.num_fragments
            except Exception as e:
                logger.debug(f"Could not get chunks fragment count: {e}")
                result["chunks"] = 0

        if self._files_table:
            try:
                stats = self._files_table.stats()
                result["files"] = stats.fragment_stats.num_fragments
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

    def optimize_tables(self) -> None:
        """Optimize tables by compacting fragments and rebuilding indexes."""
        return self._execute_in_db_thread_sync("optimize_tables")

    def _executor_optimize_tables(self, conn: Any, state: dict[str, Any]) -> None:
        """Executor method for optimize_tables - runs in DB thread.

        Uses cross-process file lock to prevent concurrent optimization operations.
        LanceDB's MVCC handles concurrent reads/writes safely, but aggressive cleanup
        during optimize() can delete fragment files still referenced by readers in
        other processes. The non-blocking lock ensures only one process optimizes
        at a time while others skip silently.
        """
        from datetime import timedelta

        from filelock import FileLock, Timeout

        # Create lock file in database directory (each database has its own lock)
        lock_path = self._db_path / ".optimize.lock"
        lock = FileLock(lock_path, timeout=0)  # Non-blocking: skip if already locked

        try:
            with lock:
                # Only one process can optimize at a time
                if self._chunks_table:
                    logger.debug("Optimizing chunks table - compacting fragments...")
                    # Keep versions for 5 minutes to allow readers to finish.
                    # Never delete unverified files - they may be referenced by readers.
                    stats = self._chunks_table.optimize(
                        cleanup_older_than=timedelta(minutes=5),
                        delete_unverified=False,
                    )
                    if stats is not None:
                        logger.debug(
                            f"Chunks table cleanup freed {stats.bytes_removed / 1024 / 1024:.2f} MB"
                        )
                    logger.debug("Chunks table optimization complete")

                if self._files_table:
                    logger.debug("Optimizing files table - compacting fragments...")
                    stats = self._files_table.optimize(
                        cleanup_older_than=timedelta(minutes=5),
                        delete_unverified=False,
                    )
                    if stats is not None:
                        logger.debug(
                            f"Files table cleanup freed {stats.bytes_removed / 1024 / 1024:.2f} MB"
                        )
                    logger.debug("Files table optimization complete")

        except Timeout:
            # Another process is optimizing - skip silently to avoid contention
            logger.debug("Skipping optimization - another process holds the lock")
        except Exception as e:
            logger.warning(f"Failed to optimize tables: {e}")

    def health_check(self) -> dict[str, Any]:
        """Perform health check and return status information."""
        return self._execute_in_db_thread_sync("health_check")

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

    def get_connection_info(self) -> dict[str, Any]:
        """Get information about the database connection."""
        return {
            "provider": "lancedb",
            "database_path": str(self._db_path),
            "connected": self.is_connected,
            "index_type": self.index_type,
        }
