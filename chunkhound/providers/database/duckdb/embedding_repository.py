"""DuckDB Embedding Repository - handles embedding CRUD operations."""

import time
from typing import Any

from loguru import logger

from chunkhound.core.models import Embedding
from chunkhound.providers.database.duckdb.connection_manager import (
    DuckDBConnectionManager,
)


class DuckDBEmbeddingRepository:
    """Repository for embedding operations in DuckDB."""

    def __init__(self, connection_manager: DuckDBConnectionManager, provider=None):
        """Initialize the embedding repository.

        Args:
            connection_manager: DuckDB connection manager for database access
            provider: Optional provider instance for transaction-aware connections
        """
        self.connection_manager = connection_manager
        self._provider = provider

    # NOTE: connection property was removed - connections are managed by the executor
    # All operations must go through the provider's executor methods

    def insert_embedding(self, embedding: Embedding) -> int:
        """Insert embedding record and return embedding ID."""
        if not self._provider:
            raise RuntimeError("Provider required for database operations")

        try:
            return self._provider._execute_in_db_thread_sync(
                "insert_embedding", embedding
            )
        except Exception as e:
            logger.error(f"Failed to insert embedding: {e}")
            raise

    def insert_embeddings_batch(
        self,
        embeddings_data: list[dict],
        batch_size: int | None = None,
        connection=None,
    ) -> int:
        """Insert multiple embedding vectors using optimized bulk insert.

        Uses ShardManager for shard assignment and USearch indexing.
        HNSW index management has been removed - indexing is now handled
        by ShardManager with USearch.

        Args:
            embeddings_data: List of dicts with keys: chunk_id, provider,
                            model, embedding, dims
            batch_size: Not used (kept for backward compatibility)
            connection: Database connection to use (required, passed from executor)

        Returns:
            Number of successfully inserted embeddings
        """
        # Connection must be provided from executor context
        if connection is None:
            if not self._provider:
                raise RuntimeError("Provider required for database operations")
            # Delegate to provider's executor
            return self._provider._execute_in_db_thread_sync(
                "insert_embeddings_batch_raw", embeddings_data
            )

        conn = connection
        if not embeddings_data:
            return 0

        actual_batch_size = len(embeddings_data)
        logger.debug(f"Starting batch insert of {actual_batch_size} embeddings")

        # Auto-detect embedding dimensions from first embedding
        first_vector = embeddings_data[0]["embedding"]
        detected_dims = len(first_vector)

        # Validate all embeddings have the same dimensions
        for i, embedding_data in enumerate(embeddings_data):
            vector = embedding_data["embedding"]
            if len(vector) != detected_dims:
                raise ValueError(
                    f"Embedding vector {i} has {len(vector)} dimensions, "
                    f"expected {detected_dims} (detected from first embedding)"
                )

        # Ensure appropriate table exists for these dimensions
        if self._provider:
            table_name = self._provider._ensure_embedding_table_exists(detected_dims)
        else:
            table_name = f"embeddings_{detected_dims}"

        logger.debug(
            f"Using table {table_name} for {detected_dims}-dimensional embeddings"
        )

        # Extract provider/model for conflict checking
        first_embedding = embeddings_data[0]
        provider = first_embedding["provider"]
        model = first_embedding["model"]

        try:
            total_inserted = 0
            start_time = time.time()

            # Separate new vs existing embeddings for optimal INSERT strategy
            chunk_ids = [emb["chunk_id"] for emb in embeddings_data]
            existing_chunk_ids = self.get_existing_embeddings_with_conn(
                conn, chunk_ids, provider, model
            )

            # Separate new vs existing embeddings
            new_embeddings = [
                emb
                for emb in embeddings_data
                if emb["chunk_id"] not in existing_chunk_ids
            ]
            update_embeddings = [
                emb
                for emb in embeddings_data
                if emb["chunk_id"] in existing_chunk_ids
            ]

            # Fast INSERT for new embeddings using VALUES table construction
            if new_embeddings:
                insert_start = time.time()

                try:
                    # Set DuckDB performance options for bulk loading
                    conn.execute("SET preserve_insertion_order = false")

                    # Build VALUES clause for bulk insert (much faster than executemany)
                    values_parts = []
                    for embedding_data in new_embeddings:
                        vector_str = str(embedding_data["embedding"])
                        values_parts.append(
                            f"({embedding_data['chunk_id']}, '{embedding_data['provider']}', '{embedding_data['model']}', {vector_str}, {embedding_data['dims']})"
                        )

                    # Single INSERT with all values (fastest approach without external deps)
                    values_clause = ",\n    ".join(values_parts)
                    conn.execute(f"""
                        INSERT INTO {table_name} (chunk_id, provider, model, embedding, dims)
                        VALUES {values_clause}
                    """)

                    insert_time = time.time() - insert_start
                    logger.debug(
                        f"Fast VALUES INSERT completed in {insert_time:.3f}s "
                        f"({len(new_embeddings) / max(insert_time, 0.001):.1f} emb/s)"
                    )
                    total_inserted += len(new_embeddings)

                except Exception as e:
                    logger.error(f"Fast VALUES INSERT failed: {e}")
                    raise

            # INSERT OR REPLACE for updates using VALUES approach
            if update_embeddings:
                update_start = time.time()

                try:
                    # Build VALUES clause for bulk updates
                    values_parts = []
                    for embedding_data in update_embeddings:
                        vector_str = str(embedding_data["embedding"])
                        values_parts.append(
                            f"({embedding_data['chunk_id']}, '{embedding_data['provider']}', '{embedding_data['model']}', {vector_str}, {embedding_data['dims']})"
                        )

                    # Single INSERT OR REPLACE with all values
                    values_clause = ",\n    ".join(values_parts)
                    conn.execute(f"""
                        INSERT OR REPLACE INTO {table_name} (chunk_id, provider, model, embedding, dims)
                        VALUES {values_clause}
                    """)

                    update_time = time.time() - update_start
                    logger.debug(
                        f"VALUES UPDATE completed in {update_time:.3f}s "
                        f"({len(update_embeddings) / max(update_time, 0.001):.1f} emb/s)"
                    )
                    total_inserted += len(update_embeddings)

                except Exception as e:
                    logger.error(f"VALUES UPDATE failed: {e}")
                    raise

            insert_time = time.time() - start_time
            logger.debug(
                f"Batch INSERT completed: {total_inserted} embeddings in {insert_time:.3f}s "
                f"({total_inserted / max(insert_time, 0.001):.1f} embeddings/sec)"
            )

            # Wire embeddings to ShardManager for HNSW index updates
            if self._provider and hasattr(self._provider, 'shard_manager'):
                if self._provider.shard_manager is not None:
                    # Only update HNSW for newly inserted embeddings
                    if new_embeddings:
                        # Get IDs of just-inserted embeddings
                        chunk_ids_placeholders = ','.join('?' * len(new_embeddings))
                        chunk_ids_params = [e['chunk_id'] for e in new_embeddings]

                        emb_ids_result = conn.execute(f"""
                            SELECT id, embedding FROM {table_name}
                            WHERE chunk_id IN ({chunk_ids_placeholders})
                              AND provider = ? AND model = ?
                            ORDER BY id
                        """, chunk_ids_params + [provider, model]).fetchall()

                        # Build embedding dicts for shard manager
                        emb_dicts = [
                            {"id": row[0], "embedding": row[1]}
                            for row in emb_ids_result
                        ]

                        # Route to shards and update HNSW indexes
                        success, needs_fix = self._provider.shard_manager.insert_embeddings(
                            emb_dicts, detected_dims, provider, model, conn
                        )

                        logger.debug(f"HNSW indexes updated for {len(emb_dicts)} embeddings")
                elif str(self._provider._connection_manager.db_path) != ":memory:":
                    # Non-memory database without ShardManager is a critical error
                    logger.error(
                        f"ShardManager not initialized - semantic search unavailable "
                        f"for {total_inserted} embeddings"
                    )

            return total_inserted

        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            raise

    def get_embedding_by_chunk_id(
        self, chunk_id: int, provider: str, model: str
    ) -> Embedding | None:
        """Get embedding for specific chunk, provider, and model."""
        if not self._provider:
            raise RuntimeError("Provider required for database operations")

        try:
            return self._provider._execute_in_db_thread_sync(
                "get_embedding_by_chunk_id", chunk_id, provider, model
            )
        except Exception as e:
            logger.error(f"Failed to get embedding for chunk {chunk_id}: {e}")
            return None

    def get_existing_embeddings(
        self, chunk_ids: list[int], provider: str, model: str
    ) -> set[int]:
        """Get set of chunk IDs that already have embeddings for given provider/model."""
        if not self._provider:
            raise RuntimeError("Provider required for database operations")

        if not chunk_ids:
            return set()

        try:
            return self._provider._execute_in_db_thread_sync(
                "get_existing_embeddings", chunk_ids, provider, model
            )
        except Exception as e:
            logger.error(f"Failed to get existing embeddings: {e}")
            return set()

    def get_existing_embeddings_with_conn(
        self, conn: Any, chunk_ids: list[int], provider: str, model: str
    ) -> set[int]:
        """Get set of chunk IDs that already have embeddings (uses provided connection).

        This method is used internally when a connection is already available
        from the executor context.
        """
        if not chunk_ids:
            return set()

        try:
            all_chunk_ids = set()

            # Get all embedding tables
            table_result = conn.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_name LIKE 'embeddings_%'
            """).fetchall()

            for (table_name,) in table_result:
                # Create placeholders for IN clause
                placeholders = ",".join("?" * len(chunk_ids))
                params = chunk_ids + [provider, model]

                results = conn.execute(
                    f"""
                    SELECT DISTINCT chunk_id
                    FROM {table_name}
                    WHERE chunk_id IN ({placeholders}) AND provider = ? AND model = ?
                """,
                    params,
                ).fetchall()

                all_chunk_ids.update(result[0] for result in results)
            return all_chunk_ids

        except Exception as e:
            logger.error(f"Failed to get existing embeddings: {e}")
            return set()

    def delete_embeddings_by_chunk_id(self, chunk_id: int) -> None:
        """Delete all embeddings for a specific chunk.

        Note: ShardManager coordination is handled by DuckDBProvider's executor.
        This method performs only the raw DELETE operations.
        """
        if not self._provider:
            raise RuntimeError("Provider required for database operations")

        try:
            self._provider._execute_in_db_thread_sync(
                "delete_embeddings_by_chunk_id", chunk_id
            )
        except Exception as e:
            logger.error(f"Failed to delete embeddings for chunk {chunk_id}: {e}")
            raise
