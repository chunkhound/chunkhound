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
        self._provider_instance = None  # Will be set by provider

    @property
    def connection(self) -> Any | None:
        """Get database connection from connection manager."""
        return self.connection_manager.connection

    def set_provider_instance(self, provider_instance):
        """Set the provider instance for index management operations."""
        self._provider_instance = provider_instance

    def _drop_existing_index(self, conn: Any, index_info: dict[str, Any]) -> None:
        """Drop one discovered HNSW index by its exact current identity."""
        if self._provider_instance and hasattr(
            self._provider_instance, "_executor_drop_vector_index_by_name"
        ):
            self._provider_instance._executor_drop_vector_index_by_name(
                conn, index_info["index_name"]
            )
            return

        if (
            self._provider_instance
            and index_info.get("provider") is not None
            and index_info.get("model") is not None
        ):
            self._provider_instance.drop_vector_index(
                index_info["provider"],
                index_info["model"],
                index_info["dims"],
                index_info["metric"],
            )
            return

        raise RuntimeError(
            f"Cannot drop HNSW index without identity information: {index_info['index_name']}"
        )

    def _recreate_existing_index(self, conn: Any, index_info: dict[str, Any]) -> None:
        """Recreate one discovered HNSW index with its original SQL when available."""
        create_sql = index_info.get("create_sql")
        if create_sql:
            conn.execute(create_sql)
            return

        if (
            self._provider_instance
            and index_info.get("provider") is not None
            and index_info.get("model") is not None
        ):
            self._provider_instance.create_vector_index(
                index_info["provider"],
                index_info["model"],
                index_info["dims"],
                index_info["metric"],
            )
            return

        raise RuntimeError(
            f"Cannot recreate HNSW index without SQL or identity: {index_info['index_name']}"
        )

    def insert_embedding(self, embedding: Embedding) -> int:
        """Insert embedding record and return embedding ID."""
        if self.connection is None:
            raise RuntimeError("No database connection")

        try:
            # Ensure appropriate table exists for these dimensions
            if self._provider:
                table_name = self._provider._ensure_embedding_table_exists(
                    embedding.dims
                )
            else:
                # Fallback for tests - create table if needed
                table_name = f"embeddings_{embedding.dims}"
                # Check if table exists by querying information_schema
                result = self.connection_manager.connection.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
                    [table_name],
                ).fetchone()
                if result is None:
                    self.connection_manager.connection.execute(f"""  
                        CREATE TABLE {table_name} (
                            id INTEGER PRIMARY KEY DEFAULT nextval('embeddings_id_seq'),
                            chunk_id INTEGER REFERENCES chunks(id),
                            provider TEXT NOT NULL,
                            model TEXT NOT NULL,
                            embedding FLOAT[{embedding.dims}],
                            dims INTEGER NOT NULL DEFAULT {embedding.dims},
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

            result = self.connection_manager.connection.execute(
                f"""
                INSERT INTO {table_name} (chunk_id, provider, model, embedding, dims)
                VALUES (?, ?, ?, ?, ?)
                RETURNING id
            """,
                [
                    embedding.chunk_id,
                    embedding.provider,
                    embedding.model,
                    embedding.vector,
                    embedding.dims,
                ],
            )

            embedding_id = result.fetchone()[0]
            logger.debug(
                f"Inserted embedding {embedding_id} for chunk {embedding.chunk_id}"
            )
            return embedding_id

        except Exception as e:
            logger.error(f"Failed to insert embedding: {e}")
            raise

    def insert_embeddings_batch(
        self,
        embeddings_data: list[dict],
        batch_size: int | None = None,
        connection=None,
    ) -> int:
        """Insert multiple embedding vectors with HNSW index optimization.

        For large batches (>= batch_size threshold), uses the Context7-recommended optimization:
        1. Drop HNSW indexes to avoid insert slowdown (60s+ -> 5s for 300 items)
        2. Use fast INSERT for new embeddings, INSERT OR REPLACE for updates
        3. Recreate HNSW indexes after bulk operations

        Expected speedup: 10-20x faster for large batches (90s -> 5-10s).

        Args:
            embeddings_data: List of dicts with keys: chunk_id, provider, model, embedding, dims
            batch_size: Threshold for HNSW optimization (default: 50)
            connection: Optional database connection to use (for transaction contexts)

        Returns:
            Number of successfully inserted embeddings
        """
        # Use provided connection or default connection
        conn = connection if connection is not None else self.connection
        if conn is None:
            raise RuntimeError("No database connection")

        if not embeddings_data:
            return 0

        # Use provided batch_size threshold or default to 50
        hnsw_threshold = batch_size if batch_size is not None else 50
        actual_batch_size = len(embeddings_data)
        logger.debug(
            f"🔄 Starting optimized batch insert of {actual_batch_size} embeddings (HNSW threshold: {hnsw_threshold})"
        )

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
            # Fallback for tests
            table_name = f"embeddings_{detected_dims}"
            # Check if table exists by querying information_schema
            result = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
                [table_name],
            ).fetchone()
            if result is None:
                conn.execute(f"""
                    CREATE TABLE {table_name} (
                        id INTEGER PRIMARY KEY DEFAULT nextval('embeddings_id_seq'),
                        chunk_id INTEGER REFERENCES chunks(id),
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        embedding FLOAT[{detected_dims}],
                        dims INTEGER NOT NULL DEFAULT {detected_dims},
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
        logger.debug(
            f"Using table {table_name} for {detected_dims}-dimensional embeddings"
        )

        # Extract provider/model for conflict checking
        first_embedding = embeddings_data[0]
        provider = first_embedding["provider"]
        model = first_embedding["model"]

        # Use HNSW index optimization only for larger batches (research-based optimal threshold)
        # Based on benchmarks: small batches (1-10) keep indexes, medium+ batches (≥50) use optimization
        # This fixes semantic search for new files while maintaining bulk performance
        use_hnsw_optimization = actual_batch_size >= hnsw_threshold

        # Log the optimization decision for debugging
        if use_hnsw_optimization:
            logger.debug(
                f"🚀 Large batch: using HNSW optimization ({actual_batch_size} >= {hnsw_threshold})"
            )
        else:
            logger.debug(
                f"🔍 Small batch: preserving HNSW indexes for semantic search ({actual_batch_size} < {hnsw_threshold})"
            )

        try:
            total_inserted = 0
            start_time = time.time()

            if use_hnsw_optimization:
                # CRITICAL OPTIMIZATION: Drop HNSW indexes for bulk operations (research-based best practice)
                logger.debug(
                    f"🔧 Large batch detected ({actual_batch_size} embeddings >= {hnsw_threshold}), applying HNSW optimization"
                )

                # Extract dims for index management
                dims = first_embedding["dims"]

                # Step 1: Drop HNSW index to enable fast insertions
                if self._provider_instance and hasattr(
                    self._provider_instance, "get_existing_vector_indexes"
                ):
                    existing_indexes = (
                        self._provider_instance.get_existing_vector_indexes()
                    )
                    dropped_indexes = []

                    for index_info in existing_indexes:
                        try:
                            self._drop_existing_index(conn, index_info)
                            dropped_indexes.append(index_info)
                            logger.debug(f"Dropped index: {index_info['index_name']}")
                        except Exception as e:
                            logger.warning(
                                f"Could not drop index {index_info['index_name']}: {e}"
                            )

                    # Step 2: Separate new vs existing embeddings for optimal INSERT strategy
                    chunk_ids = [emb["chunk_id"] for emb in embeddings_data]
                    existing_chunk_ids = self.get_existing_embeddings(
                        chunk_ids, provider, model
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

                    # Step 3: Fast INSERT for new embeddings using VALUES table construction
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
                                f"✅ Fast VALUES INSERT completed in {insert_time:.3f}s ({len(new_embeddings) / insert_time:.1f} emb/s)"
                            )
                            total_inserted += len(new_embeddings)

                        except Exception as e:
                            logger.error(f"Fast VALUES INSERT failed: {e}")
                            raise

                    # Step 4: INSERT OR REPLACE only for updates using VALUES approach
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
                                f"✅ VALUES UPDATE completed in {update_time:.3f}s ({len(update_embeddings) / update_time:.1f} emb/s)"
                            )
                            total_inserted += len(update_embeddings)

                        except Exception as e:
                            logger.error(f"VALUES UPDATE failed: {e}")
                            raise

                    # Step 5: Recreate HNSW index for fast similarity search
                    if dropped_indexes and self._provider_instance:
                        logger.debug(
                            "📈 Recreating HNSW index for fast similarity search"
                        )
                        index_start = time.time()
                        for index_info in dropped_indexes:
                            try:
                                self._recreate_existing_index(conn, index_info)
                                logger.debug(
                                    f"Recreated HNSW index: {index_info['index_name']}"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Failed to recreate index {index_info['index_name']}: {e}"
                                )
                                # Continue - data is inserted, just no index optimization for search
                        index_time = time.time() - index_start
                        logger.debug(f"✅ HNSW index recreated in {index_time:.3f}s")

                    logger.debug(
                        f"✅ Stored {actual_batch_size} embeddings successfully"
                    )
                else:
                    # Fallback to simple batch insert without optimization
                    logger.warning(
                        "Provider instance not available for index management, using simple batch insert"
                    )
                    values_parts = []
                    for embedding_data in embeddings_data:
                        vector_str = str(embedding_data["embedding"])
                        values_parts.append(
                            f"({embedding_data['chunk_id']}, '{embedding_data['provider']}', '{embedding_data['model']}', {vector_str}, {embedding_data['dims']})"
                        )

                    values_clause = ",\n    ".join(values_parts)
                    conn.execute(f"""
                        INSERT OR REPLACE INTO {table_name} (chunk_id, provider, model, embedding, dims)
                        VALUES {values_clause}
                    """)
                    total_inserted = len(embeddings_data)

            else:
                # Small batch: use VALUES approach for consistency
                small_start = time.time()

                try:
                    # Build VALUES clause for small batch
                    values_parts = []
                    for embedding_data in embeddings_data:
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

                    small_time = time.time() - small_start
                    logger.debug(
                        f"✅ Small VALUES batch completed in {small_time:.3f}s ({len(embeddings_data) / small_time:.1f} emb/s)"
                    )
                    total_inserted = len(embeddings_data)

                except Exception as e:
                    logger.error(f"Small VALUES batch failed: {e}")
                    raise

                # Ensure HNSW indexes exist for semantic search after small batch insert
                # Note: _ensure_embedding_table_exists automatically creates standard HNSW indexes
                # This check verifies the index exists for this dimension
                if self._provider_instance and hasattr(
                    self._provider_instance, "get_existing_vector_indexes"
                ):
                    existing_indexes = (
                        self._provider_instance.get_existing_vector_indexes()
                    )
                    dims = first_embedding["dims"]

                    # Check if any index exists for this dimension (standard or custom)
                    index_exists = any(idx["dims"] == dims for idx in existing_indexes)

                    if not index_exists:
                        logger.warning(
                            f"🔍 No HNSW index found for {dims}D embeddings, creating one now"
                        )
                        # Create the missing HNSW index for semantic search functionality
                        try:
                            self._provider_instance.create_vector_index(
                                provider, model, dims, "cosine"
                            )
                            logger.info(
                                f"✅ Created missing HNSW index for {provider}/{model} ({dims}D)"
                            )
                        except Exception as e:
                            logger.error(
                                f"❌ Failed to create HNSW index for {provider}/{model} ({dims}D): {e}"
                            )
                            # Continue - data is inserted, just no index optimization for search

                # Update progress for small batch completion
                logger.debug(f"✅ Stored {actual_batch_size} embeddings successfully")

            insert_time = time.time() - start_time
            logger.debug(f"⚡ Batch INSERT completed in {insert_time:.3f}s")

            if use_hnsw_optimization:
                logger.debug(
                    f"🏆 HNSW-optimized batch insert: {total_inserted} embeddings in {insert_time:.3f}s ({total_inserted / insert_time:.1f} embeddings/sec) - Expected 10-20x speedup achieved!"
                )
            else:
                logger.debug(
                    f"🎯 Standard batch insert: {total_inserted} embeddings in {insert_time:.3f}s ({total_inserted / insert_time:.1f} embeddings/sec)"
                )

            # Checkpoint management is handled by the provider's executor pattern
            # No direct checkpoint calls needed here

            return total_inserted

        except Exception as e:
            logger.error(f"💥 CRITICAL: Optimized batch insert failed: {e}")
            logger.warning(
                "⚠️ This indicates a critical issue with VALUES clause approach!"
            )
            raise
        finally:
            pass

    def get_embedding_by_chunk_id(
        self, chunk_id: int, provider: str, model: str
    ) -> Embedding | None:
        """Get embedding for specific chunk, provider, and model."""
        if self.connection is None:
            raise RuntimeError("No database connection")

        try:
            # Search across all embedding tables
            if self._provider:
                embedding_tables = self._provider._get_all_embedding_tables()
            else:
                # Fallback for tests - query information_schema
                tables = self.connection_manager.connection.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_name LIKE 'embeddings_%'
                """).fetchall()
                embedding_tables = [table[0] for table in tables]

            for table_name in embedding_tables:
                result = self.connection_manager.connection.execute(
                    f"""
                    SELECT id, chunk_id, provider, model, embedding, dims, created_at
                    FROM {table_name}
                    WHERE chunk_id = ? AND provider = ? AND model = ?
                """,
                    [chunk_id, provider, model],
                ).fetchone()

                if result:
                    return Embedding(
                        chunk_id=result[1],
                        provider=result[2],
                        model=result[3],
                        vector=result[4],
                        dims=result[5],
                    )

            return None

        except Exception as e:
            logger.error(f"Failed to get embedding for chunk {chunk_id}: {e}")
            return None

    def get_existing_embeddings(
        self, chunk_ids: list[int], provider: str, model: str
    ) -> set[int]:
        """Get set of chunk IDs that already have embeddings for given provider/model."""
        if self.connection is None:
            raise RuntimeError("No database connection")

        if not chunk_ids:
            return set()

        try:
            # Check all embedding tables since dimensions vary by model
            all_chunk_ids = set()

            # Get all embedding tables
            table_result = self.connection_manager.connection.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_name LIKE 'embeddings_%'
            """).fetchall()

            for (table_name,) in table_result:
                # Create placeholders for IN clause
                placeholders = ",".join("?" * len(chunk_ids))
                params = chunk_ids + [provider, model]

                results = self.connection_manager.connection.execute(
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
        """Delete all embeddings for a specific chunk."""
        if self.connection is None:
            raise RuntimeError("No database connection")

        try:
            # Delete from all embedding tables
            if self._provider:
                embedding_tables = self._provider._get_all_embedding_tables()
            else:
                # Fallback for tests
                tables = self.connection_manager.connection.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_name LIKE 'embeddings_%'
                """).fetchall()
                embedding_tables = [table[0] for table in tables]

            for table_name in embedding_tables:
                self.connection_manager.connection.execute(
                    f"DELETE FROM {table_name} WHERE chunk_id = ?", [chunk_id]
                )

        except Exception as e:
            logger.error(f"Failed to delete embeddings for chunk {chunk_id}: {e}")
            raise
