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
        self._provider_instance: Any | None = None  # Will be set by provider

    @property
    def connection(self) -> Any | None:
        """Get database connection from connection manager."""
        return self.connection_manager.connection

    def set_provider_instance(self, provider_instance: Any) -> None:
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
            "Cannot drop HNSW index without identity information: "
            f"{index_info['index_name']}"
        )

    def _recreate_existing_index(self, conn: Any, index_info: dict[str, Any]) -> None:
        """Recreate one discovered HNSW index with its original SQL when available."""
        if self._provider_instance and hasattr(
            self._provider_instance, "_executor_recreate_vector_index_from_info"
        ):
            # Stay on the caller's transactional connection so drop/recreate
            # sees one consistent DuckDB index catalog.
            self._provider_instance._executor_recreate_vector_index_from_info(
                conn,
                {},
                index_info,
            )
            return

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
            "Cannot recreate HNSW index without SQL or identity: "
            f"{index_info['index_name']}"
        )

    @staticmethod
    def build_embedding_upsert_sql(table_name: str) -> str:
        """Build the parameterized upsert SQL for one embedding table."""
        return f"""
            INSERT INTO {table_name} (chunk_id, provider, model, embedding, dims)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (chunk_id, provider, model) DO UPDATE
            SET embedding = EXCLUDED.embedding, dims = EXCLUDED.dims
        """

    @staticmethod
    def _quote_duckdb_identifier(identifier: str) -> str:
        """Quote a DuckDB identifier safely."""
        return f'"{identifier.replace(chr(34), chr(34) * 2)}"'

    @staticmethod
    def _is_hnsw_index(index_name: str, create_sql: str | None) -> bool:
        """Return True when the index metadata describes an HNSW index."""
        if create_sql and "USING HNSW" in create_sql.upper():
            return True
        return index_name.startswith("hnsw_") or index_name.startswith("idx_hnsw_")

    def _fallback_index_exists(
        self, conn: Any, table_name: str, index_name: str
    ) -> bool:
        """Return True when one named index exists on the fallback table."""
        result = conn.execute(
            """
            SELECT 1
            FROM duckdb_indexes()
            WHERE table_name = ? AND index_name = ?
            LIMIT 1
            """,
            [table_name, index_name],
        ).fetchone()
        return result is not None

    def _get_fallback_hnsw_indexes(
        self, conn: Any, table_name: str
    ) -> list[dict[str, str | None]]:
        """Discover exact HNSW index identity for a fallback embedding table."""
        rows = conn.execute(
            """
            SELECT index_name, sql
            FROM duckdb_indexes()
            WHERE table_name = ?
            ORDER BY index_name
            """,
            [table_name],
        ).fetchall()
        indexes: list[dict[str, str | None]] = []
        for index_name, create_sql in rows:
            if not self._is_hnsw_index(index_name, create_sql):
                continue
            indexes.append({"index_name": index_name, "create_sql": create_sql})
        return indexes

    def _get_fallback_duplicate_row_ids(
        self, conn: Any, table_name: str
    ) -> list[int]:
        """Return duplicate row ids that block the unique upsert contract."""
        rows = conn.execute(
            f"""
            SELECT id
            FROM (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        PARTITION BY chunk_id, provider, model
                        ORDER BY created_at DESC NULLS LAST, id DESC
                    ) AS row_num
                FROM {table_name}
            )
            WHERE row_num > 1
            ORDER BY id
            """
        ).fetchall()
        return [int(row[0]) for row in rows]

    def _delete_rows_by_id(self, conn: Any, table_name: str, row_ids: list[int]) -> None:
        """Delete specific rows from one embedding table by primary key."""
        if not row_ids:
            return

        placeholders = ", ".join(["?"] * len(row_ids))
        conn.execute(f"DELETE FROM {table_name} WHERE id IN ({placeholders})", row_ids)

    def _ensure_fallback_embedding_upsert_contract(
        self, conn: Any, table_name: str, dims: int
    ) -> None:
        """Upgrade an existing fallback embedding table before using ON CONFLICT."""
        provider_model_index = f"idx_{dims}_provider_model"
        if not self._fallback_index_exists(conn, table_name, provider_model_index):
            conn.execute(
                f"CREATE INDEX {provider_model_index} ON {table_name}(provider, model)"
            )

        unique_index_name = f"idx_{dims}_chunk_provider_model_unique"
        if self._fallback_index_exists(conn, table_name, unique_index_name):
            return

        duplicate_row_ids = self._get_fallback_duplicate_row_ids(conn, table_name)
        hnsw_indexes = self._get_fallback_hnsw_indexes(conn, table_name)

        try:
            conn.execute("BEGIN TRANSACTION")
            for index_info in hnsw_indexes:
                conn.execute(
                    "DROP INDEX IF EXISTS "
                    f"{self._quote_duckdb_identifier(str(index_info['index_name']))}"
                )

            if duplicate_row_ids:
                self._delete_rows_by_id(conn, table_name, duplicate_row_ids)

            conn.execute(
                f"""
                CREATE UNIQUE INDEX {unique_index_name}
                ON {table_name}(chunk_id, provider, model)
                """
            )

            for index_info in hnsw_indexes:
                create_sql = index_info["create_sql"]
                if create_sql:
                    conn.execute(create_sql)

            conn.execute("COMMIT")
        except Exception as e:
            try:
                conn.execute("ROLLBACK")
            except Exception as rollback_error:
                raise RuntimeError(
                    "Failed to upgrade fallback embedding table "
                    f"{table_name}: {e}; rollback failed: {rollback_error}"
                ) from rollback_error
            raise RuntimeError(
                f"Failed to upgrade fallback embedding table {table_name}: {e}"
            ) from e

    def _ensure_fallback_embedding_table_exists(self, conn: Any, dims: int) -> str:
        """Create the dimension-specific embedding table when no provider is available."""
        table_name = f"embeddings_{dims}"
        result = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        if result is not None:
            self._ensure_fallback_embedding_upsert_contract(conn, table_name, dims)
            return table_name

        conn.execute(f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY DEFAULT nextval('embeddings_id_seq'),
                chunk_id INTEGER REFERENCES chunks(id),
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                embedding FLOAT[{dims}],
                dims INTEGER NOT NULL DEFAULT {dims},
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{dims}_chunk_id ON {table_name}(chunk_id)")
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{dims}_provider_model ON {table_name}(provider, model)"
        )
        conn.execute(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{dims}_chunk_provider_model_unique
            ON {table_name}(chunk_id, provider, model)
            """
        )
        return table_name

    def _execute_upsert_batches(
        self, conn: Any, upsert_sql: str, batch_rows: list[tuple[Any, ...]]
    ) -> None:
        """Execute parameterized upserts in bounded chunks."""
        write_batch_size = 1000
        for offset in range(0, len(batch_rows), write_batch_size):
            conn.executemany(upsert_sql, batch_rows[offset : offset + write_batch_size])

    def _get_embedding_row_id(
        self,
        conn: Any,
        table_name: str,
        chunk_id: int,
        provider: str,
        model: str,
    ) -> int:
        """Return the canonical row id for one embedding upsert key."""
        result = conn.execute(
            f"""
            SELECT id
            FROM {table_name}
            WHERE chunk_id = ? AND provider = ? AND model = ?
            """,
            [chunk_id, provider, model],
        ).fetchone()
        if result is None:
            raise RuntimeError(
                "Embedding upsert completed without a stored row for "
                f"{table_name} ({chunk_id}, {provider}, {model})"
            )
        return int(result[0])

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
                table_name = self._ensure_fallback_embedding_table_exists(
                    self.connection_manager.connection, embedding.dims
                )

            conn = self.connection_manager.connection
            upsert_sql = self.build_embedding_upsert_sql(table_name)
            conn.execute(
                upsert_sql,
                [
                    embedding.chunk_id,
                    embedding.provider,
                    embedding.model,
                    embedding.vector,
                    embedding.dims,
                ],
            )
            embedding_id = self._get_embedding_row_id(
                conn,
                table_name,
                embedding.chunk_id,
                embedding.provider,
                embedding.model,
            )
            logger.debug(
                f"Upserted embedding {embedding_id} for chunk {embedding.chunk_id}"
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
        manage_transaction: bool = True,
    ) -> int:
        """Insert multiple embedding vectors with HNSW index optimization.

        For large batches (>= batch_size threshold), uses the Context7-recommended optimization:
        1. Drop HNSW indexes to avoid insert slowdown (60s+ -> 5s for 300 items)
        2. Use parameterized ON CONFLICT upserts for new and existing embeddings
        3. Recreate HNSW indexes after bulk operations

        Expected speedup: 10-20x faster for large batches (90s -> 5-10s).

        Args:
            embeddings_data: List of dicts with keys: chunk_id, provider, model, embedding, dims
            batch_size: Threshold for HNSW optimization (default: 50)
            connection: Optional database connection to use (for transaction contexts)
            manage_transaction: Whether this helper should own the batch transaction

        Returns:
            Number of successfully inserted embeddings
        """
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

        if self._provider:
            table_name = self._provider._ensure_embedding_table_exists(detected_dims)
        else:
            table_name = self._ensure_fallback_embedding_table_exists(
                conn, detected_dims
            )
        logger.debug(
            f"Using table {table_name} for {detected_dims}-dimensional embeddings"
        )

        first_embedding = embeddings_data[0]
        provider = first_embedding["provider"]
        model = first_embedding["model"]
        upsert_sql = self.build_embedding_upsert_sql(table_name)
        batch_rows = [
            (
                embedding_data["chunk_id"],
                embedding_data["provider"],
                embedding_data["model"],
                embedding_data["embedding"],
                embedding_data["dims"],
            )
            for embedding_data in embeddings_data
        ]

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
            transaction_started = False

            if manage_transaction:
                conn.execute("BEGIN TRANSACTION")
                transaction_started = True

            if use_hnsw_optimization:
                logger.debug(
                    f"🔧 Large batch detected ({actual_batch_size} embeddings >= {hnsw_threshold}), applying HNSW optimization"
                )

                dropped_indexes: list[dict[str, Any]] = []

                if self._provider_instance and hasattr(
                    self._provider_instance, "get_existing_vector_indexes"
                ):
                    dropped_indexes = [
                        index_info
                        for index_info in self._provider_instance.get_existing_vector_indexes()
                        if index_info["table_name"] == table_name
                    ]

                try:
                    if dropped_indexes:
                        conn.execute("SET preserve_insertion_order = false")
                        for index_info in dropped_indexes:
                            self._drop_existing_index(conn, index_info)
                            logger.debug(f"Dropped index: {index_info['index_name']}")

                    insert_start = time.time()
                    self._execute_upsert_batches(conn, upsert_sql, batch_rows)
                    total_inserted = len(batch_rows)
                    insert_time = time.time() - insert_start
                    logger.debug(
                        f"✅ Parameterized UPSERT completed in {insert_time:.3f}s ({total_inserted / insert_time:.1f} emb/s)"
                    )

                    if dropped_indexes:
                        logger.debug(
                            "📈 Recreating exact HNSW indexes after bulk embedding upsert"
                        )
                        for index_info in dropped_indexes:
                            self._recreate_existing_index(conn, index_info)
                            logger.debug(
                                f"Recreated HNSW index: {index_info['index_name']}"
                            )

                    if transaction_started:
                        conn.execute("COMMIT")
                        transaction_started = False

                except Exception as e:
                    if transaction_started:
                        try:
                            conn.execute("ROLLBACK")
                            transaction_started = False
                        except Exception as rollback_error:
                            raise RuntimeError(
                                "insert_embeddings_batch failed and rollback failed: "
                                f"{rollback_error}"
                            ) from rollback_error
                    raise RuntimeError(
                        "insert_embeddings_batch failed while enforcing strict HNSW restore: "
                        f"{e}"
                    ) from e

                logger.debug(f"✅ Stored {actual_batch_size} embeddings successfully")

            else:
                small_start = time.time()

                try:
                    self._execute_upsert_batches(conn, upsert_sql, batch_rows)

                    small_time = time.time() - small_start
                    logger.debug(
                        f"✅ Small parameterized UPSERT batch completed in {small_time:.3f}s ({len(embeddings_data) / small_time:.1f} emb/s)"
                    )
                    total_inserted = len(embeddings_data)
                except Exception as e:
                    if transaction_started:
                        try:
                            conn.execute("ROLLBACK")
                        except Exception as rollback_error:
                            raise RuntimeError(
                                "insert_embeddings_batch failed and rollback failed: "
                                f"{rollback_error}"
                            ) from rollback_error
                        transaction_started = False
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

                if transaction_started:
                    conn.execute("COMMIT")
                    transaction_started = False

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
            if transaction_started:
                try:
                    conn.execute("ROLLBACK")
                    transaction_started = False
                except Exception as rollback_error:
                    logger.error(
                        "💥 CRITICAL: Optimized batch insert failed and rollback "
                        f"failed: {rollback_error}"
                    )
                    raise RuntimeError(
                        "insert_embeddings_batch failed and rollback failed: "
                        f"{rollback_error}"
                    ) from rollback_error
            logger.error(f"💥 CRITICAL: Optimized batch insert failed: {e}")
            raise

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
