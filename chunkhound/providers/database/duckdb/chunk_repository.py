"""DuckDB chunk repository implementation - handles chunk CRUD operations."""

import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.core.models import Chunk
from chunkhound.core.types.common import ChunkType, Language

if TYPE_CHECKING:
    from chunkhound.providers.database.duckdb.connection_manager import (
        DuckDBConnectionManager,
    )


class DuckDBChunkRepository:
    """Repository for chunk CRUD operations in DuckDB."""

    def __init__(self, connection_manager: "DuckDBConnectionManager", provider=None):
        """Initialize chunk repository.

        Args:
            connection_manager: DuckDB connection manager instance
            provider: Optional provider instance for transaction-aware connections
        """
        self._connection_manager = connection_manager
        self._provider = provider

    @property
    def connection(self) -> Any | None:
        """Get database connection from connection manager."""
        return self._connection_manager.connection

    def insert_chunk(self, chunk: Chunk) -> int:
        """Insert chunk record and return chunk ID."""
        if self.connection is None:
            raise RuntimeError("No database connection")

        try:
            # Delegate to provider's executor for thread safety
            if self._provider:
                return self._provider._execute_in_db_thread_sync(
                    "insert_chunk_single", chunk
                )
            else:
                # Fallback for tests
                result = self._connection_manager.connection.execute(
                    """
                    INSERT INTO chunks (file_id, chunk_type, symbol, code, start_line, end_line,
                                      start_byte, end_byte, language, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    RETURNING id
                """,
                    [
                        chunk.file_id,
                        chunk.chunk_type.value if chunk.chunk_type else None,
                        chunk.symbol,
                        chunk.code,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.start_byte,
                        chunk.end_byte,
                        chunk.language.value if chunk.language else None,
                        json.dumps(chunk.metadata) if chunk.metadata else None,
                    ],
                ).fetchone()

                return result[0] if result else 0

        except Exception as e:
            logger.error(f"Failed to insert chunk: {e}")
            raise

    def insert_chunks_batch(self, chunks: list[Chunk]) -> list[int]:
        """Insert multiple chunks in batch using optimized DuckDB bulk loading."""
        if self.connection is None:
            raise RuntimeError("No database connection")

        if not chunks:
            return []

        try:
            # Prepare values for bulk INSERT statement
            values_clauses = []
            params = []

            for chunk in chunks:
                values_clauses.append("(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
                params.extend(
                    [
                        chunk.file_id,
                        chunk.chunk_type.value if chunk.chunk_type else None,
                        chunk.symbol,
                        chunk.code,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.start_byte,
                        chunk.end_byte,
                        chunk.language.value if chunk.language else None,
                        json.dumps(chunk.metadata) if chunk.metadata else None,
                    ]
                )

            # Use single bulk INSERT with RETURNING for optimal performance
            values_sql = ", ".join(values_clauses)
            query = f"""
                INSERT INTO chunks (file_id, chunk_type, symbol, code, start_line, end_line,
                                  start_byte, end_byte, language, metadata)
                VALUES {values_sql}
                RETURNING id
            """

            # Execute bulk insert and get all IDs in one operation
            if self._provider:
                # Provider not used for batch operations currently
                results = self._connection_manager.connection.execute(
                    query, params
                ).fetchall()
            else:
                results = self._connection_manager.connection.execute(
                    query, params
                ).fetchall()
            chunk_ids = [result[0] for result in results]

            return chunk_ids

        except Exception as e:
            logger.error(f"Failed to insert chunks batch: {e}")
            raise

    def get_chunk_by_id(
        self, chunk_id: int, as_model: bool = False
    ) -> dict[str, Any] | Chunk | None:
        """Get chunk record by ID."""
        if self.connection is None:
            raise RuntimeError("No database connection")

        try:
            if self._provider:
                result = self._provider._execute_in_db_thread_sync(
                    "get_chunk_by_id_query", chunk_id
                )
            else:
                result = self._connection_manager.connection.execute(
                    """
                    SELECT id, file_id, chunk_type, symbol, code, start_line, end_line,
                           start_byte, end_byte, language, created_at, updated_at, metadata
                    FROM chunks WHERE id = ?
                """,
                    [chunk_id],
                ).fetchone()

            if not result:
                return None

            chunk_dict = {
                "id": result[0],
                "file_id": result[1],
                "chunk_type": result[2],
                "symbol": result[3],
                "code": result[4],
                "start_line": result[5],
                "end_line": result[6],
                "start_byte": result[7],
                "end_byte": result[8],
                "language": result[9],
                "created_at": result[10],
                "updated_at": result[11],
                "metadata": json.loads(result[12]) if result[12] else {},
            }

            if as_model:
                return Chunk(
                    file_id=result[1],
                    chunk_type=ChunkType(result[2]) if result[2] else ChunkType.UNKNOWN,
                    symbol=result[3],
                    code=result[4],
                    start_line=result[5],
                    end_line=result[6],
                    start_byte=result[7],
                    end_byte=result[8],
                    language=Language(result[9]) if result[9] else Language.UNKNOWN,
                    metadata=chunk_dict["metadata"],
                )

            return chunk_dict

        except Exception as e:
            logger.error(f"Failed to get chunk by ID {chunk_id}: {e}")
            return None

    def get_chunks_by_file_id(
        self, file_id: int, as_model: bool = False
    ) -> list[dict[str, Any] | Chunk]:
        """Get all chunks for a specific file."""
        if self.connection is None:
            raise RuntimeError("No database connection")

        try:
            if self._provider:
                results = self._provider._execute_in_db_thread_sync(
                    "get_chunks_by_file_id_query", file_id
                )
            else:
                results = self._connection_manager.connection.execute(
                    """
                    SELECT id, file_id, chunk_type, symbol, code, start_line, end_line,
                           start_byte, end_byte, language, created_at, updated_at, metadata
                    FROM chunks WHERE file_id = ?
                    ORDER BY start_line
                """,
                    [file_id],
                ).fetchall()

            chunks = []
            for result in results:
                chunk_dict = {
                    "id": result[0],
                    "file_id": result[1],
                    "chunk_type": result[2],
                    "symbol": result[3],
                    "code": result[4],
                    "start_line": result[5],
                    "end_line": result[6],
                    "start_byte": result[7],
                    "end_byte": result[8],
                    "language": result[9],
                    "created_at": result[10],
                    "updated_at": result[11],
                    "metadata": json.loads(result[12]) if result[12] else {},
                }

                if as_model:
                    chunks.append(
                        Chunk(
                            file_id=result[1],
                            chunk_type=ChunkType(result[2])
                            if result[2]
                            else ChunkType.UNKNOWN,
                            symbol=result[3],
                            code=result[4],
                            start_line=result[5],
                            end_line=result[6],
                            start_byte=result[7],
                            end_byte=result[8],
                            language=Language(result[9])
                            if result[9]
                            else Language.UNKNOWN,
                            metadata=chunk_dict["metadata"],
                        )
                    )
                else:
                    chunks.append(chunk_dict)

            return chunks

        except Exception as e:
            logger.error(f"Failed to get chunks for file {file_id}: {e}")
            return []

    def delete_file_chunks(self, file_id: int) -> None:
        """Delete all chunks for a file."""
        if self.connection is None:
            raise RuntimeError("No database connection")

        try:
            # Delegate to provider if available for proper executor handling
            if self._provider:
                self._provider._execute_in_db_thread_sync("delete_file_chunks", file_id)
            else:
                # Fallback for tests - simplified version without embedding cleanup
                self._connection_manager.connection.execute(
                    "DELETE FROM chunks WHERE file_id = ?", [file_id]
                )

        except Exception as e:
            logger.error(f"Failed to delete chunks for file {file_id}: {e}")
            raise

    def delete_chunk(self, chunk_id: int) -> None:
        """Delete a single chunk by ID."""
        if self.connection is None:
            raise RuntimeError("No database connection")

        try:
            # Delegate to provider if available for proper executor handling
            if self._provider:
                self._provider._execute_in_db_thread_sync("delete_chunk", chunk_id)
            else:
                # Fallback for tests - simplified version without embedding cleanup
                self._connection_manager.connection.execute(
                    "DELETE FROM chunks WHERE id = ?", [chunk_id]
                )

        except Exception as e:
            logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            raise

    def update_chunk(self, chunk_id: int, **kwargs) -> None:
        """Update chunk record with new values."""
        if self.connection is None:
            raise RuntimeError("No database connection")

        if not kwargs:
            return

        try:
            # Build dynamic update query
            set_clauses = []
            values = []

            valid_fields = [
                "chunk_type",
                "symbol",
                "code",
                "start_line",
                "end_line",
                "start_byte",
                "end_byte",
                "language",
            ]

            for key, value in kwargs.items():
                if key in valid_fields:
                    set_clauses.append(f"{key} = ?")
                    values.append(value)

            if set_clauses:
                set_clauses.append("updated_at = CURRENT_TIMESTAMP")
                values.append(chunk_id)

                query = f"UPDATE chunks SET {', '.join(set_clauses)} WHERE id = ?"
                if self._provider:
                    self._provider._execute_in_db_thread_sync(
                        "update_chunk_query", chunk_id, query, values
                    )
                else:
                    self._connection_manager.connection.execute(query, values)

        except Exception as e:
            logger.error(f"Failed to update chunk {chunk_id}: {e}")
            raise

    def get_chunks_in_range(
        self, file_id: int, start_line: int, end_line: int
    ) -> list[dict]:
        """Get all chunks overlapping a line range (pattern from context_retriever.py).

        Args:
            file_id: ID of the file to search within
            start_line: Start line of the range
            end_line: End line of the range

        Returns:
            List of chunk dictionaries overlapping the range, ordered by start_line
        """
        if self.connection is None:
            raise RuntimeError("No database connection")

        try:
            # Overlap condition: chunk overlaps if any of:
            # - chunk start_line is within range
            # - chunk end_line is within range
            # - chunk spans the entire range
            query = """
                SELECT id, file_id, chunk_type, symbol, code, start_line, end_line,
                       start_byte, end_byte, language, created_at, updated_at, metadata
                FROM chunks
                WHERE file_id = ?
                AND (
                    (start_line BETWEEN ? AND ?) OR
                    (end_line BETWEEN ? AND ?) OR
                    (start_line <= ? AND end_line >= ?)
                )
                ORDER BY start_line
            """

            if self._provider:
                results = self._provider._execute_in_db_thread_sync(
                    "get_chunks_in_range_query", file_id, start_line, end_line, query
                )
            else:
                results = self._connection_manager.connection.execute(
                    query,
                    [file_id, start_line, end_line, start_line, end_line, start_line, end_line],
                ).fetchall()

            chunks = []
            for result in results:
                metadata_json = result[12]
                chunk_dict = {
                    "id": result[0],
                    "file_id": result[1],
                    "chunk_type": result[2],
                    "symbol": result[3],
                    "code": result[4],
                    "start_line": result[5],
                    "end_line": result[6],
                    "start_byte": result[7],
                    "end_byte": result[8],
                    "language": result[9],
                    "created_at": result[10],
                    "updated_at": result[11],
                    "metadata": json.loads(metadata_json) if metadata_json else {},
                }
                chunks.append(chunk_dict)

            return chunks

        except Exception as e:
            logger.error(f"Failed to get chunks in range for file {file_id}: {e}")
            return []

    def get_all_chunks_with_metadata(self) -> list[dict[str, Any]]:
        """Get all chunks with their metadata including file paths (provider-agnostic)."""
        if self.connection is None:
            raise RuntimeError("No database connection")

        try:
            # Use SQL to get chunks with file paths (DuckDB approach)
            query = """
                SELECT c.id, c.file_id, f.path as file_path, c.code,
                       c.start_line, c.end_line, c.chunk_type, c.language, c.symbol,
                       c.metadata
                FROM chunks c
                JOIN files f ON c.file_id = f.id
                ORDER BY c.id
            """

            if self._provider:
                results = self._provider._execute_in_db_thread_sync(
                    "get_all_chunks_with_metadata_query", query
                )
            else:
                results = self._connection_manager.connection.execute(query).fetchall()

            # Convert to list of dictionaries
            result = []
            for row in results:
                result.append(
                    {
                        "id": row[0],
                        "file_id": row[1],
                        "file_path": row[2],
                        "content": row[3],
                        "start_line": row[4],
                        "end_line": row[5],
                        "chunk_type": row[6],
                        "language": row[7],
                        "name": row[8],
                        "metadata": json.loads(row[9]) if row[9] else {},
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Failed to get all chunks with metadata: {e}")
            return []
