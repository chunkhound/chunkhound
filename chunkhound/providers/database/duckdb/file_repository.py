"""DuckDB file repository implementation for ChunkHound - handles file CRUD operations."""

from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.core.models import File
from chunkhound.core.types.common import Language

if TYPE_CHECKING:
    from chunkhound.providers.database.duckdb.connection_manager import (
        DuckDBConnectionManager,
    )


class DuckDBFileRepository:
    """Repository for file CRUD operations using DuckDB."""

    def __init__(self, connection_manager: "DuckDBConnectionManager", provider=None):
        """Initialize file repository with connection manager.

        Args:
            connection_manager: DuckDB connection manager instance
            provider: Optional provider instance for transaction-aware connections
        """
        self.connection_manager = connection_manager
        self._provider = provider

    @property
    def connection(self) -> Any | None:
        """Get the database connection from connection manager."""
        return self.connection_manager.connection

    def _extract_file_id(self, file_record: dict[str, Any] | File) -> int | None:
        """Safely extract file ID from either dict or File model."""
        if isinstance(file_record, File):
            return file_record.id
        elif isinstance(file_record, dict) and "id" in file_record:
            return file_record["id"]
        else:
            return None

    def insert_file(self, file: File) -> int:
        """Insert file record and return file ID.

        If file with same path exists, updates metadata.
        """
        if self.connection is None:
            raise RuntimeError("No database connection")

        try:
            # First try to find existing file by path
            existing = self.get_file_by_path(str(file.path))
            if existing:
                # File exists, update it
                file_id = self._extract_file_id(existing)
                if file_id is not None:
                    self.update_file(
                        file_id, size_bytes=file.size_bytes, mtime=file.mtime,
                        content_hash=file.content_hash
                    )
                    return file_id

            # No existing file, insert new one
            if self._provider:
                # Delegate to provider for proper executor handling
                return self._provider._execute_in_db_thread_sync("insert_file", file)
            else:
                # Fallback for tests
                result = self.connection_manager.connection.execute(
                    """
                    INSERT INTO files (path, name, extension, size, modified_time, language)
                    VALUES (?, ?, ?, ?, to_timestamp(?), ?)
                    RETURNING id
                """,
                    [
                        file.path,
                        file.name,
                        file.extension,
                        file.size_bytes,
                        file.mtime,
                        file.language.value if file.language else None,
                    ],
                ).fetchone()

                file_id = result[0] if result else 0
                return file_id

        except Exception as e:
            logger.error(f"Failed to insert file {file.path}: {e}")
            # Return existing file ID if constraint error (duplicate)
            if "Duplicate key" in str(e) and "violates unique constraint" in str(e):
                existing = self.get_file_by_path(str(file.path))
                if existing and isinstance(existing, dict) and "id" in existing:
                    logger.info(f"Returning existing file ID for {file.path}")
                    return existing["id"]
            raise

    def get_file_by_path(
        self, path: str, as_model: bool = False
    ) -> dict[str, Any] | File | None:
        """Get file record by path."""
        if self.connection is None:
            raise RuntimeError("No database connection")

        try:
            if self._provider:
                return self._provider._execute_in_db_thread_sync(
                    "get_file_by_path", path, as_model
                )
            else:
                # Fallback for tests
                # Normalize path to handle both absolute and relative paths
                from chunkhound.core.utils import normalize_path_for_lookup

                base_dir = (
                    self._provider.get_base_directory() if self._provider else None
                )
                lookup_path = normalize_path_for_lookup(path, base_dir)
                result = self.connection_manager.connection.execute(
                    """
                    SELECT id, path, name, extension, size, modified_time, language, created_at, updated_at
                    FROM files WHERE path = ?
                """,
                    [lookup_path],
                ).fetchone()

            if not result:
                return None

            file_dict = {
                "id": result[0],
                "path": result[1],
                "name": result[2],
                "extension": result[3],
                "size": result[4],
                "modified_time": result[5],
                "language": result[6],
                "created_at": result[7],
                "updated_at": result[8],
            }

            if as_model:
                return File(
                    path=result[1],
                    mtime=result[5],
                    size_bytes=result[4],
                    language=Language(result[6]) if result[6] else Language.UNKNOWN,
                )

            return file_dict

        except Exception as e:
            logger.error(f"Failed to get file by path {path}: {e}")
            return None

    def get_file_by_id(
        self, file_id: int, as_model: bool = False
    ) -> dict[str, Any] | File | None:
        """Get file record by ID."""
        if self.connection is None:
            raise RuntimeError("No database connection")

        try:
            if self._provider:
                return self._provider._execute_in_db_thread_sync(
                    "get_file_by_id_query", file_id, as_model
                )
            else:
                # Fallback for tests
                result = self.connection_manager.connection.execute(
                    """
                    SELECT id, path, name, extension, size, modified_time, language, created_at, updated_at
                    FROM files WHERE id = ?
                """,
                    [file_id],
                ).fetchone()

            if not result:
                return None

            file_dict = {
                "id": result[0],
                "path": result[1],
                "name": result[2],
                "extension": result[3],
                "size": result[4],
                "modified_time": result[5],
                "language": result[6],
                "created_at": result[7],
                "updated_at": result[8],
            }

            if as_model:
                return File(
                    path=result[1],
                    mtime=result[5],
                    size_bytes=result[4],
                    language=Language(result[6]) if result[6] else Language.UNKNOWN,
                )

            return file_dict

        except Exception as e:
            logger.error(f"Failed to get file by ID {file_id}: {e}")
            return None

    def update_file(
        self, file_id: int, size_bytes: int | None = None, mtime: float | None = None,
        content_hash: str | None = None
    ) -> None:
        """Update file record with new values.

        Args:
            file_id: ID of the file to update
            size_bytes: New file size in bytes
            mtime: New modification timestamp
            content_hash: Content hash for change detection
        """
        if self.connection is None:
            raise RuntimeError("No database connection")

        # Skip if no updates provided
        if size_bytes is None and mtime is None and content_hash is None:
            return

        try:
            # Build dynamic update query
            set_clauses = []
            values = []

            # Add size update if provided
            if size_bytes is not None:
                set_clauses.append("size = ?")
                values.append(size_bytes)

            # Add timestamp update if provided
            if mtime is not None:
                set_clauses.append("modified_time = to_timestamp(?)")
                values.append(mtime)

            # Add content hash update if provided
            if content_hash is not None:
                set_clauses.append("content_hash = ?")
                values.append(content_hash)

            if set_clauses:
                set_clauses.append("updated_at = CURRENT_TIMESTAMP")
                values.append(file_id)

                query = f"UPDATE files SET {', '.join(set_clauses)} WHERE id = ?"
                if self._provider:
                    self._provider._execute_in_db_thread_sync(
                        "update_file", file_id, size_bytes, mtime, content_hash
                    )
                else:
                    # Fallback for tests
                    self.connection_manager.connection.execute(query, values)

        except Exception as e:
            logger.error(f"Failed to update file {file_id}: {e}")
            raise

    def delete_file_completely(self, file_path: str) -> bool:
        """Delete a file and all its chunks/embeddings completely."""
        if self.connection is None:
            raise RuntimeError("No database connection")

        try:
            # Get file ID first
            file_record = self.get_file_by_path(file_path)
            if not file_record:
                return False

            file_id = (
                file_record["id"] if isinstance(file_record, dict) else file_record.id
            )

            # Delete in correct order due to foreign key constraints
            # 1. Delete embeddings first
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
                    f"""
                    DELETE FROM {table_name}
                    WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ?)
                """,
                    [file_id],
                )

            # 2. Delete chunks
            self.connection_manager.connection.execute(
                "DELETE FROM chunks WHERE file_id = ?", [file_id]
            )

            # 3. Delete file
            self.connection_manager.connection.execute(
                "DELETE FROM files WHERE id = ?", [file_id]
            )

            logger.debug(f"File {file_path} and all associated data deleted")
            return True

        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False

    def get_file_stats(self, file_id: int) -> dict[str, Any]:
        """Get statistics for a specific file."""
        if self.connection is None:
            raise RuntimeError("No database connection")

        try:
            # Get file info
            file_result = self.connection_manager.connection.execute(
                """
                SELECT path, name, extension, size, language
                FROM files WHERE id = ?
            """,
                [file_id],
            ).fetchone()

            if not file_result:
                return {}

            # Get chunk count and types
            chunk_results = self.connection_manager.connection.execute(
                """
                SELECT chunk_type, COUNT(*) as count
                FROM chunks WHERE file_id = ?
                GROUP BY chunk_type
            """,
                [file_id],
            ).fetchall()

            chunk_types = {result[0]: result[1] for result in chunk_results}
            total_chunks = sum(chunk_types.values())

            # Get embedding count across all embedding tables
            embedding_count = 0
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
                count = self.connection_manager.connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM {table_name} e
                    JOIN chunks c ON e.chunk_id = c.id
                    WHERE c.file_id = ?
                """,
                    [file_id],
                ).fetchone()[0]
                embedding_count += count

            return {
                "file_id": file_id,
                "path": file_result[0],
                "name": file_result[1],
                "extension": file_result[2],
                "size": file_result[3],
                "language": file_result[4],
                "chunks": total_chunks,
                "chunk_types": chunk_types,
                "embeddings": embedding_count,
            }

        except Exception as e:
            logger.error(f"Failed to get file stats for {file_id}: {e}")
            return {}

    def _maybe_checkpoint(self, force: bool = False) -> None:
        """Perform checkpoint if needed - delegate to provider."""
        if self._provider:
            self._provider._maybe_checkpoint(force)
        else:
            # Fallback for tests - no checkpoint needed
            pass
