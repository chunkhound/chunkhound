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

    # NOTE: connection property was removed - connections are managed by the executor
    # All operations must go through the provider's executor methods

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
        if not self._provider:
            raise RuntimeError("Provider required for database operations")

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
            return self._provider._execute_in_db_thread_sync("insert_file", file)

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
        if not self._provider:
            raise RuntimeError("Provider required for database operations")

        try:
            return self._provider._execute_in_db_thread_sync(
                "get_file_by_path", path, as_model
            )
        except Exception as e:
            logger.error(f"Failed to get file by path {path}: {e}")
            return None

    def get_file_by_id(
        self, file_id: int, as_model: bool = False
    ) -> dict[str, Any] | File | None:
        """Get file record by ID."""
        if not self._provider:
            raise RuntimeError("Provider required for database operations")

        try:
            return self._provider._execute_in_db_thread_sync(
                "get_file_by_id_query", file_id, as_model
            )
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
        if not self._provider:
            raise RuntimeError("Provider required for database operations")

        # Skip if no updates provided
        if size_bytes is None and mtime is None and content_hash is None:
            return

        try:
            self._provider._execute_in_db_thread_sync(
                "update_file", file_id, size_bytes, mtime, content_hash
            )
        except Exception as e:
            logger.error(f"Failed to update file {file_id}: {e}")
            raise

    def delete_file_completely(self, file_path: str) -> bool:
        """Delete a file and all its chunks/embeddings completely."""
        if not self._provider:
            raise RuntimeError("Provider required for database operations")

        try:
            return self._provider._execute_in_db_thread_sync(
                "delete_file_completely", file_path
            )
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False

    def get_file_stats(self, file_id: int) -> dict[str, Any]:
        """Get statistics for a specific file."""
        if not self._provider:
            raise RuntimeError("Provider required for database operations")

        try:
            return self._provider._execute_in_db_thread_sync(
                "get_file_stats", file_id
            )
        except Exception as e:
            logger.error(f"Failed to get file stats for {file_id}: {e}")
            return {}
