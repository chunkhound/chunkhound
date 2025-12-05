"""
Test disk usage limiting functionality.

This module tests the disk usage limiting feature that prevents ChunkHound
from exceeding configured database size limits during indexing operations.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.exceptions import DiskUsageLimitExceededError
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.core.types.common import Language
from chunkhound.services.batch_processor import ParsedFileResult


def _create_parsed_file_result(
    file_path: Path,
    chunks: list[dict] | None = None,
    status: str = "ok"
) -> ParsedFileResult:
    """Helper to create ParsedFileResult for testing."""
    return ParsedFileResult(
        file_path=file_path,
        chunks=chunks or [],
        language=Language.PYTHON,
        file_size=100,
        file_mtime=0.0,
        status=status,
        error=None,
        content_hash=None,
    )


class TestDatabaseConfigDiskUsage:
    """Test disk usage configuration parameter."""

    def test_max_disk_usage_mb_default(self):
        """Test default value for max_disk_usage_mb."""
        config = DatabaseConfig()
        assert config.max_disk_usage_mb is None

    def test_max_disk_usage_mb_custom_value(self):
        """Test setting custom max_disk_usage_mb value."""
        config = DatabaseConfig(max_disk_usage_mb=550.0)
        assert config.max_disk_usage_mb == 550.0

    def test_load_from_env_with_disk_limit(self, monkeypatch):
        """Test loading max_disk_usage_mb from environment variables."""
        monkeypatch.setenv("CHUNKHOUND_DATABASE__MAX_DISK_USAGE_GB", "2.5")

        config = DatabaseConfig.load_from_env()
        assert config["max_disk_usage_mb"] == 2560.0

    def test_load_from_env_invalid_disk_limit_ignored(self, monkeypatch):
        """Test that invalid max_disk_usage_mb env var is silently ignored."""
        monkeypatch.setenv("CHUNKHOUND_DATABASE__MAX_DISK_USAGE_GB", "not-a-number")

        config = DatabaseConfig.load_from_env()
        # Invalid value should not be in config dict
        assert "max_disk_usage_mb" not in config

    def test_extract_cli_overrides_disk_limit(self):
        """Test CLI override for max_disk_usage_mb."""
        config = DatabaseConfig()
        args = MagicMock()
        args.max_disk_usage_gb = 3.0

        overrides = config.extract_cli_overrides(args)
        assert overrides["max_disk_usage_mb"] == 3072.0

    def test_repr_includes_disk_limit(self):
        """Test that __repr__ includes max_disk_usage_mb."""
        config = DatabaseConfig(max_disk_usage_mb=4300.8)
        repr_str = repr(config)
        assert "max_disk_usage_mb=4300.8" in repr_str


class TestDiskUsageLimitExceededError:
    """Test the DiskUsageLimitExceededError exception."""

    def test_exception_creation(self):
        """Test creating the exception with required parameters."""
        error = DiskUsageLimitExceededError(
            current_size_mb=2150.4,
            limit_mb=2048.0
        )
        assert error.current_size_mb == 2150.4
        assert error.limit_mb == 2048.0

    def test_exception_message(self):
        """Test the exception message format."""
        error = DiskUsageLimitExceededError(
            current_size_mb=1536.0,
            limit_mb=1024.0
        )
        expected_msg = "Database disk usage limit exceeded: 1536.0 MB >= 1024.0 MB"
        assert str(error) == expected_msg

    def test_exception_inheritance(self):
        """Test that the exception inherits from ChunkHoundError."""
        from chunkhound.core.exceptions import ChunkHoundError

        error = DiskUsageLimitExceededError(1126.4, 1024.0)
        assert isinstance(error, ChunkHoundError)


class TestIndexingCoordinatorDiskUsage:
    """Test disk usage checking in IndexingCoordinator."""

    def test_check_disk_usage_limit_no_config(self, tmp_path):
        """Test that no limit check occurs when config is None."""
        db = DuckDBProvider(db_path=tmp_path / "db", base_directory=tmp_path)
        coord = IndexingCoordinator(database_provider=db, base_directory=tmp_path)

        # Should not raise any exception
        coord._check_disk_usage_limit()

    def test_check_disk_usage_limit_no_limit_set(self, tmp_path):
        """Test that no limit check occurs when limit is not set."""
        db = DuckDBProvider(db_path=tmp_path / "db", base_directory=tmp_path)
        config = DatabaseConfig(max_disk_usage_mb=None)
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=tmp_path,
            config=config
        )

        # Should not raise any exception
        coord._check_disk_usage_limit()

    def test_check_disk_usage_limit_under_limit(self, tmp_path):
        """Test that no exception is raised when under the limit."""
        db_path = tmp_path / "test.db"
        # Create the actual database file with small content
        actual_db_path = db_path / "chunks.db"
        actual_db_path.parent.mkdir(parents=True, exist_ok=True)
        actual_db_path.write_text("small content")

        db = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        config = DatabaseConfig(max_disk_usage_mb=1024.0)  # 1 GB limit
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=tmp_path,
            config=config
        )

        # File is much smaller than 1GB, should not raise
        coord._check_disk_usage_limit()

    def test_check_disk_usage_limit_exceeded(self, tmp_path):
        """Test that error is returned when limit is exceeded."""
        db_path = tmp_path / "test_db_dir"
        # Create the actual database file and use a limit of 0.0 to ensure it always exceeds
        actual_db_path = db_path / "chunks.db"
        actual_db_path.parent.mkdir(parents=True, exist_ok=True)
        actual_db_path.write_bytes(b"x" * 100)

        db = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        config = DatabaseConfig(max_disk_usage_mb=0.0)  # Zero limit - any file exceeds
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=tmp_path,
            config=config
        )

        error = coord._check_disk_usage_limit()
        assert error is not None
        assert isinstance(error, DiskUsageLimitExceededError)
        assert error.limit_mb == 0.0
        assert error.current_size_mb >= 0  # File has some size (may be 0)

    def test_check_disk_usage_limit_file_not_exists(self, tmp_path):
        """Test behavior when database file doesn't exist."""
        db_path = tmp_path / "nonexistent.db"

        db = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        config = DatabaseConfig(max_disk_usage_mb=1024.0)
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=tmp_path,
            config=config
        )

        # Should not raise exception for non-existent file
        coord._check_disk_usage_limit()

    def test_check_disk_usage_limit_stat_error(self, tmp_path, monkeypatch):
        """Test handling of stat errors gracefully."""
        db_path = tmp_path / "test.db"
        # Create the actual database file
        actual_db_path = db_path / "chunks.db"
        actual_db_path.parent.mkdir(parents=True, exist_ok=True)
        actual_db_path.write_text("content")

        db = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        config = DatabaseConfig(max_disk_usage_mb=1024.0)
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=tmp_path,
            config=config
        )

        # Mock stat to raise OSError (after provider initialization)
        def mock_stat(*args, **kwargs):
            raise OSError("Permission denied")

        monkeypatch.setattr("pathlib.Path.stat", mock_stat)

        # Should not raise DiskUsageLimitExceededError, just log warning
        coord._check_disk_usage_limit()

    def test_check_disk_usage_limit_exact_boundary(self, tmp_path):
        """Test that exact limit size triggers exceeded (uses >= comparison)."""
        db_path = tmp_path / "test.db"
        # Create the actual database file with exactly 1MB content
        actual_db_path = db_path / "chunks.db"
        actual_db_path.parent.mkdir(parents=True, exist_ok=True)
        # Write exactly 1MB (1024*1024 bytes)
        actual_db_path.write_bytes(b"x" * (1024 * 1024))

        db = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        config = DatabaseConfig(max_disk_usage_mb=1.0)  # 1MB limit
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=tmp_path,
            config=config
        )

        error = coord._check_disk_usage_limit()
        assert error is not None
        assert isinstance(error, DiskUsageLimitExceededError)
        assert error.limit_mb == 1.0
        assert abs(error.current_size_mb - 1.0) < 0.001  # Should be very close to 1.0

    def test_check_disk_usage_limit_directory_size(self, tmp_path):
        """Test directory size calculation for directory-based providers."""
        db_path = tmp_path / "test.db"
        # Create directory structure with multiple files
        db_path.mkdir(parents=True, exist_ok=True)

        # Create main database file (1MB)
        (db_path / "chunks.db").write_bytes(b"x" * (1024 * 1024))

        # Create additional files (0.5MB each)
        (db_path / "chunks.db.wal").write_bytes(b"y" * (512 * 1024))
        (db_path / "chunks.db.tmp").write_bytes(b"z" * (512 * 1024))

        # Create subdirectory with file (0.25MB)
        subdir = db_path / "subdir"
        subdir.mkdir()
        (subdir / "extra.db").write_bytes(b"w" * (256 * 1024))

        db = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        config = DatabaseConfig(max_disk_usage_mb=2.0)  # 2MB limit
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=tmp_path,
            config=config
        )

        error = coord._check_disk_usage_limit()
        assert error is not None
        assert isinstance(error, DiskUsageLimitExceededError)
        assert error.limit_mb == 2.0
        # Total should be 1 + 0.5 + 0.5 + 0.25 = 2.25MB
        assert error.current_size_mb > 2.0

    def test_check_disk_usage_limit_directory_size_calculation(self, tmp_path):
        """Test that directory size calculation works correctly for database directories."""
        # Use a local working directory instead of tmp to avoid test failures caused by not having OS privileges to create symlink
        test_dir = Path.cwd() / "test_disk_size_db"
        test_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Create database directory with files
            db_path = test_dir / "test.db"
            db_path.mkdir(parents=True, exist_ok=True)
            (db_path / "chunks.db").write_bytes(b"x" * (512 * 1024))  # 0.5MB
            (db_path / "chunks.db.wal").write_bytes(b"y" * (256 * 1024))  # 0.25MB

            db = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
            config = DatabaseConfig(max_disk_usage_mb=1.0)  # 1MB limit
            coord = IndexingCoordinator(
                database_provider=db,
                base_directory=tmp_path,
                config=config
            )

            # Should work normally - calculates size of all files in directory
            coord._check_disk_usage_limit()
        finally:
            # Clean up test directory
            import shutil
            if test_dir.exists():
                shutil.rmtree(test_dir)

    def test_check_disk_usage_limit_filesystem_errors(self, tmp_path, monkeypatch):
        """Test handling of various filesystem errors."""
        db_path = tmp_path / "test.db"
        db_path.mkdir(parents=True)
        (db_path / "chunks.db").write_bytes(b"x" * 100)

        db = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        config = DatabaseConfig(max_disk_usage_mb=1024.0)
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=tmp_path,
            config=config
        )

        # Mock rglob to raise PermissionError
        def mock_rglob(*args, **kwargs):
            raise PermissionError("Access denied")

        monkeypatch.setattr("pathlib.Path.rglob", mock_rglob)

        # Should not raise exception, just log warning and continue
        coord._check_disk_usage_limit()


class TestIndexingCoordinatorIntegration:
    """Test integration of disk usage limiting with processing."""

    def test_process_directory_disk_limit_exceeded(self, tmp_path):
        """Test that process_directory returns disk_limit_exceeded status."""
        db_path = tmp_path / "test.db"

        db = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        # Connect to initialize the database with proper schema
        db.connect()
        config = DatabaseConfig(max_disk_usage_mb=0.0)  # Zero limit
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=tmp_path,
            config=config
        )

        # Create a test file to process
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        import asyncio
        result = asyncio.run(coord.process_directory(tmp_path, patterns=["*.py"]))

        assert result["status"] == "disk_limit_exceeded"
        assert "current_size_mb" in result
        assert "limit_mb" in result
        assert result["limit_mb"] == 0.0

    def test_store_parsed_results_disk_limit_exceeded(self, tmp_path):
        """Test that _store_parsed_results returns disk limit error when limit exceeded."""
        db_path = tmp_path / "test.db"

        db = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        # Connect to initialize the database with proper schema
        db.connect()
        config = DatabaseConfig(max_disk_usage_mb=0.0)  # Zero limit
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=tmp_path,
            config=config
        )

        # Create a parsed result
        test_file = tmp_path / "test.py"
        result = _create_parsed_file_result(test_file, [])

        import asyncio
        stats = asyncio.run(coord._store_parsed_results([result]))

        # Check that disk limit error is in the errors
        assert len(stats["errors"]) == 1
        error = stats["errors"][0]
        assert error.get("disk_limit_exceeded") is True
        assert error["limit_mb"] == 0.0
        assert error["current_size_mb"] >= 0

    def test_process_directory_normal_operation(self, tmp_path):
        """Test that processing works normally when under disk limit."""
        db_path = tmp_path / "test.db"
        # Create the actual database file with small content
        actual_db_path = db_path / "chunks.db"
        actual_db_path.parent.mkdir(parents=True, exist_ok=True)
        actual_db_path.write_text("small")

        db = DuckDBProvider(db_path=db_path, base_directory=tmp_path)
        config = DatabaseConfig(max_disk_usage_mb=1024.0)  # 1GB limit
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=tmp_path,
            config=config
        )

        # Create a test file to process
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        import asyncio
        result = asyncio.run(coord.process_directory(tmp_path, patterns=["*.py"]))

        # Should succeed normally
        assert result["status"] == "success"
        assert "files_processed" in result