"""Database configuration for ChunkHound.

This module provides database-specific configuration with support for
multiple database providers and storage backends.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class DatabaseConfig(BaseModel):
    """Database configuration with support for multiple providers.

    Configuration can be provided via:
    - Environment variables (CHUNKHOUND_DATABASE_*)
    - Configuration files
    - CLI arguments
    - Default values
    """

    # Database location
    path: Path | None = Field(default=None, description="Path to database directory")

    # Provider selection
    provider: Literal["duckdb", "lancedb"] = Field(
        default="duckdb", description="Database provider to use"
    )

    # LanceDB-specific settings
    lancedb_index_type: Literal["auto", "ivf_hnsw_sq", "ivf_rq"] | None = Field(
        default=None,
        description="LanceDB vector index type: auto (default), ivf_hnsw_sq, or ivf_rq (requires 0.25.3+)",
    )

    lancedb_optimize_fragment_threshold: int = Field(
        default=100,
        ge=0,
        description="Minimum fragment count to trigger optimization (0 = always optimize, 50 = aggressive, 100 = balanced, 500 = conservative)",
    )

    # Disk usage limits
    max_disk_usage_mb: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum database size in MB before indexing is stopped (None = no limit)",
    )

    # Retry configuration
    retry_on_timeout: bool = Field(
        default=True,
        description="Enable automatic retry for database timeouts"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts for timed-out operations"
    )
    retry_backoff_seconds: float = Field(
        default=1.0,
        ge=0.0,
        description="Base backoff time between retries (exponential)"
    )

    # LanceDB-specific optimization
    lancedb_optimize_during_indexing: bool = Field(
        default=True,
        description="Run optimization during indexing to prevent fragmentation"
    )
    lancedb_indexing_fragment_threshold: int = Field(
        default=25,
        ge=1,
        description="Fragment count threshold for optimization during indexing"
    )

    @field_validator("path")
    def validate_path(cls, v: Path | None) -> Path | None:
        """Convert string paths to Path objects."""
        if v is not None and not isinstance(v, Path):
            return Path(v)
        return v

    @field_validator("provider")
    def validate_provider(cls, v: str) -> str:
        """Validate database provider selection."""
        valid_providers = ["duckdb", "lancedb"]
        if v not in valid_providers:
            raise ValueError(f"Invalid provider: {v}. Must be one of {valid_providers}")
        return v

    def get_db_path(self) -> Path:
        """Get the actual database location for the configured provider.

        Returns the final path used by the provider, including all
        provider-specific transformations:
        - DuckDB: path/chunks.db (file) or :memory: for in-memory
        - LanceDB: path/lancedb.lancedb/ (directory with .lancedb suffix)

        This is the authoritative source for database location checks.
        """
        if self.path is None:
            raise ValueError("Database path not configured")

        # Skip directory creation for in-memory databases (":memory:" is invalid on Windows)
        is_memory = str(self.path) == ":memory:"
        if not is_memory:
            self.path.mkdir(parents=True, exist_ok=True)

        if self.provider == "duckdb":
            return self.path if is_memory else self.path / "chunks.db"
        elif self.provider == "lancedb":
            # LanceDB adds .lancedb suffix to prevent naming collisions
            # and clarify storage structure (see lancedb_provider.py:111-113)
            lancedb_base = self.path / "lancedb"
            return lancedb_base.parent / f"{lancedb_base.stem}.lancedb"
        else:
            raise ValueError(f"Unknown database provider: {self.provider}")

    def is_configured(self) -> bool:
        """Check if database is properly configured."""
        return self.path is not None

    @classmethod
    def add_cli_arguments(
        cls, parser: argparse.ArgumentParser, required_path: bool = False
    ) -> None:
        """Add database-related CLI arguments."""
        parser.add_argument(
            "--db",
            "--database-path",
            type=Path,
            help="Database file path (default: from config file or .chunkhound.db)",
            required=required_path,
        )

        parser.add_argument(
            "--database-provider",
            choices=["duckdb", "lancedb"],
            help="Database provider to use",
        )

        parser.add_argument(
            "--max-disk-usage-gb",
            type=float,
            help="Maximum database size in GB before indexing is stopped",
        )

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        """Load database config from environment variables."""
        config = {}
        # Support both new and legacy env var names
        if db_path := (
            os.getenv("CHUNKHOUND_DATABASE__PATH") or os.getenv("CHUNKHOUND_DB_PATH")
        ):
            config["path"] = Path(db_path)
        if provider := os.getenv("CHUNKHOUND_DATABASE__PROVIDER"):
            config["provider"] = provider
        if index_type := os.getenv("CHUNKHOUND_DATABASE__LANCEDB_INDEX_TYPE"):
            config["lancedb_index_type"] = index_type
        if threshold := os.getenv("CHUNKHOUND_DATABASE__LANCEDB_OPTIMIZE_FRAGMENT_THRESHOLD"):
            config["lancedb_optimize_fragment_threshold"] = int(threshold)
        # Disk usage limit from environment
        if max_disk_gb := os.getenv("CHUNKHOUND_DATABASE__MAX_DISK_USAGE_GB"):
            try:
                config["max_disk_usage_mb"] = float(max_disk_gb) * 1024.0
            except ValueError:
                # Invalid value - silently ignore
                pass

        # Retry configuration
        if retry_on_timeout := os.getenv("CHUNKHOUND_DATABASE__RETRY_ON_TIMEOUT"):
            config["retry_on_timeout"] = retry_on_timeout.lower() in ("true", "1", "yes")
        if max_retries := os.getenv("CHUNKHOUND_DATABASE__MAX_RETRIES"):
            try:
                config["max_retries"] = int(max_retries)
            except ValueError:
                pass
        if retry_backoff := os.getenv("CHUNKHOUND_DATABASE__RETRY_BACKOFF_SECONDS"):
            try:
                config["retry_backoff_seconds"] = float(retry_backoff)
            except ValueError:
                pass

        # LanceDB-specific optimization
        if optimize_during_indexing := os.getenv("CHUNKHOUND_DATABASE__LANCEDB_OPTIMIZE_DURING_INDEXING"):
            config["lancedb_optimize_during_indexing"] = optimize_during_indexing.lower() in ("true", "1", "yes")
        if fragment_threshold := os.getenv("CHUNKHOUND_DATABASE__LANCEDB_INDEXING_FRAGMENT_THRESHOLD"):
            try:
                config["lancedb_indexing_fragment_threshold"] = int(fragment_threshold)
            except ValueError:
                pass

        return config

    @classmethod
    def extract_cli_overrides(cls, args: Any) -> dict[str, Any]:
        """Extract database config from CLI arguments."""
        overrides = {}
        if hasattr(args, "db") and args.db:
            overrides["path"] = args.db
        if hasattr(args, "database_path") and args.database_path:
            overrides["path"] = args.database_path
        if hasattr(args, "database_provider") and args.database_provider:
            overrides["provider"] = args.database_provider
        if hasattr(args, "max_disk_usage_gb") and args.max_disk_usage_gb is not None:
            overrides["max_disk_usage_mb"] = args.max_disk_usage_gb * 1024.0
        return overrides

    def __repr__(self) -> str:
        """String representation of database configuration."""
        parts = [f"provider={self.provider}", f"path={self.path}"]
        if self.max_disk_usage_mb is not None:
            parts.append(f"max_disk_usage_mb={self.max_disk_usage_mb}")
        return f"DatabaseConfig({', '.join(parts)})"
