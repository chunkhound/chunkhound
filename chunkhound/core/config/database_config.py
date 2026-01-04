"""Database configuration for ChunkHound.

This module provides database-specific configuration with support for
multiple database providers and storage backends.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class MultiRepoConfig(BaseModel):
    """Multi-repository database configuration.

    Supports two modes:
    - per-repo: Each repository has its own .chunkhound/db (legacy mode)
    - global: All repositories indexed to single database at global_db_path
    """

    enabled: bool = Field(default=False, description="Enable multi-repository support")
    mode: Literal["per-repo", "global"] = Field(
        default="per-repo",
        description="Database mode: per-repo (default) or global (single DB)",
    )
    global_db_path: Path = Field(
        default_factory=lambda: Path.home() / ".chunkhound" / "global" / "db",
        description="Path to global database (used when mode=global)",
    )

    # Daemon configuration
    daemon_port: int = Field(
        default=5173,
        ge=1,
        le=65535,
        description="Port for HTTP daemon server",
    )
    daemon_host: str = Field(
        default="127.0.0.1",
        description="Host for HTTP daemon server (use 0.0.0.0 for all interfaces)",
    )

    # Watcher configuration
    watcher_debounce_seconds: float = Field(
        default=2.0,
        ge=0.1,
        le=60.0,
        description="Debounce delay in seconds for file watcher events",
    )
    watcher_max_pending_events: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum total pending events before force-processing oldest",
    )
    watcher_max_pending_per_project: int = Field(
        default=500,
        ge=10,
        le=10000,
        description="Maximum pending events per project before force-flush",
    )

    # Proxy configuration
    proxy_timeout_seconds: float = Field(
        default=120.0,
        ge=5.0,
        le=3600.0,
        description="Timeout in seconds for proxy client HTTP requests",
    )

    # Auto-start configuration
    auto_start_daemon: bool = Field(
        default=False,
        description="Auto-start daemon when stdio session starts in global mode",
    )

    @field_validator("global_db_path")
    def validate_global_db_path(cls, v: Path | None) -> Path:
        """Convert string paths to Path objects."""
        if v is not None and not isinstance(v, Path):
            return Path(v)
        return v or Path.home() / ".chunkhound" / "global" / "db"


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

    # Multi-repository support
    multi_repo: MultiRepoConfig = Field(
        default_factory=MultiRepoConfig,
        description="Multi-repository database configuration",
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

    def get_db_path(self, current_dir: Path | None = None) -> Path:
        """Get the actual database location for the configured provider and mode.

        Args:
            current_dir: Current directory for per-repo mode path resolution

        Returns the final path used by the provider, including all
        provider-specific transformations:
        - DuckDB: path/chunks.db (file) or :memory: for in-memory
        - LanceDB: path/lancedb.lancedb/ (directory with .lancedb suffix)

        This is the authoritative source for database location checks.

        Raises:
            ValueError: If database path not configured or invalid mode
        """
        # Determine which path to use based on multi-repo mode
        if self.multi_repo.enabled and self.multi_repo.mode == "global":
            # Global mode: use global_db_path
            db_dir = self.multi_repo.global_db_path
        else:
            # Per-repo mode (legacy): use configured path or derive from current_dir
            if self.path is None:
                if current_dir is None:
                    raise ValueError(
                        "Database path not configured and current_dir not provided"
                    )
                db_dir = current_dir / ".chunkhound" / "db"
            else:
                db_dir = self.path

        # Skip directory creation for in-memory databases (":memory:" is invalid on Windows)
        is_memory = str(db_dir) == ":memory:"

        # Backwards-compatible handling:
        # - Older ChunkHound versions used `database.path` as the direct DuckDB
        #   file location (for example, `.chunkhound/db` as a file).
        # - Newer versions treat `database.path` as a directory and store the
        #   DuckDB file as `path/chunks.db`.
        #
        # When the configured path already exists as a file, we treat it as a
        # legacy DuckDB database file and return it directly instead of trying
        # to create a directory at that location.
        if self.provider == "duckdb" and not is_memory:
            if db_dir.exists() and db_dir.is_file():
                return db_dir

        if not is_memory:
            # For directory-style layouts, ensure the base path exists.
            db_dir.mkdir(parents=True, exist_ok=True)

        # Return provider-specific path
        if self.provider == "duckdb":
            return db_dir if is_memory else db_dir / "chunks.db"
        elif self.provider == "lancedb":
            # LanceDB adds .lancedb suffix to prevent naming collisions
            # and clarify storage structure (see lancedb_provider.py:111-113)
            lancedb_base = db_dir / "lancedb"
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

        # Database path and provider
        if db_path := (
            os.getenv("CHUNKHOUND_DATABASE__PATH") or os.getenv("CHUNKHOUND_DB_PATH")
        ):
            config["path"] = Path(db_path)
        if provider := os.getenv("CHUNKHOUND_DATABASE__PROVIDER"):
            config["provider"] = provider
        if index_type := os.getenv("CHUNKHOUND_DATABASE__LANCEDB_INDEX_TYPE"):
            config["lancedb_index_type"] = index_type
        if threshold := os.getenv(
            "CHUNKHOUND_DATABASE__LANCEDB_OPTIMIZE_FRAGMENT_THRESHOLD"
        ):
            config["lancedb_optimize_fragment_threshold"] = int(threshold)
        # Disk usage limit from environment
        if max_disk_gb := os.getenv("CHUNKHOUND_DATABASE__MAX_DISK_USAGE_GB"):
            try:
                config["max_disk_usage_mb"] = float(max_disk_gb) * 1024.0
            except ValueError:
                # Invalid value - silently ignore
                pass

        # Multi-repo configuration
        multi_repo_config = {}
        if enabled := os.getenv("CHUNKHOUND_DATABASE__MULTI_REPO__ENABLED"):
            multi_repo_config["enabled"] = enabled.lower() in ("true", "1", "yes")
        if mode := os.getenv("CHUNKHOUND_DATABASE__MULTI_REPO__MODE"):
            multi_repo_config["mode"] = mode
        if global_db_path := os.getenv(
            "CHUNKHOUND_DATABASE__MULTI_REPO__GLOBAL_DB_PATH"
        ):
            multi_repo_config["global_db_path"] = Path(global_db_path)

        # Daemon configuration
        if daemon_port := os.getenv("CHUNKHOUND_DATABASE__MULTI_REPO__DAEMON_PORT"):
            try:
                multi_repo_config["daemon_port"] = int(daemon_port)
            except ValueError:
                pass
        if daemon_host := os.getenv("CHUNKHOUND_DATABASE__MULTI_REPO__DAEMON_HOST"):
            multi_repo_config["daemon_host"] = daemon_host

        # Watcher configuration
        if debounce := os.getenv(
            "CHUNKHOUND_DATABASE__MULTI_REPO__WATCHER_DEBOUNCE_SECONDS"
        ):
            try:
                multi_repo_config["watcher_debounce_seconds"] = float(debounce)
            except ValueError:
                pass
        if max_events := os.getenv(
            "CHUNKHOUND_DATABASE__MULTI_REPO__WATCHER_MAX_PENDING_EVENTS"
        ):
            try:
                multi_repo_config["watcher_max_pending_events"] = int(max_events)
            except ValueError:
                pass
        if max_per_project := os.getenv(
            "CHUNKHOUND_DATABASE__MULTI_REPO__WATCHER_MAX_PENDING_PER_PROJECT"
        ):
            try:
                multi_repo_config["watcher_max_pending_per_project"] = int(
                    max_per_project
                )
            except ValueError:
                pass

        # Proxy configuration
        if timeout := os.getenv(
            "CHUNKHOUND_DATABASE__MULTI_REPO__PROXY_TIMEOUT_SECONDS"
        ):
            try:
                multi_repo_config["proxy_timeout_seconds"] = float(timeout)
            except ValueError:
                pass

        # Auto-start configuration
        if auto_start := os.getenv(
            "CHUNKHOUND_DATABASE__MULTI_REPO__AUTO_START_DAEMON"
        ):
            multi_repo_config["auto_start_daemon"] = auto_start.lower() in (
                "true",
                "1",
                "yes",
            )

        if multi_repo_config:
            config["multi_repo"] = multi_repo_config

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
