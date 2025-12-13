"""Base class for MCP servers providing common initialization and lifecycle management.

This module provides a base class that handles:
- Service initialization (database, embeddings)
- Configuration validation
- Lifecycle management (startup/shutdown)
- Common error handling patterns

Architecture Note: MCP server (stdio-only) inherits from this base
to ensure consistent initialization while respecting protocol-specific constraints.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from chunkhound.core.config import EmbeddingProviderFactory
from chunkhound.core.config.config import Config
from chunkhound.database_factory import DatabaseServices, create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from chunkhound.services.realtime_indexing_service import RealtimeIndexingService


class MCPServerBase(ABC):
    """Base class for MCP server implementations.

    Provides common initialization, configuration validation, and lifecycle
    management for stdio MCP server.

    Subclasses must implement:
    - _register_tools(): Register protocol-specific tool handlers
    - run(): Main server execution loop
    """

    def __init__(self, config: Config, debug_mode: bool = False, args: Any = None):
        """Initialize base MCP server.

        Args:
            config: Validated configuration object
            debug_mode: Enable debug logging to stderr
            args: Original CLI arguments for direct path access
        """
        self.config = config
        self.args = args  # Store original CLI args for direct path access
        self.debug_mode = debug_mode or os.getenv("CHUNKHOUND_DEBUG", "").lower() in (
            "true",
            "1",
            "yes",
        )

        # Service components - initialized lazily or eagerly based on subclass
        self.services: DatabaseServices | None = None
        self.embedding_manager: EmbeddingManager | None = None
        self.llm_manager: LLMManager | None = None
        self.realtime_indexing: RealtimeIndexingService | None = None

        # Initialization state
        self._initialized = False
        self._init_lock = asyncio.Lock()

        # Scan progress tracking
        self._scan_complete = False
        self._scan_progress = {
            "files_processed": 0,
            "chunks_created": 0,
            "is_scanning": False,
            "scan_started_at": None,
            "scan_completed_at": None,
        }

        # Set MCP mode to suppress stderr output that interferes with JSON-RPC
        os.environ["CHUNKHOUND_MCP_MODE"] = "1"

    def debug_log(self, message: str) -> None:
        """Log debug message to file if debug mode is enabled."""
        if self.debug_mode:
            # Write to debug file instead of stderr to preserve JSON-RPC protocol
            debug_file = os.getenv(
                "CHUNKHOUND_DEBUG_FILE", "/tmp/chunkhound_mcp_debug.log"
            )
            try:
                with open(debug_file, "a") as f:
                    from datetime import datetime

                    timestamp = datetime.now().isoformat()
                    f.write(f"[{timestamp}] [MCP] {message}\n")
                    f.flush()
            except Exception:
                # Silently fail if we can't write to debug file
                pass

    async def initialize(self) -> None:
        """Initialize services and database connection.

        This method is idempotent - safe to call multiple times.
        Uses locking to ensure thread-safe initialization.

        Raises:
            ValueError: If required configuration is missing
            Exception: If services fail to initialize
        """
        async with self._init_lock:
            if self._initialized:
                return

            self.debug_log("Starting service initialization")

            # Validate database configuration
            if not self.config.database or not self.config.database.path:
                raise ValueError("Database configuration not initialized")

            db_path = Path(self.config.database.path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Initialize embedding manager
            self.embedding_manager = EmbeddingManager()

            # Setup embedding provider (optional - continue if it fails)
            try:
                if self.config.embedding:
                    provider = EmbeddingProviderFactory.create_provider(
                        self.config.embedding
                    )
                    self.embedding_manager.register_provider(provider, set_default=True)
                    self.debug_log(
                        f"Embedding provider registered: {self.config.embedding.provider}"
                    )
            except ValueError as e:
                # API key or configuration issue - expected for search-only usage
                self.debug_log(f"Embedding provider setup skipped: {e}")
            except Exception as e:
                # Unexpected error - log but continue
                self.debug_log(f"Unexpected error setting up embedding provider: {e}")

            # Initialize LLM manager with dual providers (optional - continue if it fails)
            try:
                if self.config.llm:
                    utility_config, synthesis_config = self.config.llm.get_provider_configs()
                    self.llm_manager = LLMManager(utility_config, synthesis_config)
                    self.debug_log(
                        f"LLM providers registered: {self.config.llm.provider} "
                        f"(utility: {utility_config['model']}, synthesis: {synthesis_config['model']})"
                    )
            except ValueError as e:
                # API key or configuration issue - expected if LLM not needed
                self.debug_log(f"LLM provider setup skipped: {e}")
            except Exception as e:
                # Unexpected error - log but continue
                self.debug_log(f"Unexpected error setting up LLM provider: {e}")

            # Create services using unified factory (lazy connect for fast init)
            self.services = create_services(
                db_path=db_path,
                config=self.config,
                embedding_manager=self.embedding_manager,
            )

            # Determine target path for scanning and watching
            if self.args and hasattr(self.args, "path"):
                target_path = Path(self.args.path)
                self.debug_log(f"Using direct path from args: {target_path}")
            else:
                # Fallback to config resolution (shouldn't happen in normal usage)
                target_path = self.config.target_dir or db_path.parent.parent
                self.debug_log(f"Using fallback path resolution: {target_path}")

            # Mark as initialized immediately (tools available)
            self._initialized = True
            self.debug_log("Service initialization complete")

            # Defer DB connect + realtime start to background so initialize is fast
            asyncio.create_task(self._deferred_connect_and_start(target_path))

    async def _deferred_connect_and_start(self, target_path: Path) -> None:
        """Connect DB and start realtime monitoring in background."""
        try:
            # Ensure services exist
            if not self.services:
                return
            # Connect to database lazily
            if not self.services.provider.is_connected:
                self.services.provider.connect()

            # Start real-time indexing service
            self.debug_log("Starting real-time indexing service (deferred)")
            self.realtime_indexing = RealtimeIndexingService(
                self.services, self.config, debug_sink=self.debug_log
            )
            monitoring_task = asyncio.create_task(
                self.realtime_indexing.start(target_path)
            )
            # Schedule background scan AFTER monitoring is confirmed ready
            asyncio.create_task(
                self._coordinated_initial_scan(target_path, monitoring_task)
            )
        except Exception as e:
            self.debug_log(f"Deferred connect/start failed: {e}")

    async def _coordinated_initial_scan(
        self, target_path: Path, monitoring_task: asyncio.Task
    ) -> None:
        """Perform initial scan after monitoring is confirmed ready."""
        try:
            # Wait for monitoring to be ready (with timeout)
            await asyncio.wait_for(
                self.realtime_indexing.monitoring_ready.wait(), timeout=10.0
            )
            self.debug_log("Monitoring confirmed ready, starting initial scan")

            # Add small delay to ensure any startup files are captured by monitoring
            await asyncio.sleep(1.0)

            # Now perform the initial scan
            self._scan_progress["is_scanning"] = True
            self._scan_progress["scan_started_at"] = datetime.now().isoformat()
            await self._background_initial_scan(target_path)

        except asyncio.TimeoutError:
            self.debug_log(
                "Monitoring setup timeout - proceeding with initial scan anyway"
            )
            # Still do the scan even if monitoring isn't ready
            self._scan_progress["is_scanning"] = True
            self._scan_progress["scan_started_at"] = datetime.now().isoformat()
            await self._background_initial_scan(target_path)

    async def _background_initial_scan(self, target_path: Path) -> None:
        """Perform initial directory scan in background without blocking startup."""
        try:
            self.debug_log("Starting background initial directory scan")

            # Progress callback to update scan state
            def progress_callback(message: str):
                # Parse progress messages to update counters
                if "files processed" in message:
                    # Extract numbers from progress messages
                    import re

                    match = re.search(r"(\d+) files processed.*?(\d+) chunks", message)
                    if match:
                        self._scan_progress["files_processed"] = int(match.group(1))
                        self._scan_progress["chunks_created"] = int(match.group(2))
                self.debug_log(message)

            # Create indexing service for background scan
            indexing_service = DirectoryIndexingService(
                indexing_coordinator=self.services.indexing_coordinator,
                config=self.config,
                progress_callback=progress_callback,
            )

            # Perform scan with lower priority
            stats = await indexing_service.process_directory(
                target_path, no_embeddings=False
            )

            # Update final stats
            self._scan_progress.update(
                {
                    "files_processed": stats.files_processed,
                    "chunks_created": stats.chunks_created,
                    "is_scanning": False,
                    "scan_completed_at": datetime.now().isoformat(),
                }
            )
            self._scan_complete = True

            self.debug_log(
                f"Background scan completed: {stats.files_processed} files, {stats.chunks_created} chunks"
            )

        except Exception as e:
            self.debug_log(f"Background initial scan failed: {e}")
            self._scan_progress["is_scanning"] = False
            self._scan_progress["scan_error"] = str(e)

    async def cleanup(self) -> None:
        """Clean up resources and close database connection.

        This method is idempotent - safe to call multiple times.
        """
        # Stop real-time indexing first
        if self.realtime_indexing:
            self.debug_log("Stopping real-time indexing service")
            await self.realtime_indexing.stop()

        if self.services and self.services.provider.is_connected:
            self.debug_log("Closing database connection")
            # Use new close() method for proper cleanup, with fallback to disconnect()
            if hasattr(self.services.provider, "close"):
                self.services.provider.close()
            else:
                self.services.provider.disconnect()
            self._initialized = False

    def ensure_services(self) -> DatabaseServices:
        """Ensure services are initialized and return them.

        Returns:
            DatabaseServices instance

        Raises:
            RuntimeError: If services are not initialized
        """
        if not self.services:
            raise RuntimeError("Services not initialized. Call initialize() first.")

        # Ensure database connection is active
        if not self.services.provider.is_connected:
            self.services.provider.connect()

        return self.services

    def ensure_embedding_manager(self) -> EmbeddingManager:
        """Ensure embedding manager is available and has providers.

        Returns:
            EmbeddingManager instance

        Raises:
            RuntimeError: If no embedding providers are available
        """
        if not self.embedding_manager or not self.embedding_manager.list_providers():
            raise RuntimeError(
                "No embedding providers available. Configure an embedding provider "
                "in .chunkhound.json or set CHUNKHOUND_EMBEDDING__API_KEY environment variable."
            )
        return self.embedding_manager

    @abstractmethod
    def _register_tools(self) -> None:
        """Register tools with the server implementation.

        Subclasses must implement this to register tools using their
        protocol-specific decorators/patterns.
        """
        pass

    @abstractmethod
    async def run(self) -> None:
        """Run the server.

        Subclasses must implement their protocol-specific server loop.
        """
        pass
