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
import copy
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

        # Background tasks
        self._deferred_start_task: asyncio.Task | None = None
        self._realtime_start_task: asyncio.Task | None = None
        self._scan_task: asyncio.Task | None = None
        self._scan_lock = asyncio.Lock()
        self._scan_target_path: Path | None = None
        self._startup_failure_message: str | None = None

        # Scan progress tracking
        self._scan_complete = False
        self._scan_progress = {
            "files_processed": 0,
            "chunks_created": 0,
            "is_scanning": False,
            "scan_started_at": None,
            "scan_completed_at": None,
            "realtime": RealtimeIndexingService.health_snapshot_for_config(config),
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
            self._startup_failure_message = None

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
                        "Embedding provider registered: "
                        f"{self.config.embedding.provider}"
                    )
            except ValueError as e:
                # API key or configuration issue - expected for search-only usage
                self.debug_log(f"Embedding provider setup skipped: {e}")
            except Exception as e:
                # Unexpected error - log but continue
                self.debug_log(f"Unexpected error setting up embedding provider: {e}")

            # Initialize LLM manager with dual providers
            # (optional - continue if it fails)
            try:
                if self.config.llm:
                    utility_config, synthesis_config = (
                        self.config.llm.get_provider_configs()
                    )
                    self.llm_manager = LLMManager(utility_config, synthesis_config)
                    self.debug_log(
                        f"LLM providers registered: {self.config.llm.provider} "
                        f"(utility: {utility_config['model']}, "
                        f"synthesis: {synthesis_config['model']})"
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
            self._scan_target_path = target_path.resolve()

            # Mark as initialized immediately (tools available)
            self._initialized = True
            self.debug_log("Service initialization complete")

            # Defer DB connect + realtime start to background so initialize is fast
            self._deferred_start_task = asyncio.create_task(
                self._deferred_connect_and_start(self._scan_target_path)
            )

    def _configured_realtime_backend(self) -> str | None:
        """Return the configured realtime backend when it is explicitly supported."""
        try:
            backend = getattr(
                getattr(self.config, "indexing", None), "realtime_backend", None
            )
        except Exception:
            return None
        if backend in {"watchman", "watchdog", "polling"}:
            return str(backend)
        return None

    def requires_strict_startup_barrier(self) -> bool:
        """Return whether daemon startup must block on realtime readiness."""
        return self._configured_realtime_backend() == "watchman"

    def _set_startup_failure(self, message: str) -> None:
        """Persist a startup failure for later fail-fast barrier checks."""
        self._startup_failure_message = message
        self._record_realtime_failure(message)

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
                self.services,
                self.config,
                debug_sink=self.debug_log,
                status_callback=self._update_realtime_status,
                resync_callback=self._request_realtime_resync,
            )
            monitoring_task = asyncio.create_task(
                self.realtime_indexing.start(target_path)
            )
            self._realtime_start_task = monitoring_task
            monitoring_task.add_done_callback(self._handle_realtime_start_task_done)
            # Schedule background scan AFTER monitoring is confirmed ready
            self._scan_task = asyncio.create_task(
                self._coordinated_initial_scan(target_path, monitoring_task)
            )
        except Exception as e:
            self.debug_log(f"Deferred connect/start failed: {e}")
            self._set_startup_failure(f"Deferred connect/start failed: {e}")

    async def await_startup_barrier(self) -> None:
        """Block daemon exposure until strict realtime startup requirements pass."""
        if not self.requires_strict_startup_barrier():
            return

        if self._deferred_start_task is None:
            raise RuntimeError(
                "Watchman startup barrier requested before deferred startup began"
            )

        await asyncio.shield(self._deferred_start_task)
        if self._startup_failure_message is not None:
            raise RuntimeError(self._startup_failure_message)

        if self._realtime_start_task is None:
            message = (
                "Watchman startup barrier requested but realtime startup task "
                "was never created"
            )
            self._set_startup_failure(message)
            raise RuntimeError(message)

        try:
            await asyncio.shield(self._realtime_start_task)
        except asyncio.CancelledError as error:
            message = "Watchman realtime startup was cancelled before readiness"
            self._set_startup_failure(message)
            raise RuntimeError(message) from error
        except Exception as error:
            message = self._startup_failure_message or str(error)
            self._set_startup_failure(message)
            raise RuntimeError(message) from error

        if self._startup_failure_message is not None:
            raise RuntimeError(self._startup_failure_message)

        if (
            self.realtime_indexing is None
            or not self.realtime_indexing.monitoring_ready.is_set()
        ):
            message = "Watchman startup finished without monitoring readiness"
            self._set_startup_failure(message)
            raise RuntimeError(message)

    async def _coordinated_initial_scan(
        self, target_path: Path, monitoring_task: asyncio.Task
    ) -> None:
        """Perform initial scan after monitoring is confirmed ready."""
        ready_task = asyncio.create_task(self.realtime_indexing.monitoring_ready.wait())
        try:
            timeout = (
                self.realtime_indexing._MONITORING_READY_TIMEOUT_SECONDS
                if self.realtime_indexing
                else 10.0
            )
            done, pending = await asyncio.wait(
                {ready_task, monitoring_task},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if ready_task in done:
                self.debug_log("Monitoring confirmed ready, starting initial scan")
                # Add small delay to ensure any startup files are captured
                # by monitoring.
                await asyncio.sleep(1.0)
            elif monitoring_task in done:
                try:
                    monitoring_task.result()
                except Exception as e:
                    self.debug_log(
                        f"Realtime startup failed before monitoring readiness: {e}"
                    )
                    self._record_realtime_failure(f"Realtime startup failed: {e}")
                    if self.requires_strict_startup_barrier():
                        self.debug_log(
                            "Strict realtime startup barrier failed; "
                            "skipping initial scan"
                        )
                        return
                else:
                    self.debug_log(
                        "Realtime startup completed without monitoring readiness; "
                        "proceeding with initial scan"
                    )
            else:
                self.debug_log(
                    "Monitoring setup timeout - proceeding with initial scan anyway"
                )
                for task in pending:
                    if task is not monitoring_task:
                        task.cancel()

            await self._run_directory_scan(target_path, trigger="initial")
        finally:
            if not ready_task.done():
                ready_task.cancel()
                try:
                    await ready_task
                except asyncio.CancelledError:
                    pass

    def _update_realtime_status(self, status: dict[str, Any]) -> None:
        """Persist the latest realtime snapshot for daemon status surfaces."""
        self._scan_progress["realtime"] = copy.deepcopy(status)

    def _record_realtime_failure(self, message: str) -> None:
        """Persist a startup failure into the shared realtime status snapshot."""
        realtime = copy.deepcopy(
            self._scan_progress.get("realtime")
            or RealtimeIndexingService.health_snapshot_for_config(self.config)
        )
        realtime["service_state"] = "degraded"
        realtime["last_error"] = message
        realtime["last_error_at"] = datetime.now().isoformat()
        self._scan_progress["realtime"] = realtime

    def _handle_realtime_start_task_done(self, task: asyncio.Task) -> None:
        """Capture realtime startup task failures so they are never silent."""
        if task.cancelled():
            return

        try:
            exc = task.exception()
        except Exception as error:
            self.debug_log(f"Failed to inspect realtime startup task: {error}")
            return

        if exc is None:
            return

        self.debug_log(f"Realtime startup task failed: {exc}")
        self._set_startup_failure(f"Realtime startup task failed: {exc}")

    async def _request_realtime_resync(
        self, reason: str, details: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Run a serialized reconciliation scan and return the embed result."""
        if not self._scan_target_path:
            raise RuntimeError("Realtime resync requested before target path resolved")

        detail_suffix = f" details={details}" if details else ""
        self.debug_log(f"Realtime resync requested: reason={reason}{detail_suffix}")
        await self._run_directory_scan(
            self._scan_target_path,
            trigger="realtime_resync",
            reason=reason,
            no_embeddings=True,
        )
        if self.services is None:
            return None

        exclude_patterns = list(getattr(self.config.indexing, "exclude", []) or [])
        embed_result = await (
            self.services.indexing_coordinator.generate_missing_embeddings(
                exclude_patterns=exclude_patterns
            )
        )
        generated = embed_result.get("generated", 0)
        self.debug_log(
            "Realtime resync embedding follow-up completed: "
            f"status={embed_result.get('status')} generated={generated}"
        )
        return embed_result

    async def _run_directory_scan(
        self,
        target_path: Path,
        trigger: str,
        reason: str | None = None,
        no_embeddings: bool = False,
    ) -> None:
        """Perform an initial or reconciliation scan without overlapping other scans."""
        async with self._scan_lock:
            try:
                self._scan_progress["is_scanning"] = True
                self._scan_progress["scan_started_at"] = datetime.now().isoformat()
                self._scan_progress["scan_error"] = None
                self.debug_log(
                    f"Starting {trigger} directory scan"
                    + (f" ({reason})" if reason else "")
                )

                # Progress callback to update scan state
                def progress_callback(message: str):
                    # Parse progress messages to update counters
                    if "files processed" in message:
                        # Extract numbers from progress messages
                        import re

                        match = re.search(
                            r"(\d+) files processed.*?(\d+) chunks", message
                        )
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
                    target_path, no_embeddings=no_embeddings
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
                    f"{trigger} scan completed: "
                    f"{stats.files_processed} files, {stats.chunks_created} chunks"
                )

            except Exception as e:
                self.debug_log(f"{trigger} scan failed: {e}")
                self._scan_progress["is_scanning"] = False
                self._scan_progress["scan_error"] = str(e)
                raise

    async def cleanup(self) -> None:
        """Clean up resources and close database connection.

        This method is idempotent - safe to call multiple times.
        """
        if (
            self._deferred_start_task is not None
            and not self._deferred_start_task.done()
        ):
            self.debug_log("Cancelling deferred realtime startup task")
            self._deferred_start_task.cancel()
            try:
                await self._deferred_start_task
            except asyncio.CancelledError:
                pass

        if (
            self._realtime_start_task is not None
            and not self._realtime_start_task.done()
        ):
            self.debug_log("Cancelling realtime start task")
            self._realtime_start_task.cancel()
            try:
                await self._realtime_start_task
            except asyncio.CancelledError:
                pass

        # Cancel background scan task if still running
        if self._scan_task is not None and not self._scan_task.done():
            self.debug_log("Cancelling background scan task")
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass

        # Stop real-time indexing
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
                "in .chunkhound.json or set "
                "CHUNKHOUND_EMBEDDING__API_KEY environment variable."
            )
        return self.embedding_manager

    def _build_filtered_tool_dicts(self) -> list[dict[str, Any]]:
        """Build a JSON-serialisable list of available tool schemas.

        Filters tools based on embedding/LLM/reranker availability and
        dynamically restricts schema enums when capabilities are unavailable.

        Returns:
            List of dicts with keys ``name``, ``description``, ``inputSchema``.
        """
        from .common import has_reranker_support
        from .tools import TOOL_REGISTRY

        tools = []
        for tool_name, tool in TOOL_REGISTRY.items():
            if tool.requires_embeddings and (
                not self.embedding_manager
                or not self.embedding_manager.list_providers()
            ):
                continue
            if tool.requires_llm and not self.llm_manager:
                continue
            if tool.requires_reranker and not has_reranker_support(
                self.embedding_manager
            ):
                continue

            tool_params = copy.deepcopy(tool.parameters)

            if tool_name == "search" and (
                not self.embedding_manager
                or not self.embedding_manager.list_providers()
            ):
                if "type" in tool_params.get("properties", {}):
                    tool_params["properties"]["type"]["enum"] = ["regex"]

            tools.append(
                {
                    "name": tool_name,
                    "description": tool.description,
                    "inputSchema": tool_params,
                }
            )
        return tools

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
