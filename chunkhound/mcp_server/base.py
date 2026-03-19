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
import sys
import time
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
from chunkhound.services.realtime_indexing_service import (
    RealtimeIndexingService,
    RealtimeStartupStatusTracker,
)


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
        self._startup_publish_complete = asyncio.Event()
        if self._realtime_startup_mode() != "daemon":
            self._startup_publish_complete.set()
        self._startup_tracker = RealtimeStartupStatusTracker(
            mode=self._realtime_startup_mode(),
            debug_sink=self._startup_log,
        )
        self._reset_warm_ready_tracking()

        # Scan progress tracking
        self._scan_complete = False
        self._scan_progress = {
            "files_processed": 0,
            "chunks_created": 0,
            "is_scanning": False,
            "scan_started_at": None,
            "scan_completed_at": None,
            "realtime": RealtimeIndexingService.health_snapshot_for_config(
                config,
                startup_mode=self._realtime_startup_mode(),
            ),
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
                    timestamp = datetime.now().isoformat()
                    f.write(f"[{timestamp}] [MCP] {message}\n")
                    f.flush()
            except Exception:
                # Silently fail if we can't write to debug file
                pass

    def _startup_log(self, message: str) -> None:
        """Emit startup breadcrumbs to debug logs and the daemon stderr log path."""
        self.debug_log(message)
        if self._realtime_startup_mode() != "daemon":
            return
        try:
            timestamp = datetime.now().isoformat()
            sys.stderr.write(f"[{timestamp}] [startup] {message}\n")
            sys.stderr.flush()
        except Exception:
            pass

    def _realtime_startup_mode(self) -> str:
        """Return the status mode for startup timing surfaces."""
        return "stdio"

    def _reset_warm_ready_tracking(self) -> None:
        self._warm_ready_started_monotonic: float | None = None
        self._warm_ready_summary_emitted = False
        self._warm_ready_initial_scan_total_seconds: float | None = None
        self._warm_ready_initial_scan_completed_monotonic: float | None = None
        self._warm_ready_initial_scan_skipped_monotonic: float | None = None
        self._warm_ready_fresh_instance_resync_requested = False
        self._warm_ready_fresh_instance_resync_total_seconds: float | None = None
        self._warm_ready_fresh_instance_resync_completed_monotonic: float | None = (
            None
        )
        self._warm_ready_fresh_instance_resync_outcome: str | None = None
        self._warm_ready_fresh_instance_resync_authoritative = False
        self._warm_ready_double_work_logged = False

    @staticmethod
    def _duration_seconds(
        start_monotonic: float | None,
        end_monotonic: float | None,
    ) -> float | None:
        if start_monotonic is None or end_monotonic is None:
            return None
        return round(max(end_monotonic - start_monotonic, 0.0), 3)

    @staticmethod
    def _format_duration(duration_seconds: float | None) -> str:
        if duration_seconds is None:
            return "n/a"
        return f"{duration_seconds:.3f}s"

    @staticmethod
    def _is_fresh_instance_resync(
        reason: str,
        details: dict[str, Any] | None,
    ) -> bool:
        return (
            reason == "realtime_loss_of_sync"
            and isinstance(details, dict)
            and details.get("loss_of_sync_reason") == "fresh_instance"
        )

    def _warm_ready_window_active(self) -> bool:
        return (
            self._warm_ready_started_monotonic is not None
            and not self._warm_ready_summary_emitted
        )

    @staticmethod
    def _is_authoritative_fresh_instance_resync_outcome(outcome: str | None) -> bool:
        return outcome in {"success", "complete", "up_to_date", "embeddings_disabled"}

    def _should_skip_initial_scan_for_startup_fresh_instance(self) -> bool:
        return (
            self._warm_ready_fresh_instance_resync_authoritative
            and self._warm_ready_fresh_instance_resync_completed_monotonic is not None
        )

    def _timing_log(self, message: str, *, daemon_visible: bool = False) -> None:
        formatted = f"warm-ready: {message}"
        if daemon_visible:
            self._startup_log(formatted)
            return
        self.debug_log(formatted)

    def _emit_warm_ready_summary_if_ready(self) -> None:
        if self._warm_ready_summary_emitted:
            return
        if self._warm_ready_started_monotonic is None:
            return
        if (
            self._warm_ready_initial_scan_completed_monotonic is None
            and self._warm_ready_initial_scan_skipped_monotonic is None
        ):
            return
        if (
            self._warm_ready_fresh_instance_resync_requested
            and self._warm_ready_fresh_instance_resync_completed_monotonic is None
        ):
            return

        startup_snapshot = self._startup_tracker.snapshot()
        blocking_startup = startup_snapshot.get("total_duration_seconds")
        if not isinstance(blocking_startup, (int, float)):
            return

        last_adjacent_completed = self._warm_ready_initial_scan_completed_monotonic
        if self._warm_ready_initial_scan_skipped_monotonic is not None:
            if last_adjacent_completed is None:
                last_adjacent_completed = (
                    self._warm_ready_initial_scan_skipped_monotonic
                )
            else:
                last_adjacent_completed = max(
                    last_adjacent_completed,
                    self._warm_ready_initial_scan_skipped_monotonic,
                )
        if self._warm_ready_fresh_instance_resync_completed_monotonic is not None:
            if last_adjacent_completed is None:
                last_adjacent_completed = (
                    self._warm_ready_fresh_instance_resync_completed_monotonic
                )
            else:
                last_adjacent_completed = max(
                    last_adjacent_completed,
                    self._warm_ready_fresh_instance_resync_completed_monotonic,
                )
        if last_adjacent_completed is None:
            return
        warm_ready_total = self._duration_seconds(
            self._warm_ready_started_monotonic,
            last_adjacent_completed,
        )
        summary_parts = [
            f"blocking_startup={self._format_duration(float(blocking_startup))}",
            f"warm_ready={self._format_duration(warm_ready_total)}",
        ]
        if self._warm_ready_fresh_instance_resync_total_seconds is not None:
            summary_parts.append(
                "fresh_instance_resync="
                f"{self._format_duration(self._warm_ready_fresh_instance_resync_total_seconds)}"
            )
        if self._warm_ready_initial_scan_total_seconds is not None:
            summary_parts.append(
                "initial_scan="
                f"{self._format_duration(self._warm_ready_initial_scan_total_seconds)}"
            )
        self._timing_log(
            "summary " + " ".join(summary_parts),
            daemon_visible=self._realtime_startup_mode() == "daemon",
        )
        self._warm_ready_summary_emitted = True

    def _default_realtime_scan_status(self) -> dict[str, Any]:
        return RealtimeIndexingService.health_snapshot_for_config(
            self.config,
            startup_mode=self._realtime_startup_mode(),
        )

    def _sync_startup_snapshot(self) -> None:
        realtime = copy.deepcopy(
            self._scan_progress.get("realtime") or self._default_realtime_scan_status()
        )
        realtime["startup"] = self._startup_tracker.snapshot()
        self._scan_progress["realtime"] = realtime

    def _start_startup_phase(self, phase_name: str) -> None:
        self._startup_tracker.start_phase(phase_name)
        self._sync_startup_snapshot()

    def _complete_startup_phase(self, phase_name: str) -> None:
        self._startup_tracker.complete_phase(phase_name)
        self._sync_startup_snapshot()

    def _fail_startup(self, message: str, *, phase_name: str | None = None) -> None:
        self._startup_tracker.fail_startup(message, phase_name=phase_name)
        self._sync_startup_snapshot()

    def _complete_startup(self) -> None:
        self._startup_tracker.complete_startup()
        self._sync_startup_snapshot()

    def _mark_startup_exposure_ready(self) -> None:
        self._startup_tracker.mark_exposure_ready()
        self._sync_startup_snapshot()

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
            self._startup_tracker.reset(self._realtime_startup_mode())
            self._reset_warm_ready_tracking()
            self._warm_ready_started_monotonic = time.monotonic()
            self._scan_progress["realtime"] = self._default_realtime_scan_status()
            if self._realtime_startup_mode() == "daemon":
                self._startup_publish_complete.clear()
            else:
                self._startup_publish_complete.set()
            self._start_startup_phase("initialize")

            try:
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
                        self.embedding_manager.register_provider(
                            provider,
                            set_default=True,
                        )
                        self.debug_log(
                            "Embedding provider registered: "
                            f"{self.config.embedding.provider}"
                        )
                except ValueError as e:
                    # API key or configuration issue - expected for search-only usage
                    self.debug_log(f"Embedding provider setup skipped: {e}")
                except Exception as e:
                    # Unexpected error - log but continue
                    self.debug_log(
                        f"Unexpected error setting up embedding provider: {e}"
                    )

                # Initialize LLM manager with dual providers
                # (optional - continue if it fails)
                try:
                    if self.config.llm:
                        utility_config, synthesis_config = (
                            self.config.llm.get_provider_configs()
                        )
                        self.llm_manager = LLMManager(
                            utility_config,
                            synthesis_config,
                        )
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
                self._complete_startup_phase("initialize")

                # Defer DB connect + realtime start to background so initialize is fast
                self._deferred_start_task = asyncio.create_task(
                    self._deferred_connect_and_start(self._scan_target_path)
                )
            except Exception as error:
                self._fail_startup(
                    f"Initialization failed: {error}",
                    phase_name="initialize",
                )
                raise

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

    def _set_startup_failure(
        self,
        message: str,
        *,
        phase_name: str | None = None,
    ) -> None:
        """Persist a startup failure for later fail-fast barrier checks."""
        self._startup_failure_message = message
        if phase_name is None:
            current_phase = self._startup_tracker.snapshot().get("current_phase")
            phase_name = current_phase if isinstance(current_phase, str) else None
        self._fail_startup(message, phase_name=phase_name)
        self._record_realtime_failure(message)

    async def _deferred_connect_and_start(self, target_path: Path) -> None:
        """Connect DB and start realtime monitoring in background."""
        try:
            # Ensure services exist
            if not self.services:
                return
            # Connect to database lazily
            self._start_startup_phase("db_connect")
            if not self.services.provider.is_connected:
                self.services.provider.connect()
            self._complete_startup_phase("db_connect")

            # Start real-time indexing service
            self.debug_log("Starting real-time indexing service (deferred)")
            self._start_startup_phase("realtime_start")
            self.realtime_indexing = RealtimeIndexingService(
                self.services,
                self.config,
                debug_sink=self.debug_log,
                startup_log_sink=self._startup_log,
                status_callback=self._update_realtime_status,
                resync_callback=self._request_realtime_resync,
                startup_tracker=self._startup_tracker,
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
            self._set_startup_failure(
                f"Deferred connect/start failed: {e}",
            )

    async def await_startup_barrier(self) -> None:
        """Block daemon exposure until strict realtime startup requirements pass."""
        self._start_startup_phase("startup_barrier")
        if not self.requires_strict_startup_barrier():
            self._complete_startup_phase("startup_barrier")
            return

        if self._deferred_start_task is None:
            message = "Watchman startup barrier requested before deferred startup began"
            self._set_startup_failure(message, phase_name="startup_barrier")
            raise RuntimeError(
                "Watchman startup barrier requested before deferred startup began"
            )

        await asyncio.shield(self._deferred_start_task)
        if self._startup_failure_message is not None:
            self._set_startup_failure(
                self._startup_failure_message,
                phase_name="startup_barrier",
            )
            raise RuntimeError(self._startup_failure_message)

        if self._realtime_start_task is None:
            message = (
                "Watchman startup barrier requested but realtime startup task "
                "was never created"
            )
            self._set_startup_failure(message, phase_name="startup_barrier")
            raise RuntimeError(message)

        try:
            await asyncio.shield(self._realtime_start_task)
        except asyncio.CancelledError as error:
            message = "Watchman realtime startup was cancelled before readiness"
            self._set_startup_failure(message, phase_name="startup_barrier")
            raise RuntimeError(message) from error
        except Exception as error:
            message = self._startup_failure_message or str(error)
            self._set_startup_failure(message, phase_name="startup_barrier")
            raise RuntimeError(message) from error

        if self._startup_failure_message is not None:
            self._set_startup_failure(
                self._startup_failure_message,
                phase_name="startup_barrier",
            )
            raise RuntimeError(self._startup_failure_message)

        if (
            self.realtime_indexing is None
            or not self.realtime_indexing.monitoring_ready.is_set()
        ):
            message = "Watchman startup finished without monitoring readiness"
            self._set_startup_failure(message, phase_name="startup_barrier")
            raise RuntimeError(message)
        self._complete_startup_phase("startup_barrier")

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
                daemon_visible = self._realtime_startup_mode() == "daemon"
                monitoring_ready_monotonic = time.monotonic()
                publish_wait_started_monotonic = monitoring_ready_monotonic
                if self._realtime_startup_mode() == "daemon":
                    if not self._startup_publish_complete.is_set():
                        self.debug_log(
                            "Monitoring confirmed ready; waiting for daemon publish "
                            "before initial scan"
                        )
                    await self._startup_publish_complete.wait()
                    if self._startup_failure_message is not None:
                        self.debug_log(
                            "Daemon publish never completed; skipping initial scan"
                        )
                        return
                publish_wait_duration = self._duration_seconds(
                    publish_wait_started_monotonic,
                    time.monotonic(),
                )
                self._timing_log(
                    "initial scan coordination "
                    f"monitoring_ready_to_publish={self._format_duration(publish_wait_duration)}",
                    daemon_visible=daemon_visible,
                )
                self.debug_log("Monitoring confirmed ready, coordinating initial scan")
                # Add small delay to ensure any startup files are captured
                # by monitoring.
                self._timing_log(
                    "initial scan coordination settle_sleep=1.000s",
                    daemon_visible=daemon_visible,
                )
                settle_sleep_started_monotonic = time.monotonic()
                await asyncio.sleep(1.0)
                settle_sleep_duration = self._duration_seconds(
                    settle_sleep_started_monotonic,
                    time.monotonic(),
                )
                self._timing_log(
                    "initial scan coordination "
                    f"settle_sleep_completed={self._format_duration(settle_sleep_duration)}",
                    daemon_visible=daemon_visible,
                )
                if self._should_skip_initial_scan_for_startup_fresh_instance():
                    self._warm_ready_initial_scan_skipped_monotonic = time.monotonic()
                    self._timing_log(
                        "startup reused successful fresh-instance reconciliation "
                        "and skipped the deferred initial scan",
                        daemon_visible=daemon_visible,
                    )
                    self._emit_warm_ready_summary_if_ready()
                    return
                if (
                    self._warm_ready_fresh_instance_resync_total_seconds is not None
                    and not self._warm_ready_double_work_logged
                ):
                    self._timing_log(
                        "startup paid both fresh-instance resync and initial scan "
                        "back-to-back",
                        daemon_visible=daemon_visible,
                    )
                    self._warm_ready_double_work_logged = True
                self.debug_log("Monitoring confirmed ready, starting initial scan")
                initial_scan_started_monotonic = time.monotonic()
                monitoring_ready_to_scan_start = self._duration_seconds(
                    monitoring_ready_monotonic,
                    initial_scan_started_monotonic,
                )
                self._timing_log(
                    "initial scan starting "
                    "monitoring_ready_to_scan_start="
                    f"{self._format_duration(monitoring_ready_to_scan_start)}",
                    daemon_visible=daemon_visible,
                )
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
            if ready_task in done:
                initial_scan_completed_monotonic = time.monotonic()
                initial_scan_total_duration = self._duration_seconds(
                    monitoring_ready_monotonic,
                    initial_scan_completed_monotonic,
                )
                self._timing_log(
                    "initial scan completed "
                    f"total={self._format_duration(initial_scan_total_duration)}",
                    daemon_visible=daemon_visible,
                )
                self._warm_ready_initial_scan_total_seconds = (
                    initial_scan_total_duration
                )
                self._warm_ready_initial_scan_completed_monotonic = (
                    initial_scan_completed_monotonic
                )
                self._emit_warm_ready_summary_if_ready()
        finally:
            if not ready_task.done():
                ready_task.cancel()
                try:
                    await ready_task
                except asyncio.CancelledError:
                    pass

    def _update_realtime_status(self, status: dict[str, Any]) -> None:
        """Persist the latest realtime snapshot for daemon status surfaces."""
        realtime = copy.deepcopy(status)
        realtime["startup"] = self._startup_tracker.snapshot()
        self._scan_progress["realtime"] = realtime

    def _record_realtime_failure(self, message: str) -> None:
        """Persist a startup failure into the shared realtime status snapshot."""
        realtime = copy.deepcopy(
            self._scan_progress.get("realtime") or self._default_realtime_scan_status()
        )
        realtime["service_state"] = "degraded"
        realtime["last_error"] = message
        realtime["last_error_at"] = datetime.now().isoformat()
        realtime["startup"] = self._startup_tracker.snapshot()
        self._scan_progress["realtime"] = realtime

    def _handle_realtime_start_task_done(self, task: asyncio.Task) -> None:
        """Capture realtime startup task failures so they are never silent."""
        if task.cancelled():
            if self._startup_tracker.snapshot()["state"] == "running":
                self._set_startup_failure(
                    "Realtime startup task was cancelled before completion",
                    phase_name="realtime_start",
                )
            return

        try:
            exc = task.exception()
        except Exception as error:
            self.debug_log(f"Failed to inspect realtime startup task: {error}")
            return

        if exc is None:
            if self._realtime_startup_mode() == "stdio":
                self._complete_startup()
            return

        self.debug_log(f"Realtime startup task failed: {exc}")
        self._set_startup_failure(
            f"Realtime startup task failed: {exc}",
            phase_name="realtime_start",
        )

    async def _request_realtime_resync(
        self, reason: str, details: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Run a serialized reconciliation scan and return the embed result."""
        if not self._scan_target_path:
            raise RuntimeError("Realtime resync requested before target path resolved")

        resync_started_monotonic = time.monotonic()
        loss_of_sync_reason = (
            details.get("loss_of_sync_reason")
            if isinstance(details, dict)
            else None
        )
        is_fresh_instance = self._is_fresh_instance_resync(reason, details)
        daemon_visible = self._realtime_startup_mode() == "daemon" and (
            self._warm_ready_window_active()
        )
        if is_fresh_instance and daemon_visible:
            self._warm_ready_fresh_instance_resync_requested = True
        detail_suffix = f" details={details}" if details else ""
        self.debug_log(f"Realtime resync requested: reason={reason}{detail_suffix}")
        self._timing_log(
            "realtime resync requested "
            f"reason={reason} loss_of_sync_reason={loss_of_sync_reason or 'none'} "
            f"fresh_instance={is_fresh_instance}",
            daemon_visible=daemon_visible,
        )
        resync_outcome = "scan_only"
        try:
            rescan_started_monotonic = time.monotonic()
            self._timing_log(
                "realtime resync directory scan starting "
                f"reason={reason} loss_of_sync_reason={loss_of_sync_reason or 'none'}",
                daemon_visible=daemon_visible,
            )
            await self._run_directory_scan(
                self._scan_target_path,
                trigger="realtime_resync",
                reason=reason,
                no_embeddings=True,
            )
            rescan_duration = self._duration_seconds(
                rescan_started_monotonic,
                time.monotonic(),
            )
            self._timing_log(
                "realtime resync directory scan completed "
                f"reason={reason} duration={self._format_duration(rescan_duration)}",
                daemon_visible=daemon_visible,
            )
            if self.services is None:
                return None
            embeddings_disabled = getattr(self.config, "embeddings_disabled", False)
            if not isinstance(embeddings_disabled, bool):
                embeddings_disabled = False
            if embeddings_disabled:
                resync_outcome = "embeddings_disabled"
                self._timing_log(
                    "realtime resync embedding follow-up skipped "
                    "reason=embeddings_disabled",
                    daemon_visible=daemon_visible,
                )
                self.debug_log(
                    "Realtime resync embedding follow-up skipped: "
                    "embeddings explicitly disabled"
                )
                return {
                    "status": "complete",
                    "generated": 0,
                    "message": "Embeddings explicitly disabled",
                }

            exclude_patterns = list(getattr(self.config.indexing, "exclude", []) or [])
            embed_started_monotonic = time.monotonic()
            self._timing_log(
                "realtime resync embedding follow-up starting "
                f"reason={reason} loss_of_sync_reason={loss_of_sync_reason or 'none'}",
                daemon_visible=daemon_visible,
            )
            embed_result = await (
                self.services.indexing_coordinator.generate_missing_embeddings(
                    exclude_patterns=exclude_patterns
                )
            )
            generated = embed_result.get("generated", 0)
            resync_outcome = str(embed_result.get("status") or "unknown")
            embed_duration = self._duration_seconds(
                embed_started_monotonic,
                time.monotonic(),
            )
            self._timing_log(
                "realtime resync embedding follow-up completed "
                "duration="
                f"{self._format_duration(embed_duration)} "
                f"status={embed_result.get('status')} generated={generated}",
                daemon_visible=daemon_visible,
            )
            self.debug_log(
                "Realtime resync embedding follow-up completed: "
                f"status={embed_result.get('status')} generated={generated}"
            )
            return embed_result
        except Exception:
            resync_outcome = "error"
            raise
        finally:
            resync_completed_monotonic = time.monotonic()
            total_duration = self._duration_seconds(
                resync_started_monotonic,
                resync_completed_monotonic,
            )
            self._timing_log(
                "realtime resync completed "
                f"reason={reason} loss_of_sync_reason={loss_of_sync_reason or 'none'} "
                f"fresh_instance={is_fresh_instance} outcome={resync_outcome} "
                f"total={self._format_duration(total_duration)}",
                daemon_visible=daemon_visible,
            )
            if is_fresh_instance and daemon_visible:
                self._warm_ready_fresh_instance_resync_total_seconds = total_duration
                self._warm_ready_fresh_instance_resync_completed_monotonic = (
                    resync_completed_monotonic
                )
                self._warm_ready_fresh_instance_resync_outcome = resync_outcome
                self._warm_ready_fresh_instance_resync_authoritative = (
                    self._is_authoritative_fresh_instance_resync_outcome(
                        resync_outcome
                    )
                )
                self._emit_warm_ready_summary_if_ready()

    async def _run_directory_scan(
        self,
        target_path: Path,
        trigger: str,
        reason: str | None = None,
        no_embeddings: bool = False,
    ) -> None:
        """Perform an initial or reconciliation scan without overlapping other scans."""
        scan_lock_wait_started_monotonic = time.monotonic()
        daemon_visible = self._realtime_startup_mode() == "daemon" and (
            trigger == "initial" or self._warm_ready_window_active()
        )
        async with self._scan_lock:
            execution_started_monotonic = time.monotonic()
            lock_wait_duration = self._duration_seconds(
                scan_lock_wait_started_monotonic,
                execution_started_monotonic,
            )
            self._timing_log(
                "directory scan lock acquired "
                f"trigger={trigger} reason={reason or 'none'} "
                f"wait={self._format_duration(lock_wait_duration)}",
                daemon_visible=daemon_visible,
            )
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
                execution_duration = self._duration_seconds(
                    execution_started_monotonic,
                    time.monotonic(),
                )
                self._timing_log(
                    "directory scan completed "
                    f"trigger={trigger} reason={reason or 'none'} "
                    f"duration={self._format_duration(execution_duration)} "
                    f"files={stats.files_processed} chunks={stats.chunks_created}",
                    daemon_visible=daemon_visible,
                )

            except Exception as e:
                self.debug_log(f"{trigger} scan failed: {e}")
                self._scan_progress["is_scanning"] = False
                self._scan_progress["scan_error"] = str(e)
                execution_duration = self._duration_seconds(
                    execution_started_monotonic,
                    time.monotonic(),
                )
                self._timing_log(
                    "directory scan failed "
                    f"trigger={trigger} reason={reason or 'none'} "
                    f"duration={self._format_duration(execution_duration)} "
                    f"error={e}",
                    daemon_visible=daemon_visible,
                )
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
