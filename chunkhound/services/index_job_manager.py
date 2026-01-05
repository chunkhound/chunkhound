"""Background index job manager for async indexing operations.

Manages background indexing jobs that run independently of HTTP request lifecycle,
allowing search operations to continue while indexing is in progress.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from chunkhound.core.config.config import Config
    from chunkhound.services.indexing_coordinator import IndexingCoordinator
    from chunkhound.services.project_registry import ProjectRegistry
    from chunkhound.services.watcher_manager import WatcherManager


class JobStatus(str, Enum):
    """Status of an indexing job."""

    QUEUED = "queued"
    RUNNING = "running"
    EMBEDDING = "embedding"  # Separate phase for embedding generation
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class IndexJob:
    """Represents a background indexing job."""

    job_id: str
    project_path: Path
    project_name: str | None
    tags: list[str] = field(default_factory=list)

    # Status tracking
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None

    # Progress tracking
    phase: str = "queued"
    files_discovered: int = 0
    files_processed: int = 0
    chunks_created: int = 0
    embeddings_total: int = 0
    embeddings_generated: int = 0

    # Result/error tracking
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "project_path": str(self.project_path),
            "project_name": self.project_name,
            "tags": self.tags,
            "status": self.status.value,
            "phase": self.phase,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": {
                "files_discovered": self.files_discovered,
                "files_processed": self.files_processed,
                "chunks_created": self.chunks_created,
                "embeddings_total": self.embeddings_total,
                "embeddings_generated": self.embeddings_generated,
            },
            "result": self.result,
            "error": self.error,
            "elapsed_seconds": self._elapsed_seconds(),
        }

    def _elapsed_seconds(self) -> float:
        """Calculate elapsed time for the job."""
        if self.completed_at:
            return round(self.completed_at - self.created_at, 2)
        if self.started_at:
            return round(time.time() - self.started_at, 2)
        return 0.0


class IndexJobManager:
    """Manages background indexing jobs.

    Provides:
    - Job creation and lifecycle management
    - Progress tracking during indexing and embedding
    - Concurrent job limit enforcement
    - Job history with automatic cleanup
    """

    # Maximum concurrent indexing jobs
    MAX_CONCURRENT_JOBS = 2

    # How long to keep completed/failed jobs in history
    JOB_HISTORY_TTL_SECONDS = 3600  # 1 hour

    def __init__(
        self,
        indexing_coordinator: IndexingCoordinator,
        config: Config,
        project_registry: ProjectRegistry | None = None,
        watcher_manager: WatcherManager | None = None,
        debug_sink: Any = None,
    ):
        """Initialize job manager.

        Args:
            indexing_coordinator: Coordinator for indexing operations
            config: Application configuration
            project_registry: Optional registry for multi-repo mode
            watcher_manager: Optional watcher manager for file monitoring
            debug_sink: Optional callback for debug logging
        """
        self.indexing_coordinator = indexing_coordinator
        self.config = config
        self.project_registry = project_registry
        self.watcher_manager = watcher_manager
        self.debug_sink = debug_sink

        # Active jobs
        self._jobs: dict[str, IndexJob] = {}
        self._job_tasks: dict[str, asyncio.Task] = {}

        # Lock for job management
        self._lock = asyncio.Lock()

        # Cleanup task
        self._cleanup_task: asyncio.Task | None = None
        self._shutdown = False

    def _debug(self, msg: str) -> None:
        """Log debug message via debug sink if available."""
        if self.debug_sink:
            self.debug_sink(msg)
        else:
            logger.debug(msg)

    async def start(self) -> None:
        """Start the job manager."""
        self._shutdown = False
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._debug("IndexJobManager started")

    async def stop(self) -> None:
        """Stop the job manager and cancel all running jobs."""
        self._shutdown = True

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cancel all running jobs
        async with self._lock:
            for job_id, task in list(self._job_tasks.items()):
                if not task.done():
                    task.cancel()
                    job = self._jobs.get(job_id)
                    if job:
                        job.status = JobStatus.CANCELLED
                        job.completed_at = time.time()

        self._debug("IndexJobManager stopped")

    async def create_job(
        self,
        project_path: Path,
        project_name: str | None = None,
        tags: list[str] | None = None,
    ) -> IndexJob:
        """Create and queue a new indexing job.

        Args:
            project_path: Path to the project directory
            project_name: Optional project name
            tags: Optional list of tags

        Returns:
            The created IndexJob

        Raises:
            RuntimeError: If max concurrent jobs exceeded
        """
        async with self._lock:
            # Check concurrent job limit
            running_count = sum(
                1
                for j in self._jobs.values()
                if j.status
                in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.EMBEDDING)
            )
            if running_count >= self.MAX_CONCURRENT_JOBS:
                raise RuntimeError(
                    f"Maximum concurrent jobs ({self.MAX_CONCURRENT_JOBS}) exceeded. "
                    f"Wait for existing jobs to complete."
                )

            # Check if already indexing this project
            for job in self._jobs.values():
                if job.project_path == project_path and job.status in (
                    JobStatus.QUEUED,
                    JobStatus.RUNNING,
                    JobStatus.EMBEDDING,
                ):
                    raise RuntimeError(
                        f"Project {project_path} is already being indexed (job {job.job_id})"
                    )

            # Check for nested path conflicts BEFORE starting indexing
            if self.project_registry:
                self.project_registry.validate_path_not_nested(project_path)

            # Create job
            job_id = str(uuid.uuid4())[:8]  # Short ID for readability
            job = IndexJob(
                job_id=job_id,
                project_path=project_path,
                project_name=project_name,
                tags=tags or [],
            )
            self._jobs[job_id] = job

            # Start the job in background
            task = asyncio.create_task(self._run_job(job))
            self._job_tasks[job_id] = task

            self._debug(f"Created indexing job {job_id} for {project_path}")
            return job

    def get_job(self, job_id: str) -> IndexJob | None:
        """Get a job by ID.

        Args:
            job_id: Job ID to look up

        Returns:
            The IndexJob or None if not found
        """
        return self._jobs.get(job_id)

    def list_jobs(
        self, include_completed: bool = True, limit: int = 20
    ) -> list[IndexJob]:
        """List all jobs.

        Args:
            include_completed: Whether to include completed/failed jobs
            limit: Maximum number of jobs to return

        Returns:
            List of IndexJob objects, newest first
        """
        jobs = list(self._jobs.values())

        if not include_completed:
            jobs = [
                j
                for j in jobs
                if j.status
                in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.EMBEDDING)
            ]

        # Sort by creation time, newest first
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def get_active_jobs(self) -> list[IndexJob]:
        """Get all currently active (running/embedding) jobs.

        Returns:
            List of active IndexJob objects
        """
        return [
            j
            for j in self._jobs.values()
            if j.status in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.EMBEDDING)
        ]

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled, False if not found or already completed
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            if job.status not in (
                JobStatus.QUEUED,
                JobStatus.RUNNING,
                JobStatus.EMBEDDING,
            ):
                return False  # Already completed/failed

            task = self._job_tasks.get(job_id)
            if task and not task.done():
                task.cancel()

            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            self._debug(f"Cancelled job {job_id}")
            return True

    async def _run_job(self, job: IndexJob) -> None:
        """Execute an indexing job.

        Args:
            job: The job to execute
        """
        try:
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            job.phase = "indexing"

            self._debug(f"Starting job {job.job_id}: {job.project_path}")

            # Get file patterns from config
            include_patterns = list(self.config.indexing.include)
            exclude_patterns = list(self.config.indexing.exclude)

            # Progress callback to update job state during indexing
            def update_indexing_progress(
                files_processed: int, chunks_created: int
            ) -> None:
                job.files_processed = files_processed
                job.chunks_created = chunks_created

            # Run indexing with progress callback
            result = await self.indexing_coordinator.process_directory(
                job.project_path,
                patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                progress_callback=update_indexing_progress,
            )

            # Final update from result
            job.files_processed = result.get("files_processed", 0)
            job.chunks_created = result.get("total_chunks", 0)

            if result.get("status") == "error":
                job.status = JobStatus.FAILED
                job.error = result.get("error", "Unknown error")
                job.completed_at = time.time()
                job.result = result
                return

            # Phase 2: Generate embeddings
            job.status = JobStatus.EMBEDDING
            job.phase = "embedding"

            # Get count of chunks needing embeddings
            # Note: This is just for progress estimation, actual table name may vary
            if hasattr(self.indexing_coordinator, "_db"):
                try:
                    # Try to get count using provider's get_unembedded_chunk_count if available
                    if hasattr(
                        self.indexing_coordinator._db, "get_unembedded_chunk_count"
                    ):
                        job.embeddings_total = (
                            self.indexing_coordinator._db.get_unembedded_chunk_count()
                        )
                except Exception as e:
                    self._debug(f"Failed to get embedding count: {e}")

            # Generate embeddings with progress tracking
            embed_result = await self._generate_embeddings_with_progress(job)
            result["embeddings_generated"] = embed_result.get("generated", 0)

            # Register project in registry if available
            if self.project_registry:
                project = self.project_registry.register_project(
                    job.project_path, name=job.project_name
                )

                # Apply tags
                if job.tags:
                    self.project_registry.set_project_tags(
                        str(job.project_path), job.tags
                    )

                # Update file count stats
                if hasattr(self.indexing_coordinator._db, "update_indexed_root_stats"):
                    self.indexing_coordinator._db.update_indexed_root_stats(
                        str(job.project_path)
                    )

                # Start watcher
                if self.watcher_manager:
                    self.watcher_manager.start_watcher(project)

            # Job completed successfully
            job.status = JobStatus.COMPLETED
            job.phase = "completed"
            job.completed_at = time.time()
            job.result = result

            self._debug(
                f"Job {job.job_id} completed: {job.files_processed} files, "
                f"{job.chunks_created} chunks, {job.embeddings_generated} embeddings"
            )

        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            self._debug(f"Job {job.job_id} cancelled")
            raise

        except Exception as e:
            logger.exception(f"Job {job.job_id} failed")
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()

    async def _generate_embeddings_with_progress(self, job: IndexJob) -> dict[str, Any]:
        """Generate embeddings with progress tracking.

        Args:
            job: The job to track progress for

        Returns:
            Result dict from embedding generation
        """
        # Get initial count of chunks needing embeddings
        try:
            if hasattr(self.indexing_coordinator._db, "get_unembedded_chunk_count"):
                job.embeddings_total = (
                    self.indexing_coordinator._db.get_unembedded_chunk_count()
                )
        except Exception:
            pass

        # Start progress polling task
        stop_polling = asyncio.Event()

        async def poll_progress() -> None:
            """Poll database for embedding progress."""
            while not stop_polling.is_set():
                try:
                    # Get count of embedded chunks
                    if hasattr(
                        self.indexing_coordinator._db, "get_embedded_chunk_count"
                    ):
                        embedded = (
                            self.indexing_coordinator._db.get_embedded_chunk_count()
                        )
                        job.embeddings_generated = embedded
                    elif job.embeddings_total > 0:
                        # Estimate from remaining unembedded
                        if hasattr(
                            self.indexing_coordinator._db, "get_unembedded_chunk_count"
                        ):
                            remaining = self.indexing_coordinator._db.get_unembedded_chunk_count()
                            job.embeddings_generated = job.embeddings_total - remaining
                except Exception:
                    pass
                await asyncio.sleep(1.0)  # Poll every second

        # Run embedding generation with progress polling
        poll_task = asyncio.create_task(poll_progress())
        try:
            result = await self.indexing_coordinator.generate_missing_embeddings()
        finally:
            stop_polling.set()
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass

        # Final update from result
        job.embeddings_generated = result.get("generated", 0)

        return result

    async def _cleanup_loop(self) -> None:
        """Background task to clean up old completed jobs."""
        while not self._shutdown:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Re-check after sleep (shutdown may have been requested during sleep)
                if self._shutdown:
                    return

                cutoff = time.time() - self.JOB_HISTORY_TTL_SECONDS
                async with self._lock:
                    to_remove = [
                        job_id
                        for job_id, job in self._jobs.items()
                        if job.completed_at
                        and job.completed_at < cutoff
                        and job.status
                        in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
                    ]
                    for job_id in to_remove:
                        del self._jobs[job_id]
                        self._job_tasks.pop(job_id, None)

                    if to_remove:
                        self._debug(f"Cleaned up {len(to_remove)} old jobs")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in job cleanup loop: {e}")
                await asyncio.sleep(60)  # Back off on error
