"""Database utility functions for CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chunkhound.core.config.config import Config


class DaemonProxy:
    """Proxy for communicating with the ChunkHound daemon.

    Used by CLI commands to forward operations to the running daemon
    instead of accessing the database directly.
    """

    def __init__(self, daemon_url: str, timeout: float = 30.0):
        """Initialize daemon proxy.

        Args:
            daemon_url: Base URL of the daemon (e.g., http://127.0.0.1:5173)
            timeout: Request timeout in seconds
        """
        self.daemon_url = daemon_url.rstrip("/")
        self.timeout = timeout

    def index_project(
        self,
        path: Path,
        name: str | None = None,
        tags: list[str] | None = None,
        async_mode: bool = True,
    ) -> dict[str, Any]:
        """Request daemon to index a project.

        Args:
            path: Path to the project directory
            name: Optional project name
            tags: Optional list of tags to apply
            async_mode: If True, return immediately with job ID (default)

        Returns:
            Response dict with status and details (or job_id if async)
        """
        import httpx

        # Use short timeout for async mode, long timeout for sync mode
        request_timeout = self.timeout if async_mode else max(self.timeout, 600.0)

        response = httpx.post(
            f"{self.daemon_url}/projects/index",
            json={
                "path": str(path.resolve()),
                "name": name,
                "tags": tags or [],
                "async": async_mode,
            },
            timeout=request_timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get status of an indexing job.

        Args:
            job_id: Job ID to check

        Returns:
            Job status dict
        """
        import httpx

        response = httpx.get(
            f"{self.daemon_url}/jobs/{job_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def list_jobs(
        self, include_completed: bool = True, limit: int = 20
    ) -> list[dict[str, Any]]:
        """List all indexing jobs.

        Args:
            include_completed: Whether to include completed jobs
            limit: Maximum number of jobs to return

        Returns:
            List of job status dicts
        """
        import httpx

        response = httpx.get(
            f"{self.daemon_url}/jobs",
            params={"completed": str(include_completed).lower(), "limit": limit},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("jobs", [])

    def get_stats(self, path: str | Path | None = None) -> dict[str, Any]:
        """Get database statistics.

        Args:
            path: Optional path to filter stats by project

        Returns:
            Dict with files, chunks, embeddings counts
        """
        import httpx

        params = {}
        if path:
            params["path"] = str(path)

        response = httpx.get(
            f"{self.daemon_url}/stats",
            params=params if params else None,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel an indexing job.

        Args:
            job_id: Job ID to cancel

        Returns:
            Result dict
        """
        import httpx

        response = httpx.post(
            f"{self.daemon_url}/jobs/{job_id}/cancel",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: float | None = None,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Wait for an indexing job to complete.

        Args:
            job_id: Job ID to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum time to wait (None for no limit)
            progress_callback: Optional callback(job_status) for progress updates

        Returns:
            Final job status dict

        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        import time

        start_time = time.time()

        while True:
            status = self.get_job_status(job_id)

            if progress_callback:
                progress_callback(status)

            job_status = status.get("status")
            if job_status in ("completed", "failed", "cancelled"):
                return status

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

            time.sleep(poll_interval)

    def remove_project(
        self, name_or_path: str, cascade: bool = False
    ) -> dict[str, Any]:
        """Request daemon to remove a project.

        Args:
            name_or_path: Project name or path
            cascade: If True, also delete indexed data

        Returns:
            Response dict with status
        """
        import httpx

        response = httpx.post(
            f"{self.daemon_url}/projects/remove",
            json={"name": name_or_path, "cascade": cascade},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def list_projects(self) -> list[dict[str, Any]]:
        """Get list of all indexed projects.

        Returns:
            List of project info dicts
        """
        import httpx

        response = httpx.get(
            f"{self.daemon_url}/projects",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("projects", [])

    def get_project_tags(self, name_or_path: str) -> list[str]:
        """Get tags for a specific project.

        Args:
            name_or_path: Project name or path

        Returns:
            List of tags
        """
        import httpx

        response = httpx.post(
            f"{self.daemon_url}/projects/tags",
            json={"operation": "list", "name": name_or_path},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("tags", [])

    def list_all_tags(self) -> list[str]:
        """Get all unique tags across all projects.

        Returns:
            List of all tags
        """
        import httpx

        response = httpx.post(
            f"{self.daemon_url}/projects/tags",
            json={"operation": "list"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("tags", [])

    def add_project_tags(self, name_or_path: str, tags: list[str]) -> dict[str, Any]:
        """Add tags to a project.

        Args:
            name_or_path: Project name or path
            tags: Tags to add

        Returns:
            Response dict with updated tags
        """
        import httpx

        response = httpx.post(
            f"{self.daemon_url}/projects/tags",
            json={"operation": "add", "name": name_or_path, "tags": tags},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def remove_project_tags(self, name_or_path: str, tags: list[str]) -> dict[str, Any]:
        """Remove tags from a project.

        Args:
            name_or_path: Project name or path
            tags: Tags to remove

        Returns:
            Response dict with updated tags
        """
        import httpx

        response = httpx.post(
            f"{self.daemon_url}/projects/tags",
            json={"operation": "remove", "name": name_or_path, "tags": tags},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def set_project_tags(
        self, name_or_path: str | Path, tags: list[str]
    ) -> dict[str, Any]:
        """Set tags for a project (replaces existing).

        Args:
            name_or_path: Project name or path
            tags: Tags to set

        Returns:
            Response dict with updated tags
        """
        import httpx

        response = httpx.post(
            f"{self.daemon_url}/projects/tags",
            json={"operation": "set", "name": str(name_or_path), "tags": tags},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


def get_daemon_proxy_if_running(config: Config) -> DaemonProxy | None:
    """Get a daemon proxy if daemon is running in global mode.

    Args:
        config: Configuration object

    Returns:
        DaemonProxy if daemon is running, None otherwise
    """
    # Check if global mode is enabled
    if not config.database.multi_repo.enabled:
        return None
    if config.database.multi_repo.mode != "global":
        return None

    # Check if daemon is running
    from chunkhound.services.daemon_manager import DaemonManager

    daemon = DaemonManager(config=config)
    status = daemon.status()

    if status.running and status.url:
        timeout = config.database.multi_repo.proxy_timeout_seconds
        return DaemonProxy(status.url, timeout=timeout)

    return None


def is_global_mode(config: Config) -> bool:
    """Check if global database mode is enabled.

    Args:
        config: Configuration object

    Returns:
        True if global mode is enabled
    """
    return (
        config.database.multi_repo.enabled
        and config.database.multi_repo.mode == "global"
    )


def require_daemon_or_exit(config: Config, formatter: Any) -> DaemonProxy:
    """Get daemon proxy or exit with error message.

    In global mode, direct database access is not allowed to prevent
    lock contention. This function enforces that constraint.

    Args:
        config: Configuration object
        formatter: RichOutputFormatter for error output

    Returns:
        DaemonProxy if daemon is running

    Raises:
        SystemExit: If global mode is enabled but daemon not running
    """
    import sys

    proxy = get_daemon_proxy_if_running(config)
    if proxy:
        return proxy

    # Check if we're in global mode - if so, daemon is required
    if is_global_mode(config):
        formatter.error(
            "Global mode is enabled but daemon is not running.\n"
            "Direct database access is disabled to prevent lock contention.\n\n"
            "Start the daemon with:\n"
            "  chunkhound daemon start --background\n\n"
            "Or enable auto-start:\n"
            "  export CHUNKHOUND_DATABASE__MULTI_REPO__AUTO_START_DAEMON=true"
        )
        sys.exit(1)

    # Not in global mode - should not reach here, but return None equivalent
    # Caller should handle None case for per-repo mode
    raise ValueError("require_daemon_or_exit called but not in global mode")


def verify_database_exists(config: Config, current_dir: Path | None = None) -> Path:
    """Verify database exists, raising if not found.

    Args:
        config: Configuration with database settings
        current_dir: Current directory for per-repo mode path resolution

    Raises:
        FileNotFoundError: If database doesn't exist
        ValueError: If database path not configured
    """
    # Get database path (handles global mode automatically)
    try:
        db_path = config.database.get_db_path(current_dir=current_dir or Path.cwd())
    except ValueError as e:
        raise ValueError(f"Database path not configured: {e}") from e

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            f"Run 'chunkhound index <directory>' to create the database first."
        )
    return db_path
