"""Daemon manager for ChunkHound HTTP server lifecycle.

This service manages the ChunkHound daemon (HTTP server) lifecycle:
- Starting the daemon in foreground or background
- Stopping a running daemon
- Checking daemon status
- Managing PID files and logs

The daemon is the HTTP MCP server that owns the global database and
handles file watching for all indexed projects.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from chunkhound.core.config.config import Config


@dataclass
class DaemonStatus:
    """Status information for the daemon.

    Attributes:
        running: Whether the daemon is running
        pid: Process ID if running
        host: Host the daemon is bound to
        port: Port the daemon is listening on
        url: Full URL to connect to the daemon
        uptime_seconds: How long the daemon has been running
        pid_file: Path to the PID file
    """

    running: bool
    pid: int | None = None
    host: str | None = None
    port: int | None = None
    url: str | None = None
    uptime_seconds: float | None = None
    pid_file: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "running": self.running,
            "pid": self.pid,
            "host": self.host,
            "port": self.port,
            "url": self.url,
            "uptime_seconds": self.uptime_seconds,
            "pid_file": str(self.pid_file) if self.pid_file else None,
        }


class DaemonManager:
    """Manages the ChunkHound daemon lifecycle.

    The DaemonManager handles starting, stopping, and monitoring the
    ChunkHound HTTP server daemon. It uses PID files for process tracking
    and provides health check capabilities.

    Usage:
        manager = DaemonManager()

        # Start daemon
        manager.start(background=True)

        # Check status
        status = manager.status()
        if status.running:
            print(f"Daemon running at {status.url}")

        # Stop daemon
        manager.stop()
    """

    # Default paths (used when no config provided)
    DEFAULT_DATA_DIR = Path.home() / ".chunkhound"
    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 5173

    def __init__(
        self,
        data_dir: Path | None = None,
        host: str | None = None,
        port: int | None = None,
        config: Config | None = None,
    ):
        """Initialize daemon manager.

        Args:
            data_dir: Base directory for daemon files (default: ~/.chunkhound)
            host: Host to bind to (default from config or 127.0.0.1)
            port: Port to listen on (default from config or 5173)
            config: Optional Config object for loading settings
        """
        self.data_dir = data_dir or self.DEFAULT_DATA_DIR

        # Load defaults from config if provided
        config_host = self.DEFAULT_HOST
        config_port = self.DEFAULT_PORT
        if config is not None:
            config_host = config.database.multi_repo.daemon_host
            config_port = config.database.multi_repo.daemon_port

        self.host = host or config_host
        self.port = port or config_port

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.pid_file = self.data_dir / "daemon.pid"
        self.log_dir = self.data_dir / "logs"
        self.log_file = self.log_dir / "daemon.log"

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def url(self) -> str:
        """Get the daemon URL."""
        return f"http://{self.host}:{self.port}"

    def _read_pid(self) -> int | None:
        """Read PID from PID file.

        Returns:
            PID if file exists and valid, None otherwise
        """
        if not self.pid_file.exists():
            return None

        try:
            pid_str = self.pid_file.read_text().strip()
            return int(pid_str)
        except (ValueError, OSError):
            return None

    def _write_pid(self, pid: int) -> None:
        """Write PID to PID file.

        Args:
            pid: Process ID to write
        """
        self.pid_file.write_text(str(pid))

    def _remove_pid(self) -> None:
        """Remove PID file."""
        try:
            self.pid_file.unlink(missing_ok=True)
        except OSError:
            pass

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is running.

        Args:
            pid: Process ID to check

        Returns:
            True if process is running
        """
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _is_port_in_use(self) -> bool:
        """Check if the daemon port is in use.

        Returns:
            True if port is in use
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((self.host, self.port))
                return False
            except OSError:
                return True

    def _check_log_for_lock_error(self) -> str | None:
        """Check daemon log for database lock errors.

        Returns:
            User-friendly error message if lock error found, None otherwise
        """
        import re

        if not self.log_file.exists():
            return None

        try:
            # Read last 50 lines of log
            log_content = self.log_file.read_text()
            lines = log_content.strip().split("\n")[-50:]
            log_text = "\n".join(lines)

            # Look for DuckDB lock error pattern
            # Example: Conflicting lock is held in /path/to/python (PID 1269862)
            lock_pattern = r"Conflicting lock is held in .+? \(PID (\d+)\)"
            match = re.search(lock_pattern, log_text)

            if match:
                pid = match.group(1)
                # Try to identify what process is holding the lock
                try:
                    import subprocess

                    result = subprocess.run(
                        ["ps", "-p", pid, "-o", "args="],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    cmd = result.stdout.strip() if result.returncode == 0 else "unknown"
                except Exception:
                    cmd = "unknown"

                if "chunkhound" in cmd.lower() and "mcp" in cmd.lower():
                    return (
                        f"Database locked by MCP session (PID: {pid}).\n"
                        f"  Kill it with: kill {pid}\n"
                        f"  Or wait for the session to end."
                    )
                else:
                    return (
                        f"Database locked by another process (PID: {pid}).\n"
                        f"  Process: {cmd[:60]}{'...' if len(cmd) > 60 else ''}\n"
                        f"  Kill it with: kill {pid}"
                    )

            # Check for generic connection errors
            if "Could not set lock on file" in log_text:
                return "Database locked by another process. Check for running ChunkHound sessions."

        except Exception:
            pass

        return None

    def _check_health(self) -> bool:
        """Check if daemon is healthy via HTTP health endpoint.

        Returns:
            True if daemon responds to health check
        """
        try:
            import httpx

            response = httpx.get(f"{self.url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def is_running(self) -> bool:
        """Check if daemon is running.

        Returns:
            True if daemon is running and healthy
        """
        # Check PID file
        pid = self._read_pid()
        if pid and self._is_process_running(pid):
            # Verify it's actually the daemon by checking port
            if self._is_port_in_use():
                return True

        # PID file stale or process not running
        if pid:
            self._remove_pid()

        return False

    def get_pid(self) -> int | None:
        """Get the daemon PID if running.

        Returns:
            PID if running, None otherwise
        """
        if self.is_running():
            return self._read_pid()
        return None

    def status(self) -> DaemonStatus:
        """Get daemon status.

        Returns:
            DaemonStatus with current state
        """
        pid = self._read_pid()
        running = False
        uptime = None

        if pid and self._is_process_running(pid):
            if self._is_port_in_use():
                running = True

                # Try to get uptime from health endpoint
                try:
                    import httpx

                    response = httpx.get(f"{self.url}/health", timeout=5.0)
                    if response.status_code == 200:
                        data = response.json()
                        uptime = data.get("uptime_seconds")
                except Exception:
                    pass

        if not running and pid:
            # Clean up stale PID file
            self._remove_pid()
            pid = None

        return DaemonStatus(
            running=running,
            pid=pid,
            host=self.host if running else None,
            port=self.port if running else None,
            url=self.url if running else None,
            uptime_seconds=uptime,
            pid_file=self.pid_file,
        )

    def start(
        self,
        background: bool = False,
        wait: bool = True,
        timeout: float = 30.0,
    ) -> bool:
        """Start the daemon.

        Args:
            background: If True, daemonize the process
            wait: If True, wait for daemon to be ready
            timeout: Maximum time to wait for daemon to start

        Returns:
            True if daemon started successfully
        """
        # Check if already running
        if self.is_running():
            logger.info(f"Daemon already running at {self.url}")
            return True

        # Check if port is in use by something else
        if self._is_port_in_use():
            logger.error(f"Port {self.port} is already in use")
            return False

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "chunkhound",
            "mcp",
            "http",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        if background:
            # Start as daemon in background
            logger.info(f"Starting ChunkHound daemon on {self.url}...")

            # Open log file with context manager pattern
            # The file must stay open for subprocess lifetime, so we use
            # subprocess's built-in file descriptor inheritance
            log_fd = os.open(
                str(self.log_file),
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o640,  # Owner read/write, group read, no world access
            )

            try:
                # Start process with file descriptor (subprocess will manage it)
                process = subprocess.Popen(
                    cmd,
                    stdout=log_fd,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            finally:
                # Close our copy of the fd - subprocess has its own copy
                os.close(log_fd)

            # Write PID file
            self._write_pid(process.pid)

            if wait:
                # Wait for daemon to be ready
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if self._check_health():
                        logger.info(f"Daemon started successfully (PID: {process.pid})")
                        return True
                    time.sleep(0.5)

                # Timeout - check if process is still alive
                if process.poll() is not None:
                    # Check log for specific errors
                    lock_error = self._check_log_for_lock_error()
                    if lock_error:
                        logger.error(lock_error)
                    else:
                        logger.error(
                            f"Daemon process exited with code {process.returncode}"
                        )
                    self._remove_pid()
                    return False

                logger.warning("Daemon started but health check timed out")
                return True

            return True

        else:
            # Run in foreground (blocking)
            logger.info(f"Starting ChunkHound daemon on {self.url} (foreground)...")
            self._write_pid(os.getpid())

            try:
                # Execute in current process
                os.execv(sys.executable, cmd)
            except Exception as e:
                logger.error(f"Failed to start daemon: {e}")
                self._remove_pid()
                return False

    def stop(self, timeout: float = 10.0) -> bool:
        """Stop the daemon.

        Args:
            timeout: Maximum time to wait for graceful shutdown

        Returns:
            True if daemon was stopped
        """
        pid = self._read_pid()

        if not pid:
            logger.info("No daemon PID file found")
            return True

        if not self._is_process_running(pid):
            logger.info("Daemon process not running, cleaning up PID file")
            self._remove_pid()
            return True

        logger.info(f"Stopping daemon (PID: {pid})...")

        # Send SIGTERM for graceful shutdown
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            logger.error(f"Failed to send SIGTERM: {e}")
            return False

        # Wait for process to exit
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self._is_process_running(pid):
                logger.info("Daemon stopped gracefully")
                self._remove_pid()
                return True
            time.sleep(0.5)

        # Force kill if still running
        logger.warning("Daemon did not stop gracefully, sending SIGKILL")
        try:
            os.kill(pid, signal.SIGKILL)
            time.sleep(1.0)
        except OSError:
            pass

        self._remove_pid()
        return not self._is_process_running(pid)

    def restart(self, background: bool = True) -> bool:
        """Restart the daemon.

        Args:
            background: If True, run daemon in background

        Returns:
            True if daemon restarted successfully
        """
        self.stop()
        time.sleep(1.0)  # Brief pause before restart
        return self.start(background=background)

    def logs(self, follow: bool = False, lines: int = 100) -> str | None:
        """Get daemon logs.

        Args:
            follow: If True, tail the log file (blocking)
            lines: Number of lines to return (if not following)

        Returns:
            Log content or None if no logs
        """
        if not self.log_file.exists():
            return None

        if follow:
            # Use tail -f (blocking)
            import subprocess

            subprocess.run(["tail", "-f", str(self.log_file)])
            return None
        else:
            # Read last N lines
            try:
                content = self.log_file.read_text()
                log_lines = content.split("\n")
                return "\n".join(log_lines[-lines:])
            except OSError:
                return None


def get_daemon_manager(config: Config | None = None) -> DaemonManager:
    """Get a DaemonManager instance with configuration.

    Args:
        config: Optional Config object for loading settings.
            If not provided, defaults are used.

    Returns:
        DaemonManager instance
    """
    return DaemonManager(config=config)
