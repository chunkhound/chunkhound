"""CLI commands for managing the ChunkHound daemon.

The daemon is an HTTP MCP server that runs in the background, providing:
- Global database mode with cross-project search
- Centralized file watching for all indexed projects
- Multiple concurrent client connections

Commands:
    chunkhound daemon start [--background]
    chunkhound daemon stop
    chunkhound daemon status
    chunkhound daemon restart
    chunkhound daemon logs [--follow]
"""

import socket
import sys
from argparse import Namespace

from loguru import logger

from chunkhound.api.cli.utils.rich_output import RichOutputFormatter
from chunkhound.services.daemon_manager import DaemonManager


def _is_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding.

    Args:
        host: Host address to check
        port: Port number to check

    Returns:
        True if port is available, False if in use
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            # If connect succeeds (result == 0), port is in use
            return result != 0
    except Exception:
        # On error, assume port might be available
        return True


def _get_daemon_manager(args: Namespace) -> DaemonManager:
    """Create DaemonManager from arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        DaemonManager instance
    """
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 5173)

    return DaemonManager(host=host, port=port)


async def start_command(args: Namespace, formatter: RichOutputFormatter) -> None:
    """Start the ChunkHound daemon.

    Args:
        args: Parsed command-line arguments
        formatter: Output formatter for displaying results
    """
    manager = None
    try:
        manager = _get_daemon_manager(args)
        background = getattr(args, "background", False)

        # Check if already running
        status = manager.status()
        if status.running:
            formatter.info(
                f"Daemon already running at {status.url} (PID: {status.pid})"
            )
            return

        # Check for port conflicts (another process using the port)
        if not _is_port_available(manager.host, manager.port):
            formatter.error(
                f"Port {manager.port} is already in use by another process.\n"
                f"Try a different port with: chunkhound daemon start --port <port>"
            )
            sys.exit(1)

        formatter.info(f"Starting ChunkHound daemon on {manager.url}...")

        if background:
            success = manager.start(background=True, wait=True)
            if success:
                status = manager.status()
                formatter.success(f"Daemon started successfully (PID: {status.pid})")
                formatter.info(f"URL: {status.url}")
                formatter.verbose_info(f"Log file: {manager.log_file}")
            else:
                formatter.error("Failed to start daemon")
                sys.exit(1)
        else:
            # Foreground mode - this will block
            formatter.info("Starting in foreground mode (press Ctrl+C to stop)...")
            success = manager.start(background=False)
            if not success:
                formatter.error("Failed to start daemon")
                sys.exit(1)

    except KeyboardInterrupt:
        formatter.info("\nShutdown requested...")
        if manager is not None:
            manager.stop()
        formatter.info("Daemon stopped")
    except Exception as e:
        formatter.error(f"Failed to start daemon: {e}")
        logger.exception("Full error details:")
        sys.exit(1)


async def stop_command(args: Namespace, formatter: RichOutputFormatter) -> None:
    """Stop the ChunkHound daemon.

    Args:
        args: Parsed command-line arguments
        formatter: Output formatter for displaying results
    """
    try:
        manager = _get_daemon_manager(args)

        # Check if running
        status = manager.status()
        if not status.running:
            formatter.info("Daemon is not running")
            return

        formatter.info(f"Stopping daemon (PID: {status.pid})...")

        success = manager.stop()
        if success:
            formatter.success("Daemon stopped successfully")
        else:
            formatter.error("Failed to stop daemon")
            sys.exit(1)

    except Exception as e:
        formatter.error(f"Failed to stop daemon: {e}")
        logger.exception("Full error details:")
        sys.exit(1)


async def status_command(args: Namespace, formatter: RichOutputFormatter) -> None:
    """Show daemon status.

    Args:
        args: Parsed command-line arguments
        formatter: Output formatter for displaying results
    """
    try:
        manager = _get_daemon_manager(args)
        status = manager.status()

        if status.running:
            formatter.section_header("Daemon Status: Running")
            formatter.info(f"PID: {status.pid}")
            formatter.info(f"URL: {status.url}")

            if status.uptime_seconds is not None:
                uptime = _format_uptime(status.uptime_seconds)
                formatter.info(f"Uptime: {uptime}")

            formatter.verbose_info(f"PID file: {status.pid_file}")
            formatter.verbose_info(f"Log file: {manager.log_file}")

            # Try to get more info from health endpoint
            try:
                import httpx

                response = httpx.get(f"{status.url}/health", timeout=5.0)
                if response.status_code == 200:
                    health = response.json()
                    formatter.info(f"Version: {health.get('version', 'unknown')}")

                    # Resource usage
                    memory_mb = health.get("memory_mb", 0)
                    if memory_mb > 0:
                        formatter.info(f"Memory: {memory_mb:.1f} MB")

                    # Projects and watchers
                    projects_indexed = health.get("projects_indexed", 0)
                    watchers_active = health.get("watchers_active", 0)
                    active_sessions = health.get("active_sessions", 0)
                    formatter.info(f"Projects indexed: {projects_indexed}")
                    formatter.info(f"Watchers active: {watchers_active}")
                    pending = health.get("pending_events", 0)
                    if pending > 0:
                        formatter.warning(f"Pending events: {pending}")
                    else:
                        formatter.info(f"Pending events: {pending}")
                    formatter.info(f"Active sessions: {active_sessions}")

                    # Database stats
                    db = health.get("database", {})
                    if db:
                        formatter.info("")
                        formatter.section_header("Database")
                        formatter.info(f"  Files: {db.get('files', 0):,}")
                        formatter.info(f"  Chunks: {db.get('chunks', 0):,}")
                        formatter.info(f"  Embeddings: {db.get('embeddings', 0):,}")
            except Exception:
                pass

        else:
            formatter.section_header("Daemon Status: Not Running")
            formatter.info(f"Expected URL: {manager.url}")
            formatter.verbose_info(f"PID file: {status.pid_file}")

    except Exception as e:
        formatter.error(f"Failed to get daemon status: {e}")
        logger.exception("Full error details:")
        sys.exit(1)


async def restart_command(args: Namespace, formatter: RichOutputFormatter) -> None:
    """Restart the ChunkHound daemon.

    Args:
        args: Parsed command-line arguments
        formatter: Output formatter for displaying results
    """
    try:
        manager = _get_daemon_manager(args)

        formatter.info("Restarting daemon...")

        success = manager.restart(background=True)
        if success:
            status = manager.status()
            formatter.success(f"Daemon restarted successfully (PID: {status.pid})")
            formatter.info(f"URL: {status.url}")
        else:
            formatter.error("Failed to restart daemon")
            sys.exit(1)

    except Exception as e:
        formatter.error(f"Failed to restart daemon: {e}")
        logger.exception("Full error details:")
        sys.exit(1)


async def logs_command(args: Namespace, formatter: RichOutputFormatter) -> None:
    """Show daemon logs.

    Args:
        args: Parsed command-line arguments
        formatter: Output formatter for displaying results
    """
    try:
        manager = _get_daemon_manager(args)
        follow = getattr(args, "follow", False)
        lines = getattr(args, "lines", 100)

        if follow:
            formatter.info(f"Following logs from {manager.log_file}...")
            formatter.info("Press Ctrl+C to stop\n")
            manager.logs(follow=True)
        else:
            content = manager.logs(follow=False, lines=lines)
            if content:
                print(content)
            else:
                formatter.info("No logs available")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        formatter.error(f"Failed to read logs: {e}")
        logger.exception("Full error details:")
        sys.exit(1)


def _format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format.

    Args:
        seconds: Uptime in seconds

    Returns:
        Formatted string like "2h 30m" or "5m 12s"
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"
    else:
        days = int(seconds / 86400)
        hours = int((seconds % 86400) / 3600)
        return f"{days}d {hours}h"


async def refresh_watchers_command(
    args: Namespace, formatter: RichOutputFormatter
) -> None:
    """Refresh file watchers for all or a specific project.

    Args:
        args: Parsed command-line arguments
        formatter: Output formatter for displaying results
    """
    try:
        manager = _get_daemon_manager(args)

        # Check if daemon is running
        status = manager.status()
        if not status.running:
            formatter.error(
                "Daemon is not running. Start it with: chunkhound daemon start"
            )
            sys.exit(1)

        project = getattr(args, "project", None)

        scope = f" for {project}" if project else " for all projects"
        formatter.info(f"Refreshing watchers{scope}...")

        try:
            import httpx

            # Call the daemon's refresh endpoint
            url = f"{status.url}/watchers/refresh"
            params = {"project": project} if project else {}

            response = httpx.post(url, params=params, timeout=30.0)

            if response.status_code == 200:
                result = response.json()
                refreshed = result.get("refreshed", 0)
                errors = result.get("errors", [])

                if refreshed > 0:
                    formatter.success(f"Refreshed {refreshed} watcher(s)")

                if errors:
                    for err in errors:
                        formatter.warning(f"  {err}")
            elif response.status_code == 404:
                formatter.error("Refresh endpoint not available on this daemon version")
                sys.exit(1)
            else:
                formatter.error(f"Refresh failed: HTTP {response.status_code}")
                sys.exit(1)

        except httpx.ConnectError:
            formatter.error("Cannot connect to daemon")
            sys.exit(1)

    except Exception as e:
        formatter.error(f"Failed to refresh watchers: {e}")
        logger.exception("Full error details:")
        sys.exit(1)


__all__ = [
    "start_command",
    "stop_command",
    "status_command",
    "restart_command",
    "logs_command",
    "refresh_watchers_command",
]
