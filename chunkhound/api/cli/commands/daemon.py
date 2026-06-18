"""Daemon command — starts the ChunkHound multi-client daemon process."""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from chunkhound.core.config.config import Config


def _startup_breadcrumb(message: str) -> None:
    """Emit daemon bootstrap breadcrumbs before MCPServerBase exists."""
    try:
        timestamp = datetime.now().isoformat()
        print(
            f"[{timestamp}] [startup] startup: {message}",
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        pass


class _StartupPhase:
    def __init__(self, name: str) -> None:
        self._name = name
        self._started_at = 0.0

    def __enter__(self) -> None:
        self._started_at = time.monotonic()
        _startup_breadcrumb(f"phase started: {self._name}")

    def __exit__(self, _exc_type: object, exc: object, _traceback: object) -> None:
        duration = time.monotonic() - self._started_at
        if exc is None:
            _startup_breadcrumb(
                f"phase completed: {self._name} duration={duration:.3f}s"
            )
            return
        _startup_breadcrumb(
            f"phase failed: {self._name} duration={duration:.3f}s error={exc!r}"
        )


async def daemon_command(args: argparse.Namespace, config: Config) -> None:
    """Launch ChunkHoundDaemon bound to the given socket path.

    Args:
        args: Parsed CLI arguments (must include ``project_dir`` and
              ``socket_path``).
        config: Pre-validated configuration instance.
    """
    _startup_breadcrumb("phase started: daemon_command")

    # CRITICAL: Import numpy early for DuckDB threading safety.
    # The daemon owns the sole DuckDB connection; this must happen before
    # initialize() opens the database.
    # See: https://duckdb.org/docs/stable/clients/python/known_issues.html
    with _StartupPhase("daemon_numpy_import"):
        try:
            import numpy  # noqa: F401
        except ImportError:
            pass

    with _StartupPhase("daemon_server_import"):
        from chunkhound.daemon.server import ChunkHoundDaemon

    with _StartupPhase("daemon_construct"):
        # argparse converts --project-dir to args.project_dir (with underscore)
        project_dir = Path(args.project_dir).resolve()
        socket_path: str = args.socket_path

        # Set args.path so that Config and MCPServerBase.initialize() resolve
        # the target directory correctly (they look for args.path).
        args.path = project_dir

        daemon = ChunkHoundDaemon(
            config=config,
            args=args,
            socket_path=socket_path,
            project_dir=project_dir,
        )

    _startup_breadcrumb("phase completed: daemon_command")
    _startup_breadcrumb("phase started: daemon_run")
    await daemon.run()


__all__: list[str] = ["daemon_command"]
