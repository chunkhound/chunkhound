"""Daemon command — starts the ChunkHound multi-client daemon process."""

import argparse
from pathlib import Path

from chunkhound.core.config.config import Config


async def daemon_command(args: argparse.Namespace, config: Config) -> None:
    """Launch ChunkHoundDaemon bound to the given socket path.

    Args:
        args: Parsed CLI arguments (must include ``project_dir`` and
              ``socket_path``).
        config: Pre-validated configuration instance.
    """
    # CRITICAL: Import numpy early for DuckDB threading safety.
    # The daemon owns the sole DuckDB connection; this must happen before
    # initialize() opens the database.
    # See: https://duckdb.org/docs/stable/clients/python/known_issues.html
    try:
        import numpy  # noqa: F401
    except ImportError:
        pass

    from chunkhound.daemon.server import ChunkHoundDaemon

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
    await daemon.run()


__all__: list[str] = ["daemon_command"]
