"""Parser for daemon command."""

import argparse


def add_daemon_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add daemon subcommand parser.

    The daemon command manages the ChunkHound HTTP server lifecycle
    for global database mode.

    Args:
        subparsers: Subparsers object to add to
    """
    daemon_parser = subparsers.add_parser(
        "daemon",
        help="Manage ChunkHound daemon (HTTP server)",
        description=(
            "Manage the ChunkHound daemon for global database mode. "
            "The daemon runs an HTTP MCP server that handles file watching "
            "and enables multiple concurrent client connections."
        ),
    )

    daemon_subparsers = daemon_parser.add_subparsers(
        dest="daemon_command",
        help="Daemon subcommands",
        required=True,
    )

    # Common arguments for all daemon commands
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    common_args.add_argument(
        "--port",
        type=int,
        default=5173,
        help="Port to listen on (default: 5173)",
    )
    common_args.add_argument(
        "-v", "--verbose", action="store_true", help="Show verbose output"
    )

    # daemon start
    start_parser = daemon_subparsers.add_parser(
        "start",
        help="Start the daemon",
        description=(
            "Start the ChunkHound HTTP daemon. By default runs in foreground; "
            "use --background to daemonize."
        ),
        parents=[common_args],
    )
    start_parser.add_argument(
        "--background",
        "-b",
        action="store_true",
        help="Run daemon in background (daemonize)",
    )

    # daemon stop
    daemon_subparsers.add_parser(
        "stop",
        help="Stop the daemon",
        description="Stop the running ChunkHound daemon.",
        parents=[common_args],
    )

    # daemon status
    daemon_subparsers.add_parser(
        "status",
        help="Show daemon status",
        description="Display status information about the ChunkHound daemon.",
        parents=[common_args],
    )

    # daemon restart
    daemon_subparsers.add_parser(
        "restart",
        help="Restart the daemon",
        description="Stop and start the ChunkHound daemon.",
        parents=[common_args],
    )

    # daemon logs
    logs_parser = daemon_subparsers.add_parser(
        "logs",
        help="Show daemon logs",
        description="Display logs from the ChunkHound daemon.",
        parents=[common_args],
    )
    logs_parser.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Follow log output (like tail -f)",
    )
    logs_parser.add_argument(
        "--lines",
        "-n",
        type=int,
        default=100,
        help="Number of lines to show (default: 100)",
    )

    # daemon refresh-watchers
    refresh_parser = daemon_subparsers.add_parser(
        "refresh-watchers",
        help="Refresh file watchers for all projects",
        description=(
            "Restart file watchers for all indexed projects. "
            "Use this if watchers have become stale or stopped responding."
        ),
        parents=[common_args],
    )
    refresh_parser.add_argument(
        "--project",
        "-p",
        type=str,
        help="Refresh watcher for specific project only",
    )
