"""Parser for migrate command."""

import argparse
from pathlib import Path


def add_migrate_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add migrate subcommand parser.

    Args:
        subparsers: Subparsers object to add to
    """
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Migrate databases to global mode",
        description="Tools for discovering and migrating per-repo databases to global mode",
    )

    migrate_subparsers = migrate_parser.add_subparsers(
        dest="migrate_command",
        help="Migration subcommands",
        required=True,
    )

    # migrate discover
    discover_parser = migrate_subparsers.add_parser(
        "discover",
        help="Find existing per-repo databases",
        description="Scan filesystem for .chunkhound/db directories and validate them",
    )
    discover_parser.add_argument(
        "--search-path",
        type=Path,
        default=Path.home(),
        help="Root path to search (default: $HOME)",
    )
    discover_parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum directory depth (default: 5)",
    )
    discover_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show verbose output"
    )

    # migrate to-global
    migrate_to_global_parser = migrate_subparsers.add_parser(
        "to-global",
        help="Migrate database to global mode",
        description="Copy per-repo database to global database with validation",
    )
    migrate_to_global_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source database path (.chunkhound/db/chunks.db)",
    )
    migrate_to_global_parser.add_argument(
        "--global-db",
        type=Path,
        default=Path.home() / ".chunkhound" / "global" / "db" / "chunks.db",
        help="Global database path (default: ~/.chunkhound/global/db/chunks.db)",
    )
    migrate_to_global_parser.add_argument(
        "--base-dir",
        type=Path,
        help="Base directory (auto-detected if omitted)",
    )
    migrate_to_global_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without modifying",
    )
    migrate_to_global_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show verbose output"
    )
