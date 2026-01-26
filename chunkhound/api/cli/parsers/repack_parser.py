"""Repack command argument parser for ChunkHound CLI."""

import argparse
from typing import Any, cast

from .common_arguments import add_common_arguments, add_config_arguments


def add_repack_subparser(subparsers: Any) -> argparse.ArgumentParser:
    """Add repack command subparser to the main parser.

    Args:
        subparsers: Subparsers object from the main argument parser

    Returns:
        The configured repack subparser
    """
    repack_parser = subparsers.add_parser(
        "repack",
        help="Fully compact the database to reclaim disk space",
        description=(
            "Performs full database compaction by copying to a fresh database "
            "file. Unlike CHECKPOINT (which only partially reclaims space), "
            "repack guarantees complete space reclamation by rebuilding the "
            "entire database."
        ),
    )

    repack_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show current database size without repacking",
    )

    repack_parser.add_argument(
        "--backup",
        action="store_true",
        help="Keep backup of original database as .duckdb.bak",
    )

    # Add common arguments (verbose, config, debug)
    add_common_arguments(repack_parser)

    # Add database config arguments (--db, etc.)
    add_config_arguments(repack_parser, ["database"])

    return cast(argparse.ArgumentParser, repack_parser)


__all__: list[str] = ["add_repack_subparser"]
