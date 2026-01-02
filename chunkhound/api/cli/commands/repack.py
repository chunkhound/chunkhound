"""Repack command module - performs full database compaction."""

import argparse
import shutil
import sys

from loguru import logger

from chunkhound.core.config.config import Config

from ..utils.database import verify_database_exists
from ..utils.rich_output import RichOutputFormatter


async def repack_command(args: argparse.Namespace, config: Config) -> None:
    """Execute the repack command to fully compact the database.

    Unlike CHECKPOINT (which only partially reclaims space when ~25% of rows
    are deleted in adjacent row groups), repack copies to a fresh database
    file to guarantee complete space reclamation.

    Args:
        args: Parsed command-line arguments
        config: Pre-validated configuration instance
    """
    formatter = RichOutputFormatter(verbose=args.verbose)

    formatter.section_header("ChunkHound Database Repack")

    # Verify database exists
    try:
        verify_database_exists(config)
    except FileNotFoundError as e:
        formatter.error(str(e))
        sys.exit(1)
    except ValueError as e:
        formatter.error(str(e))
        sys.exit(1)

    # Get actual database path (includes provider-specific suffix like .duckdb)
    db_path = config.database.get_db_path()

    # Check provider type - repack only supported for DuckDB
    if config.database.provider != "duckdb":
        formatter.error(
            f"Repack is only supported for DuckDB databases. "
            f"Current provider: {config.database.provider}"
        )
        sys.exit(1)

    # Get original size
    try:
        original_size = db_path.stat().st_size
    except OSError as e:
        formatter.error(f"Failed to read database size: {e}")
        sys.exit(1)

    formatter.info(f"Database: {db_path}")
    formatter.info(f"Current size: {_format_size(original_size)}")

    # Dry-run mode - just show stats
    if getattr(args, "dry_run", False):
        # Show storage stats if available
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider

        provider = DuckDBProvider(str(db_path), base_directory=db_path.parent)
        provider.connect()
        try:
            stats = provider.get_storage_stats()
            free_blocks = stats.get("free_blocks", 0)
            total_blocks = max(stats.get("total_blocks", 1), 1)
            block_size = stats.get("block_size", 262144)
            reclaimable = free_blocks * block_size
            free_ratio = free_blocks / total_blocks

            formatter.info(f"Free space ratio: {free_ratio:.1%}")
            formatter.info(f"Reclaimable: {_format_size(reclaimable)}")
            if free_ratio >= 0.5:
                formatter.info("Compaction recommended (free ratio >= 50%)")
            else:
                formatter.info("Compaction not needed (free ratio < 50%)")
        finally:
            provider.disconnect()

        formatter.info("Use without --dry-run to repack")
        return

    # Handle backup before optimization (provider deletes original)
    backup_path = db_path.with_suffix(".duckdb.bak")
    keep_backup = getattr(args, "backup", False)

    if keep_backup:
        formatter.progress_indicator("Creating backup...")
        try:
            shutil.copy2(db_path, backup_path)
            formatter.info(f"Backup saved: {backup_path}")
        except OSError as e:
            formatter.error(f"Failed to create backup: {e}")
            sys.exit(1)

    # Use provider for compaction (eliminates code duplication)
    from chunkhound.providers.database.duckdb_provider import DuckDBProvider

    provider = DuckDBProvider(str(db_path), base_directory=db_path.parent)

    try:
        formatter.progress_indicator("Connecting to database...")
        provider.connect()

        formatter.progress_indicator("Compacting database...")
        success = provider.optimize()

        if not success:
            formatter.warning("Optimization skipped (database already optimal)")

        # Report results
        new_size = db_path.stat().st_size
        reduction = original_size - new_size
        reduction_pct = (reduction / original_size) * 100 if original_size > 0 else 0

        formatter.success("Repack complete!")
        formatter.info(f"Before: {_format_size(original_size)}")
        formatter.info(f"After:  {_format_size(new_size)}")

        if reduction > 0:
            formatter.info(f"Saved:  {_format_size(reduction)} ({reduction_pct:.1f}%)")
        elif reduction < 0:
            formatter.warning(
                f"Size increased by {_format_size(-reduction)} "
                "(this can happen if indexes were rebuilt)"
            )
        else:
            formatter.info("No size change (database was already optimally packed)")

    except Exception as e:
        formatter.error(f"Repack failed: {e}")
        logger.exception("Full error details:")

        # Clean up backup if we created one and repack failed
        if keep_backup and backup_path.exists():
            formatter.info(f"Backup retained at: {backup_path}")

        sys.exit(1)

    finally:
        if provider.is_connected:
            provider.disconnect()


def _format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string (e.g., "1.23 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.2f} MB"
    else:
        return f"{size_bytes / 1024 / 1024 / 1024:.2f} GB"
