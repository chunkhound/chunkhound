"""Repack command module - performs full database compaction."""

import argparse
import shutil
import sys

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.core.exceptions import CompactionError
from chunkhound.core.utils import format_size
from chunkhound.registry import configure_registry, get_provider
from chunkhound.services.compaction_service import estimate_reclaimable_bytes

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
    except (FileNotFoundError, ValueError) as e:
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
    formatter.info(f"Current size: {format_size(original_size)}")

    # Dry-run mode - just show stats
    if args.dry_run:
        # Show storage stats if available
        configure_registry(config)
        provider = get_provider("database")
        try:
            stats = provider.get_storage_stats()
            free_ratio = stats.get("free_ratio", 0.0)
            row_waste = stats.get("row_waste_ratio", 0.0)
            effective_waste = stats.get("effective_waste", 0.0)

            formatter.info(f"Row-group waste: {row_waste:.1%}")
            formatter.info(f"Free blocks ratio: {free_ratio:.1%}")
            formatter.info(f"Effective waste: {effective_waste:.1%}")
            reclaimable_bytes = estimate_reclaimable_bytes(stats)
            formatter.info(
                f"Estimated reclaimable: "
                f"~{format_size(reclaimable_bytes)}"
            )
            threshold = config.database.compaction_threshold
            if effective_waste >= threshold:
                formatter.info(
                    f"Above auto-compaction threshold (waste >= {threshold:.0%})"
                )
            else:
                formatter.info(
                    f"Below auto-compaction threshold "
                    f"(waste < {threshold:.0%}). "
                    f"Manual repack always runs regardless"
                )
            min_size_mb = config.database.compaction_min_size_mb
            reclaimable_mb = reclaimable_bytes / (1024 * 1024)
            if reclaimable_mb < min_size_mb:
                formatter.info(
                    f"Estimated reclaimable {reclaimable_mb:.1f}MB below "
                    f"min-size gate ({min_size_mb}MB) — "
                    f"auto-compaction would skip"
                )
        finally:
            provider.disconnect()

        formatter.info("Use without --dry-run to repack")
        return

    # Handle backup before optimization (provider deletes original)
    backup_path = db_path.with_suffix(db_path.suffix + ".bak")
    keep_backup = args.backup

    if keep_backup:
        formatter.progress_indicator("Creating backup...")
        try:
            shutil.copy2(db_path, backup_path)
            formatter.info(f"Backup saved: {backup_path}")
        except OSError as e:
            formatter.error(f"Failed to create backup: {e}")
            sys.exit(1)

    # Use registry-based provider (consistent with other CLI commands)
    configure_registry(config)
    provider = get_provider("database")

    try:
        formatter.progress_indicator("Compacting database...")
        compacted = provider.optimize()
        if not compacted:
            formatter.warning("No compaction was performed")
            return

        # Report results
        new_size = db_path.stat().st_size
        reduction = original_size - new_size
        reduction_pct = (reduction / original_size) * 100 if original_size > 0 else 0

        formatter.success("Repack complete!")
        formatter.info(f"Before: {format_size(original_size)}")
        formatter.info(f"After:  {format_size(new_size)}")

        if reduction > 0:
            formatter.info(f"Saved:  {format_size(reduction)} ({reduction_pct:.1f}%)")
        elif reduction < 0:
            formatter.warning(
                f"Size increased by {format_size(-reduction)} "
                "(this can happen if indexes were rebuilt)"
            )
        else:
            formatter.info("No size change (database was already optimally packed)")

    except CompactionError as e:
        formatter.error(str(e))
        sys.exit(1)
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
