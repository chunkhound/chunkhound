"""DuckDB compaction utilities - shared by provider and tests.

Provides threshold-based compaction decision logic and EXPORT/IMPORT/SWAP
cycle for reclaiming disk space after deletions.
"""

import shutil
from pathlib import Path
from typing import Any

import duckdb
from loguru import logger

# Default thresholds (from database_config.py)
DEFAULT_COMPACTION_THRESHOLD = 0.5  # Free blocks ratio
DEFAULT_COMPACTION_MIN_SIZE_MB = 100  # Minimum reclaimable MB
DEFAULT_UTILIZATION_THRESHOLD = 0.5  # Row group utilization threshold (50%)

# DuckDB storage constants
ROW_GROUP_SIZE = 122_880  # DuckDB default row group size
MIN_ROW_GROUPS_FOR_UTILIZATION_CHECK = 10  # Skip utilization check for tiny DBs


def get_storage_stats(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Get DuckDB storage statistics for compaction decisions.

    Args:
        conn: Active DuckDB connection

    Returns:
        Dict with block_size, total_blocks, used_blocks, free_blocks
    """
    result = conn.execute("CALL pragma_database_size()").fetchone()
    if result is None:
        # Should never happen for valid database, but handle gracefully
        return {
            "block_size": 262144,  # Default DuckDB block size
            "total_blocks": 0,
            "used_blocks": 0,
            "free_blocks": 0,
        }
    # (database_name, database_size, block_size, total_blocks, used_blocks, free_blocks)
    return {
        "block_size": result[2],
        "total_blocks": result[3],
        "used_blocks": result[4],
        "free_blocks": result[5],
    }


def get_row_group_utilization(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Calculate row group utilization across all embeddings tables.

    DuckDB's free_blocks metric doesn't reflect logical deletions - deleted rows
    remain in row groups until EXPORT/IMPORT. This function compares logical row
    count to physical storage capacity to detect bloated tables.

    Args:
        conn: Active DuckDB connection

    Returns:
        Dict with:
        - total_row_groups: Physical row groups in storage
        - total_logical_rows: Actual row count
        - physical_capacity: row_groups * ROW_GROUP_SIZE
        - utilization: logical_rows / physical_capacity (0.0-1.0)
    """
    tables = conn.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_name LIKE 'embeddings_%'
    """).fetchall()

    total_row_groups = 0
    total_logical_rows = 0

    for (table_name,) in tables:
        # Get row group count from storage info
        result = conn.execute(f"""
            SELECT COUNT(DISTINCT row_group_id)
            FROM pragma_storage_info('{table_name}')
        """).fetchone()
        row_groups = result[0] if result else 0

        # Get logical row count
        logical_rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

        total_row_groups += row_groups
        total_logical_rows += logical_rows

    physical_capacity = total_row_groups * ROW_GROUP_SIZE
    utilization = total_logical_rows / physical_capacity if physical_capacity > 0 else 1.0

    return {
        "total_row_groups": total_row_groups,
        "total_logical_rows": total_logical_rows,
        "physical_capacity": physical_capacity,
        "utilization": utilization,
    }


def should_compact(
    conn: duckdb.DuckDBPyConnection,
    threshold: float = DEFAULT_COMPACTION_THRESHOLD,
    min_size_mb: int = DEFAULT_COMPACTION_MIN_SIZE_MB,
    utilization_threshold: float = DEFAULT_UTILIZATION_THRESHOLD,
) -> tuple[bool, dict[str, Any]]:
    """Check if compaction is warranted based on free blocks ratio or row group utilization.

    Three-stage decision logic:
    - Stage 1+2: free_ratio >= threshold AND reclaimable >= min_size_mb
    - Stage 3: row_group_utilization < utilization_threshold (catches post-DELETE bloat)

    Compaction triggers if Stage 1+2 pass OR Stage 3 passes.

    Args:
        conn: Active DuckDB connection
        threshold: Free blocks ratio threshold (0.0-1.0)
        min_size_mb: Minimum reclaimable space in MB
        utilization_threshold: Row group utilization threshold (0.0-1.0)

    Returns:
        Tuple of (should_compact, storage_stats)
    """
    stats = get_storage_stats(conn)
    total = stats["total_blocks"]
    free = stats["free_blocks"]

    # Stage 1: Check free ratio threshold
    free_ratio = free / total if total > 0 else 0.0
    if free_ratio >= threshold:
        # Stage 2: Check minimum reclaimable size
        reclaimable = free * stats["block_size"]
        if reclaimable >= min_size_mb * 1024 * 1024:
            return True, stats

    # Stage 3: Check row group utilization for post-DELETE scenarios
    # DuckDB's free_blocks doesn't reflect logical deletions, so we compare
    # logical row count to physical storage capacity
    utilization_stats = get_row_group_utilization(conn)
    stats.update(utilization_stats)

    if utilization_stats["total_row_groups"] >= MIN_ROW_GROUPS_FOR_UTILIZATION_CHECK:
        if utilization_stats["utilization"] < utilization_threshold:
            logger.debug(
                f"Compaction triggered by low utilization: "
                f"{utilization_stats['utilization']:.1%} < {utilization_threshold:.0%}"
            )
            return True, stats

    return False, stats


def compact_database(
    db_path: Path,
    conn: duckdb.DuckDBPyConnection,
    threshold: float = DEFAULT_COMPACTION_THRESHOLD,
    min_size_mb: int = DEFAULT_COMPACTION_MIN_SIZE_MB,
    utilization_threshold: float = DEFAULT_UTILIZATION_THRESHOLD,
) -> tuple[bool, duckdb.DuckDBPyConnection]:
    """Run EXPORT/IMPORT/SWAP compaction cycle if thresholds exceeded.

    Performs full database compaction via DuckDB's EXPORT/IMPORT mechanism,
    which creates a fresh database file with no fragmentation.

    Args:
        db_path: Path to the DuckDB database file
        conn: Active DuckDB connection (will be closed and replaced)
        threshold: Free blocks ratio threshold (0.0-1.0)
        min_size_mb: Minimum reclaimable space in MB
        utilization_threshold: Row group utilization threshold (0.0-1.0)

    Returns:
        Tuple of (success, new_connection) - caller must use returned connection
    """
    # Check thresholds first
    should, stats = should_compact(conn, threshold, min_size_mb, utilization_threshold)
    if not should:
        return False, conn

    free_ratio = stats["free_blocks"] / max(stats["total_blocks"], 1)
    reclaimable_mb = stats["free_blocks"] * stats["block_size"] / 1024 / 1024

    # Log trigger reason
    if "utilization" in stats and stats["utilization"] < utilization_threshold:
        logger.info(
            f"Compacting database: {stats['utilization']:.1%} row group utilization "
            f"({stats['total_logical_rows']:,} rows in {stats['total_row_groups']} row groups)"
        )
    else:
        logger.info(
            f"Compacting database: {free_ratio:.0%} free blocks, "
            f"{reclaimable_mb:.1f}MB reclaimable"
        )

    # Checkpoint before disconnect
    conn.execute("CHECKPOINT")
    conn.close()

    export_dir = db_path.parent / ".chunkhound_compaction_export"
    new_db = db_path.with_suffix(".compact.duckdb")
    old_db = db_path.with_suffix(".duckdb.old")

    try:
        # Export with read-only connection
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_conn = duckdb.connect(str(db_path), read_only=True)
        export_conn.execute(f"EXPORT DATABASE '{export_dir}' (FORMAT PARQUET)")
        export_conn.close()

        # Import into fresh database
        if new_db.exists():
            new_db.unlink()
        import_conn = duckdb.connect(str(new_db))
        import_conn.execute(f"IMPORT DATABASE '{export_dir}'")
        import_conn.close()

        # Atomic swap
        if old_db.exists():
            old_db.unlink()
        db_path.rename(old_db)
        new_db.rename(db_path)
        old_db.unlink()

        logger.info(f"Compaction complete: {db_path}")
        # Return new connection
        return True, duckdb.connect(str(db_path))

    except Exception as e:
        logger.error(f"Compaction failed: {e}")
        # Recovery: restore original if possible
        if old_db.exists() and not db_path.exists():
            old_db.rename(db_path)
        if new_db.exists():
            new_db.unlink(missing_ok=True)
        return False, duckdb.connect(str(db_path))

    finally:
        if export_dir.exists():
            shutil.rmtree(export_dir, ignore_errors=True)
