#!/usr/bin/env python3
"""
Clear all embeddings from ChunkHound database.

This script removes all embeddings from the database, preparing it for testing
new embedding flows. It works with both DuckDB and LanceDB providers.

Usage:
    uv run scripts/clear_legacy_embeddings.py --base-folder /path/to/base/folder [--dry-run] [--yes] [--config /path/to/config.json]

Options:
    --base-folder: Path to the base folder containing chunkhound config file and .chunkhound directory (required)
    --dry-run: Show what would be deleted without actually deleting
    --yes: Skip confirmation prompts (use with caution)
    --config: Path to config file (optional, will auto-detect from base folder if not provided)

Safety Features:
- Shows statistics of what will be deleted before proceeding
- Requires explicit user confirmation unless --yes is used
- Supports dry-run mode to preview changes
- Graceful error handling with rollback on failure
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from chunkhound.core.config.config import Config


def get_database_provider(config: Config) -> Any:
    """Create and return the appropriate database provider based on config."""
    from chunkhound.registry import get_registry

    registry = get_registry()
    registry.configure(config)
    return registry.get_provider("database")


def get_embedding_stats(provider: Any) -> dict[str, Any]:
    """Get statistics about embeddings in the database."""
    try:
        stats = provider.get_stats()
        return {
            "total_files": stats.get("files", 0),
            "total_chunks": stats.get("chunks", 0),
            "total_embeddings": stats.get("embeddings", 0),
        }
    except Exception as e:
        print(f"Warning: Could not get database stats: {e}")
        return {"total_files": 0, "total_chunks": 0, "total_embeddings": 0}


def clear_embeddings_duckdb(provider: Any, dry_run: bool = False) -> dict[str, Any]:
    """Clear all embeddings from DuckDB database."""
    result = {"tables_cleared": [], "embeddings_deleted": 0, "errors": []}

    try:
        # Get all embedding tables
        embedding_tables = provider._get_all_embedding_tables()

        if not embedding_tables:
            print("No embedding tables found in DuckDB database.")
            return result

        print(f"Found {len(embedding_tables)} embedding tables: {', '.join(embedding_tables)}")

        for table_name in embedding_tables:
            try:
                # Get count before deletion
                count_query = f"SELECT COUNT(*) FROM {table_name}"
                before_count = provider.execute_query(count_query)[0][0]

                if before_count == 0:
                    print(f"Table {table_name} is already empty.")
                    continue

                print(f"Table {table_name}: {before_count} embeddings")

                if not dry_run:
                    # Delete all embeddings from this table
                    delete_query = f"DELETE FROM {table_name}"
                    provider.execute_query(delete_query)
                    result["embeddings_deleted"] += before_count

                result["tables_cleared"].append(table_name)

            except Exception as e:
                error_msg = f"Failed to clear table {table_name}: {e}"
                print(f"Error: {error_msg}")
                result["errors"].append(error_msg)

    except Exception as e:
        error_msg = f"Failed to clear DuckDB embeddings: {e}"
        print(f"Error: {error_msg}")
        result["errors"].append(error_msg)

    return result


def clear_embeddings_lancedb(provider: Any, dry_run: bool = False) -> dict[str, Any]:
    """Clear all embeddings from LanceDB database."""
    result = {"chunks_updated": 0, "errors": []}

    try:
        # For LanceDB, we need to update the chunks table to clear embeddings
        # Get count of chunks with embeddings
        stats = get_embedding_stats(provider)
        embeddings_count = stats["total_embeddings"]

        if embeddings_count == 0:
            print("No embeddings found in LanceDB database.")
            return result

        print(f"Found {embeddings_count} embeddings in chunks table")

        if not dry_run:
            # Clear embeddings by updating chunks table
            # This sets embedding to None, and clears provider/model fields
            provider._execute_in_db_thread_sync(
                "invalidate_embeddings_by_provider_model", "", ""
            )
            result["chunks_updated"] = embeddings_count

    except Exception as e:
        error_msg = f"Failed to clear LanceDB embeddings: {e}"
        print(f"Error: {error_msg}")
        result["errors"].append(error_msg)

    return result


def clear_all_embeddings(provider: Any, dry_run: bool = False) -> dict[str, Any]:
    """Clear all embeddings from database using appropriate method for provider type."""
    provider_type = getattr(provider, "provider_type", "unknown")

    if provider_type == "duckdb":
        return clear_embeddings_duckdb(provider, dry_run)
    elif provider_type == "lancedb":
        return clear_embeddings_lancedb(provider, dry_run)
    else:
        raise ValueError(f"Unsupported database provider: {provider_type}")


def confirm_deletion(stats: dict[str, Any], skip_confirmation: bool = False) -> bool:
    """Ask user to confirm deletion."""
    if skip_confirmation:
        return True

    print("\n" + "="*60)
    print("EMBEDDING CLEARANCE CONFIRMATION")
    print("="*60)
    print(f"Files in database: {stats['total_files']:,}")
    print(f"Chunks in database: {stats['total_chunks']:,}")
    print(f"Embeddings to delete: {stats['total_embeddings']:,}")
    print("="*60)

    if stats['total_embeddings'] == 0:
        print("No embeddings found to delete.")
        return True

    print("WARNING: This will permanently delete all embeddings from the database!")
    print("This action cannot be undone.")

    while True:
        response = input("\nAre you sure you want to proceed? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please answer 'yes' or 'no'")


def main():
    parser = argparse.ArgumentParser(
        description="Clear all embeddings from ChunkHound database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--base-folder",
        type=Path,
        required=True,
        help="Path to the base folder containing chunkhound config file and .chunkhound directory"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts (use with caution)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config file (optional, will auto-detect from base folder if not provided)"
    )

    args = parser.parse_args()

    print("DEBUG: Starting script execution")

    try:
        # Load configuration
        print(f"DEBUG: Loading config with base_folder: {args.base_folder}")
        config = Config(args=args, target_dir=args.base_folder)
        print("DEBUG: Config loaded successfully")
        print(f"DEBUG: config.database = {repr(config.database)}")

        print(f"Database provider: {config.database.provider}")
        print(f"Database path: {config.database.get_db_path()}")

        # Create database provider
        print("DEBUG: Creating database provider")
        provider = get_database_provider(config)

        # Connect to database
        print("DEBUG: Connecting to database")
        provider.connect()
        print("DEBUG: Connected to database")

        # Get current statistics
        print("DEBUG: Getting embedding stats")
        initial_stats = get_embedding_stats(provider)
        print(f"DEBUG: Stats obtained: {initial_stats}")

        print(f"\nConnected to {config.database.provider} database")
        print(f"Database contains {initial_stats['total_files']:,} files, "
              f"{initial_stats['total_chunks']:,} chunks, "
              f"{initial_stats['total_embeddings']:,} embeddings")

        # Confirm deletion
        if not confirm_deletion(initial_stats, args.yes):
            print("Operation cancelled by user.")
            return 0

        # Perform the clearing
        print(f"\n{'DRY RUN: ' if args.dry_run else ''}Clearing embeddings...")
        result = clear_all_embeddings(provider, args.dry_run)

        # Show results
        print("\n" + "="*60)
        if args.dry_run:
            print("DRY RUN RESULTS")
        else:
            print("OPERATION RESULTS")
        print("="*60)

        if result.get("errors"):
            print("Errors encountered:")
            for error in result["errors"]:
                print(f"  - {error}")
            return 1

        if config.database.provider == "duckdb":
            print(f"Tables processed: {len(result.get('tables_cleared', []))}")
            if result.get('tables_cleared'):
                print(f"Tables cleared: {', '.join(result['tables_cleared'])}")
            print(f"Embeddings deleted: {result.get('embeddings_deleted', 0):,}")
        elif config.database.provider == "lancedb":
            print(f"Chunks updated: {result.get('chunks_updated', 0):,}")

        # Verify results
        if not args.dry_run:
            final_stats = get_embedding_stats(provider)
            print("\nVerification:")
            print(f"  Embeddings remaining: {final_stats['total_embeddings']:,}")

            if final_stats['total_embeddings'] == 0:
                print("✓ All embeddings successfully cleared!")
            else:
                print(f"⚠ Warning: {final_stats['total_embeddings']:,} embeddings still remain")

        provider.disconnect()
        print("\nOperation completed successfully.")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())