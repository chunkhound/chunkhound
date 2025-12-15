#!/usr/bin/env python3
"""
Count embeddings in a LanceDB database.

Usage:
    uv run scripts/count_lancedb_embeddings.py --db-path /path/to/database.lancedb

This script connects to a LanceDB database and counts the number of valid embeddings
in the chunks table. It uses the same validation logic as the ChunkHound LanceDB provider.

For large datasets, it processes embeddings in batches to avoid memory issues.
Progress is reported for datasets larger than 5000 chunks.
"""

import argparse
import os
from pathlib import Path

import numpy as np


def _has_valid_embedding(x) -> bool:
    """Check if embedding is valid (not None, not empty, not all zeros).

    Handles both list and numpy array embeddings. Zero-vector detection
    provides defense-in-depth for legacy placeholder vectors.
    """
    if not hasattr(x, "__len__"):
        return False
    if x is None or not isinstance(x, (list, np.ndarray)) or len(x) == 0:
        return False
    # Check not all zeros (legacy placeholder detection)
    if isinstance(x, np.ndarray):
        return np.any(x != 0)
    return any(v != 0 for v in x)


def count_embeddings(db_path: Path) -> int:
    """Count valid embeddings in the LanceDB chunks table using memory-efficient pagination."""
    import lancedb

    abs_db_path = db_path.absolute()

    # Save CWD (thread-safe)
    original_cwd = os.getcwd()
    try:
        os.chdir(abs_db_path.parent)
        conn = lancedb.connect(abs_db_path.name)
    finally:
        os.chdir(original_cwd)

    # Open chunks table
    try:
        chunks_table = conn.open_table("chunks")
    except Exception as e:
        raise RuntimeError(f"Could not open chunks table: {e}")

    # Get total row count for progress tracking
    try:
        total_rows = chunks_table.count_rows()
    except Exception:
        total_rows = None

    # Process in batches to avoid memory issues with large datasets
    batch_size = 5000  # Process 5000 rows at a time
    valid_count = 0
    offset = 0

    while True:
        try:
            # Use pagination to load chunks in batches
            # Note: LanceDB's to_pandas() may not support offset/limit consistently,
            # so we use search().limit() and manual offset handling
            if offset == 0:
                # First batch
                batch_results = chunks_table.search().where("embedding IS NOT NULL").limit(batch_size).to_list()
            else:
                # Subsequent batches - get more results and slice
                # This is less efficient but works around LanceDB's lack of native offset support
                extended_results = chunks_table.search().where("embedding IS NOT NULL").limit(offset + batch_size).to_list()
                batch_results = extended_results[offset:offset + batch_size]

            if not batch_results:
                break

            # Filter for valid embeddings in this batch
            batch_valid_count = sum(1 for result in batch_results if _has_valid_embedding(result.get("embedding")))
            valid_count += batch_valid_count

            # If we got fewer results than requested, we're done
            if len(batch_results) < batch_size:
                break

            offset += batch_size

            # Progress reporting for large datasets
            if total_rows and total_rows > batch_size:
                progress = min(offset + len(batch_results), total_rows)
                print(f"Processed {progress}/{total_rows} chunks... ({valid_count} valid embeddings so far)", flush=True)

        except Exception as e:
            raise RuntimeError(f"Could not process embeddings batch at offset {offset}: {e}")

    return valid_count


def main():
    parser = argparse.ArgumentParser(description="Count embeddings in LanceDB database")
    parser.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="Path to the LanceDB database directory (.lancedb)"
    )

    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"Error: Database path {args.db_path} does not exist")
        return 1

    if not args.db_path.is_dir():
        print(f"Error: Database path {args.db_path} is not a directory")
        return 1

    try:
        count = count_embeddings(args.db_path)
        print(f"Found {count} valid embeddings in {args.db_path}")
        return 0
    except Exception as e:
        print(f"Error counting embeddings: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
