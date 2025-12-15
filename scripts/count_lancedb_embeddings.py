#!/usr/bin/env python3
"""
Count embeddings in a LanceDB database.

Usage:
    uv run scripts/count_lancedb_embeddings.py --db-path /path/to/database.lancedb [--validation-level LEVEL]

This script connects to a LanceDB database and counts embeddings in the chunks table.
You can choose different validation levels for speed vs accuracy trade-offs.

Validation levels:
- none: Count all non-null embeddings (fastest, no data loading required)
- basic: Count non-null + non-empty embeddings (requires loading, minimal processing)
- full: Full validation including zero-vector detection (most accurate, current default)

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


def count_embeddings(db_path: Path, validation_level: str = "full") -> int:
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

    if validation_level == "none":
        # Fastest: Just count non-null embeddings using LanceDB's native capabilities
        try:
            # LanceDB doesn't have a direct count with WHERE, so we use search and count results
            all_results = chunks_table.search().where("embedding IS NOT NULL").to_list()
            return len(all_results)
        except Exception as e:
            raise RuntimeError(f"Could not count non-null embeddings: {e}")

    # For "basic" and "full" validation, we need to load and inspect the data
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
            if offset == 0:
                # First batch
                batch_results = chunks_table.search().where("embedding IS NOT NULL").limit(batch_size).to_list()
            else:
                # Subsequent batches - get more results and slice
                extended_results = chunks_table.search().where("embedding IS NOT NULL").limit(offset + batch_size).to_list()
                batch_results = extended_results[offset:offset + batch_size]

            if not batch_results:
                break

            # Apply validation based on level
            if validation_level == "basic":
                # Basic validation: check non-null and non-empty
                batch_valid_count = sum(
                    1 for result in batch_results
                    if result.get("embedding") is not None and len(result.get("embedding", [])) > 0
                )
            elif validation_level == "full":
                # Full validation: use the complete _has_valid_embedding function
                batch_valid_count = sum(1 for result in batch_results if _has_valid_embedding(result.get("embedding")))
            else:
                raise ValueError(f"Unknown validation level: {validation_level}")

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
    parser.add_argument(
        "--validation-level",
        choices=["none", "basic", "full"],
        default="full",
        help="Level of validation to perform (none=basic, basic=non-empty, full=zero-vector check)"
    )

    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"Error: Database path {args.db_path} does not exist")
        return 1

    if not args.db_path.is_dir():
        print(f"Error: Database path {args.db_path} is not a directory")
        return 1

    try:
        count = count_embeddings(args.db_path, args.validation_level)
        level_desc = {
            "none": "non-null embeddings",
            "basic": "non-empty embeddings",
            "full": "valid embeddings"
        }
        print(f"Found {count} {level_desc[args.validation_level]} in {args.db_path}")
        return 0
    except Exception as e:
        print(f"Error counting embeddings: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
