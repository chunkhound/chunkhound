#!/usr/bin/env python3
"""
Analyze LanceDB chunks table schema and provide statistics on embedding fields.

This script analyzes an existing LanceDB database to understand the current state
of embeddings, status fields, and signatures before deciding on reindexing strategies.

Usage:
    python scripts/analyze_lancedb_chunks.py /path/to/database.lancedb

Outputs:
- Total chunks count
- Unique values and counts for embedding_status field
- Unique values and counts for embedding_signature field
- Counts of chunks with null vs non-null embeddings
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

try:
    import lancedb
    import pandas as pd
except ImportError as e:
    print(f"Error: Missing required packages. Please install: {e}")
    sys.exit(1)


def collect_unique_values(table, field_name: str, batch_size: int = 10000) -> set:
    """Collect unique values for a field using batch processing to avoid memory issues."""
    unique_values = set()
    offset = 0

    while True:
        try:
            # Get batch of data
            batch = table.search().limit(batch_size).offset(offset).to_pandas()
            if batch.empty:
                break

            # Extract unique values from this batch
            if field_name in batch.columns:
                batch_values = batch[field_name].dropna().unique()
                unique_values.update(batch_values)

            offset += batch_size

            # Safety check to prevent infinite loops
            if offset > 10_000_000:  # 10M safety limit
                print(f"Warning: Reached safety limit while collecting unique values for {field_name}")
                break

        except Exception as e:
            print(f"Error collecting unique values for {field_name}: {e}")
            break

    return unique_values


def count_field_values(table, field_name: str, unique_values: set) -> dict[str, int]:
    """Count occurrences of each unique value in a field."""
    counts = {}

    for value in unique_values:
        try:
            # Use count_rows with filter for efficient counting
            filter_expr = f"{field_name} = '{value}'"
            count = table.count_rows(filter=filter_expr)
            counts[str(value)] = count
        except Exception as e:
            print(f"Error counting {field_name} = '{value}': {e}")
            counts[str(value)] = 0

    return counts


def analyze_chunks_table(db_path: str | Path) -> None:
    """Analyze the chunks table in the LanceDB database."""
    db_path = Path(db_path)

    if not db_path.exists():
        print(f"Error: Database path does not exist: {db_path}")
        return

    try:
        # Connect to LanceDB
        print(f"Connecting to LanceDB at: {db_path}")
        conn = lancedb.connect(str(db_path))

        # Open chunks table
        try:
            chunks_table = conn.open_table("chunks")
            print("Opened chunks table successfully")
        except Exception as e:
            print(f"Error opening chunks table: {e}")
            return

        # Get total count
        try:
            total_chunks = chunks_table.count_rows()
            print(f"\nTotal chunks: {total_chunks:,}")
        except Exception as e:
            print(f"Error getting total count: {e}")
            total_chunks = 0

        if total_chunks == 0:
            print("No chunks found in database.")
            return

        # Analyze embedding_status field
        print("\n=== EMBEDDING_STATUS ANALYSIS ===")
        try:
            unique_statuses = collect_unique_values(chunks_table, "embedding_status")
            if unique_statuses:
                status_counts = count_field_values(chunks_table, "embedding_status", unique_statuses)
                print(f"Unique embedding_status values: {len(unique_statuses)}")
                for status, count in sorted(status_counts.items()):
                    percentage = (count / total_chunks) * 100 if total_chunks > 0 else 0
                    print(f"  {status}: {count:,} ({percentage:.1f}%)")
            else:
                print("No embedding_status values found")
        except Exception as e:
            print(f"Error analyzing embedding_status: {e}")

        # Analyze embedding_signature field
        print("\n=== EMBEDDING_SIGNATURE ANALYSIS ===")
        try:
            unique_signatures = collect_unique_values(chunks_table, "embedding_signature")
            if unique_signatures:
                signature_counts = count_field_values(chunks_table, "embedding_signature", unique_signatures)
                print(f"Unique embedding_signature values: {len(unique_signatures)}")
                for signature, count in sorted(signature_counts.items()):
                    percentage = (count / total_chunks) * 100 if total_chunks > 0 else 0
                    print(f"  {signature}: {count:,} ({percentage:.1f}%)")
            else:
                print("No embedding_signature values found")
        except Exception as e:
            print(f"Error analyzing embedding_signature: {e}")

        # Analyze embedding field (null vs not null)
        print("\n=== EMBEDDING FIELD ANALYSIS ===")
        try:
            null_count = chunks_table.count_rows(filter="embedding IS NULL")
            not_null_count = chunks_table.count_rows(filter="embedding IS NOT NULL")

            print(f"Chunks with NULL embeddings: {null_count:,} ({(null_count / total_chunks) * 100:.1f}%)")
            print(f"Chunks with NON-NULL embeddings: {not_null_count:,} ({(not_null_count / total_chunks) * 100:.1f}%)")

            # Verify counts add up
            total_calculated = null_count + not_null_count
            if total_calculated != total_chunks:
                print(f"Warning: Calculated total ({total_calculated:,}) != actual total ({total_chunks:,})")

        except Exception as e:
            print(f"Error analyzing embedding field: {e}")

        print("\n=== ANALYSIS COMPLETE ===")

    except Exception as e:
        print(f"Error connecting to database: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LanceDB chunks table for embedding statistics"
    )
    parser.add_argument(
        "db_path",
        help="Path to the LanceDB database directory (.lancedb)"
    )

    args = parser.parse_args()
    analyze_chunks_table(args.db_path)


if __name__ == "__main__":
    main()