#!/usr/bin/env python3
"""
Check for duplicated chunks by ID in a LanceDB database and compare their content.

Usage:
    uv run scripts/check_lancedb_duplicates.py --db-path /path/to/database.lancedb

This script connects to a LanceDB database, identifies chunks with duplicate IDs,
compares their content, and generates a detailed report.

The report includes:
- General information: DB path, start time, total chunks, total duplicates, non-identical duplicates
- Detailed comparison of duplicate chunks

Output: Report filename and general information summary.
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def connect_to_lancedb(db_path: Path):
    """Connect to LanceDB database."""
    import lancedb

    abs_db_path = db_path.absolute()

    # Save CWD (thread-safe)
    original_cwd = os.getcwd()
    try:
        os.chdir(abs_db_path.parent)
        conn = lancedb.connect(abs_db_path.name)
        return conn
    finally:
        os.chdir(original_cwd)


def get_all_chunks_paginated(chunks_table, batch_size: int = 5000) -> List[Dict[str, Any]]:
    """Get all chunks from the table using pagination to handle large datasets."""
    all_chunks = []
    offset = 0

    while True:
        try:
            if offset == 0:
                # First batch
                batch_results = chunks_table.search().limit(batch_size).to_list()
            else:
                # Subsequent batches - get more results and slice
                extended_results = chunks_table.search().limit(offset + batch_size).to_list()
                batch_results = extended_results[offset:offset + batch_size]

            if not batch_results:
                break

            all_chunks.extend(batch_results)

            # Progress reporting for large datasets
            if len(all_chunks) > batch_size:
                print(f"Loaded {len(all_chunks)} chunks...", flush=True)

            # If we got fewer results than requested, we're done
            if len(batch_results) < batch_size:
                break

            offset += batch_size

        except Exception as e:
            raise RuntimeError(f"Could not load chunks batch at offset {offset}: {e}")

    return all_chunks


def find_duplicates(chunks: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Group chunks by ID and return only those with duplicates."""
    grouped = defaultdict(list)
    for chunk in chunks:
        chunk_id = chunk.get("id")
        if chunk_id is not None:
            grouped[chunk_id].append(chunk)

    # Return only groups with more than one chunk
    duplicates = {chunk_id: group for chunk_id, group in grouped.items() if len(group) > 1}
    return duplicates


def compare_chunk_content(chunk1: Dict[str, Any], chunk2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two chunks and return comparison details."""
    content1 = chunk1.get("content", "")
    content2 = chunk2.get("content", "")

    is_identical = content1 == content2

    comparison = {
        "identical": is_identical,
        "content_length_1": len(content1),
        "content_length_2": len(content2),
        "chunk1": {
            "id": chunk1.get("id"),
            "file_id": chunk1.get("file_id"),
            "start_line": chunk1.get("start_line"),
            "end_line": chunk1.get("end_line"),
            "chunk_type": chunk1.get("chunk_type"),
            "language": chunk1.get("language"),
            "name": chunk1.get("name"),
            "has_embedding": chunk1.get("embedding") is not None,
            "provider": chunk1.get("provider"),
            "model": chunk1.get("model"),
            "embedding_status": chunk1.get("embedding_status"),
            "created_time": chunk1.get("created_time"),
        },
        "chunk2": {
            "id": chunk2.get("id"),
            "file_id": chunk2.get("file_id"),
            "start_line": chunk2.get("start_line"),
            "end_line": chunk2.get("end_line"),
            "chunk_type": chunk2.get("chunk_type"),
            "language": chunk2.get("language"),
            "name": chunk2.get("name"),
            "has_embedding": chunk2.get("embedding") is not None,
            "provider": chunk2.get("provider"),
            "model": chunk2.get("model"),
            "embedding_status": chunk2.get("embedding_status"),
            "created_time": chunk2.get("created_time"),
        }
    }

    if not is_identical:
        # Show first 200 chars of each content for diff
        comparison["content_preview_1"] = content1[:200] + ("..." if len(content1) > 200 else "")
        comparison["content_preview_2"] = content2[:200] + ("..." if len(content2) > 200 else "")

    return comparison


def generate_report(db_path: Path, start_time: datetime, total_chunks: int, duplicates: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Generate the complete report."""
    total_duplicates = sum(len(group) for group in duplicates.values())
    duplicate_ids = len(duplicates)

    # Analyze duplicates
    duplicate_comparisons = []
    non_identical_count = 0

    for chunk_id, group in duplicates.items():
        # Compare each pair in the group
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                comparison = compare_chunk_content(group[i], group[j])
                duplicate_comparisons.append({
                    "chunk_id": chunk_id,
                    "group_size": len(group),
                    **comparison
                })
                if not comparison["identical"]:
                    non_identical_count += 1

    report = {
        "general_info": {
            "db_path": str(db_path),
            "start_time": start_time.isoformat(),
            "total_chunks": total_chunks,
            "total_duplicates": total_duplicates,
            "duplicate_ids": duplicate_ids,
            "total_non_identical_duplicates": non_identical_count,
        },
        "duplicate_details": duplicate_comparisons
    }

    return report


def save_report(report: Dict[str, Any], db_path: Path) -> str:
    """Save the report to a file and return the filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lancedb_duplicates_report_{timestamp}.json"
    filepath = db_path.parent / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return str(filepath)


def main():
    parser = argparse.ArgumentParser(description="Check for duplicated chunks in LanceDB database")
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
        start_time = datetime.now()

        # Connect to database
        print(f"Connecting to LanceDB at {args.db_path}...")
        conn = connect_to_lancedb(args.db_path)

        # Open chunks table
        try:
            chunks_table = conn.open_table("chunks")
        except Exception as e:
            raise RuntimeError(f"Could not open chunks table: {e}")

        # Get all chunks
        print("Loading all chunks from database...")
        all_chunks = get_all_chunks_paginated(chunks_table)
        total_chunks = len(all_chunks)
        print(f"Loaded {total_chunks} chunks total")

        # Find duplicates
        print("Analyzing for duplicates...")
        duplicates = find_duplicates(all_chunks)
        duplicate_ids = len(duplicates)

        if duplicate_ids == 0:
            print("No duplicate chunk IDs found.")
            return 0

        print(f"Found {duplicate_ids} chunk IDs with duplicates")

        # Generate report
        print("Generating report...")
        report = generate_report(args.db_path, start_time, total_chunks, duplicates)

        # Save report
        report_file = save_report(report, args.db_path)

        # Output summary
        general = report["general_info"]
        print(f"\nReport saved to: {report_file}")
        print(f"Database: {general['db_path']}")
        print(f"Start time: {general['start_time']}")
        print(f"Total chunks: {general['total_chunks']:,}")
        print(f"Total duplicates: {general['total_duplicates']:,}")
        print(f"Duplicate IDs: {general['duplicate_ids']:,}")
        print(f"Non-identical duplicates: {general['total_non_identical_duplicates']:,}")

        return 0

    except Exception as e:
        print(f"Error checking duplicates: {e}")
        return 1


if __name__ == "__main__":
    exit(main())