"""Helper utilities for simulating LanceDB fragmentation in tests.

LanceDB stores data in fragments (separate files). Each merge_insert operation
creates a new fragment until compaction runs (threshold: 100 operations by default).

This module provides utilities to create controlled fragmentation states for
testing deduplication behavior.
"""

import asyncio
from pathlib import Path
from typing import Any

from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language


def create_fragmented_state(
    provider: Any,
    num_fragments: int,
    base_path: Path,
) -> list[File]:
    """Create exact number of fragments by inserting files in small batches.

    Each file insertion triggers merge_insert, creating a new fragment.

    Args:
        provider: LanceDB provider instance
        num_fragments: Number of fragments to create
        base_path: Base directory for test files

    Returns:
        List of File objects that were inserted
    """
    files = []

    for i in range(num_fragments):
        # Create unique test file
        test_file = File(
            path=f"fragment_test_{i}.py",
            mtime=1234567890.0 + i,
            language=Language.PYTHON,
            size_bytes=100,
        )

        # Insert file (creates fragment)
        file_id = provider.insert_file(test_file)

        # Insert one chunk for this file
        chunk = Chunk(
            file_id=file_id,
            code=f"def fragment_{i}(): return {i}",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol=f"fragment_{i}",
        )
        provider.insert_chunk(chunk)

        # Create new File instance with id for return list
        file_with_id = File(
            id=file_id,
            path=f"fragment_test_{i}.py",
            mtime=1234567890.0 + i,
            language=Language.PYTHON,
            size_bytes=100,
        )
        files.append(file_with_id)

    return files


def create_file_with_duplicates_across_fragments(
    provider: Any,
    file_path: str,
    num_updates: int,
) -> int:
    """Simulate file being updated multiple times, creating fragments.

    Each update creates a new fragment with same chunk_id (if content unchanged)
    or different chunk_id (if content changed).

    Args:
        provider: LanceDB provider instance
        file_path: Path for the test file
        num_updates: Number of times to update the file

    Returns:
        file_id of the created/updated file
    """
    # Initial file insertion
    test_file = File(
        path=file_path,
        mtime=1234567890.0,
        language=Language.PYTHON,
        size_bytes=100,
    )
    file_id = provider.insert_file(test_file)

    # Initial chunk
    chunk = Chunk(
        file_id=file_id,
        code="def test_function(): return 42",
        start_line=1,
        end_line=1,
        chunk_type=ChunkType.FUNCTION,
        language=Language.PYTHON,
        symbol="test_function",
    )
    provider.insert_chunk(chunk)

    # Update file multiple times
    for i in range(1, num_updates):
        # Update file metadata (new mtime triggers re-processing)
        updated_file = File(
            id=file_id,
            path=file_path,
            mtime=1234567890.0 + i,
            language=Language.PYTHON,
            size_bytes=100 + i,
        )
        provider.insert_file(updated_file)  # merge_insert creates new fragment

        # Re-insert chunk (merge_insert on chunk_id)
        updated_chunk = Chunk(
            file_id=file_id,
            code=f"def test_function(): return {42 + i}",  # Modified content
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol="test_function",
        )
        provider.insert_chunk(updated_chunk)

    return file_id


async def create_fragmented_state_async(
    provider: Any,
    num_fragments: int,
    base_path: Path,
) -> list[File]:
    """Async version of create_fragmented_state.

    Args:
        provider: LanceDB provider instance
        num_fragments: Number of fragments to create
        base_path: Base directory for test files

    Returns:
        List of File objects that were inserted
    """
    # For now, just wrap sync version
    # Future: could parallelize insertions if provider supports async
    return await asyncio.to_thread(
        create_fragmented_state,
        provider,
        num_fragments,
        base_path,
    )


def get_fragment_count(provider: Any) -> dict[str, int]:
    """Get current fragment count from LanceDB provider.

    Args:
        provider: LanceDB provider instance

    Returns:
        Dict with fragment counts for each table
    """
    counts = {}

    if hasattr(provider, "_chunks_table") and provider._chunks_table is not None:
        try:
            # LanceDB exposes fragment count via version() method
            # Each version represents a fragment
            chunks_table = provider._chunks_table
            if hasattr(chunks_table, "version"):
                counts["chunks"] = chunks_table.version()
            else:
                # Fallback: count via stats (may not be exact fragment count)
                counts["chunks"] = -1
        except Exception:
            counts["chunks"] = -1

    if hasattr(provider, "_files_table") and provider._files_table is not None:
        try:
            files_table = provider._files_table
            if hasattr(files_table, "version"):
                counts["files"] = files_table.version()
            else:
                counts["files"] = -1
        except Exception:
            counts["files"] = -1

    return counts


def verify_no_duplicate_chunk_ids(results: list[dict[str, Any]]) -> tuple[bool, str]:
    """Verify search results contain no duplicate chunk_ids.

    Args:
        results: List of search result dictionaries

    Returns:
        Tuple of (is_valid, error_message)
        is_valid is True if no duplicates found
    """
    if not results:
        return True, ""

    chunk_ids = [r.get("chunk_id") or r.get("id") for r in results]
    chunk_ids = [cid for cid in chunk_ids if cid is not None]

    unique_ids = set(chunk_ids)

    if len(chunk_ids) == len(unique_ids):
        return True, ""

    # Find duplicates
    seen = set()
    duplicates = set()
    for cid in chunk_ids:
        if cid in seen:
            duplicates.add(cid)
        seen.add(cid)

    error_msg = f"Found {len(chunk_ids) - len(unique_ids)} duplicate chunk_ids: {list(duplicates)}"
    return False, error_msg
