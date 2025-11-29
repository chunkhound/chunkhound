"""Integration tests for LanceDB regex search.

Tests that regex search functionality works correctly with the LanceDB backend,
matching the semantic search test coverage patterns.
"""

from pathlib import Path

import pytest

def test_regex_basic(lancedb_provider, tmp_path):
    """Basic regex pattern finds matching content."""
    from chunkhound.core.models import Chunk, File
    from chunkhound.core.types.common import ChunkType, Language

    test_file = File(
        path="test.py",
        mtime=1234567890.0,
        language=Language.PYTHON,
        size_bytes=100,
    )
    file_id = lancedb_provider.insert_file(test_file)

    chunks = [
        Chunk(
            file_id=file_id,
            code="def unique_func_12345(): pass",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol="unique_func_12345",
        ),
        Chunk(
            file_id=file_id,
            code="def other_function(): return 42",
            start_line=2,
            end_line=2,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol="other_function",
        ),
    ]
    lancedb_provider.insert_chunks_batch(chunks)

    results, pagination = lancedb_provider.search_regex("unique_func_12345")

    assert len(results) == 1, "Should find exactly one match"
    assert "unique_func_12345" in results[0]["content"]
    assert pagination["total"] == 1


def test_regex_empty_results(lancedb_provider, tmp_path):
    """Non-matching pattern returns empty with valid pagination."""
    from chunkhound.core.models import Chunk, File
    from chunkhound.core.types.common import ChunkType, Language

    test_file = File(
        path="test.py",
        mtime=1234567890.0,
        language=Language.PYTHON,
        size_bytes=100,
    )
    file_id = lancedb_provider.insert_file(test_file)

    chunk = Chunk(
        file_id=file_id,
        code="def hello(): pass",
        start_line=1,
        end_line=1,
        chunk_type=ChunkType.FUNCTION,
        language=Language.PYTHON,
        symbol="hello",
    )
    lancedb_provider.insert_chunks_batch([chunk])

    results, pagination = lancedb_provider.search_regex("nonexistent_pattern_xyz")

    assert len(results) == 0, "Should return empty results"
    assert pagination["total"] == 0
    assert pagination["has_more"] is False


def test_regex_pagination(lancedb_provider, tmp_path):
    """Pagination works with page_size and offset."""
    from chunkhound.core.models import Chunk, File
    from chunkhound.core.types.common import ChunkType, Language

    test_file = File(
        path="test.py",
        mtime=1234567890.0,
        language=Language.PYTHON,
        size_bytes=100,
    )
    file_id = lancedb_provider.insert_file(test_file)

    # Create multiple chunks with common pattern
    chunks = [
        Chunk(
            file_id=file_id,
            code=f"def common_pattern_{i}(): return {i}",
            start_line=i + 1,
            end_line=i + 1,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol=f"common_pattern_{i}",
        )
        for i in range(5)
    ]
    lancedb_provider.insert_chunks_batch(chunks)

    # First page
    results1, pagination1 = lancedb_provider.search_regex("common_pattern", page_size=2, offset=0)
    assert len(results1) == 2, "First page should have 2 results"
    assert pagination1["has_more"] is True
    assert pagination1["total"] == 5

    # Second page
    results2, pagination2 = lancedb_provider.search_regex("common_pattern", page_size=2, offset=2)
    assert len(results2) == 2, "Second page should have 2 results"
    assert pagination2["has_more"] is True

    # Third page (partial)
    results3, pagination3 = lancedb_provider.search_regex("common_pattern", page_size=2, offset=4)
    assert len(results3) == 1, "Third page should have 1 result"
    assert pagination3["has_more"] is False


def test_regex_invalid_pattern_raises(lancedb_provider, tmp_path):
    """Invalid regex raises RuntimeError (aligned with semantic search)."""
    from chunkhound.core.models import Chunk, File
    from chunkhound.core.types.common import ChunkType, Language

    # Insert file and chunk so regex is actually evaluated
    test_file = File(
        path="test.py",
        mtime=1234567890.0,
        language=Language.PYTHON,
        size_bytes=100,
    )
    file_id = lancedb_provider.insert_file(test_file)

    chunk = Chunk(
        file_id=file_id,
        code="def hello(): pass",
        start_line=1,
        end_line=1,
        chunk_type=ChunkType.FUNCTION,
        language=Language.PYTHON,
        symbol="hello",
    )
    lancedb_provider.insert_chunks_batch([chunk])

    with pytest.raises(RuntimeError, match="Regex search failed"):
        lancedb_provider.search_regex("[invalid(regex")
