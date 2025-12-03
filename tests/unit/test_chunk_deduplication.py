"""Unit tests for chunk deduplication module."""

import pytest

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import ChunkType, FileId, Language, LineNumber
from chunkhound.utils.chunk_deduplication import deduplicate_chunks


def test_exact_content_deduplication():
    """Chunks with identical content should be deduplicated by specificity."""
    chunks = [
        Chunk(
            symbol="foo",
            start_line=LineNumber(1),
            end_line=LineNumber(3),
            code="def foo():\n    pass",
            chunk_type=ChunkType.FUNCTION,
            file_id=FileId(1),
            language=Language.PYTHON,
        ),
        Chunk(
            symbol="foo_struct",
            start_line=LineNumber(1),
            end_line=LineNumber(3),
            code="def foo():\n    pass",  # Identical content
            chunk_type=ChunkType.STRUCT,  # Lower specificity
            file_id=FileId(1),
            language=Language.PYTHON,
        ),
    ]

    result = deduplicate_chunks(chunks, Language.PYTHON)

    assert len(result) == 1
    assert result[0].chunk_type == ChunkType.FUNCTION  # Higher specificity


def test_yaml_duplicate_removal():
    """YAML files with duplicate config values should be deduplicated."""
    chunks = [
        Chunk(
            symbol="metadata.name",
            start_line=LineNumber(4),
            end_line=LineNumber(4),
            code="  name: example-config",
            chunk_type=ChunkType.KEY_VALUE,
            file_id=FileId(1),
            language=Language.YAML,
        ),
        Chunk(
            symbol="configMapRef.name",
            start_line=LineNumber(21),
            end_line=LineNumber(21),
            code="                name: example-config",  # Same normalized content
            chunk_type=ChunkType.KEY_VALUE,
            file_id=FileId(1),
            language=Language.YAML,
        ),
    ]

    result = deduplicate_chunks(chunks, Language.YAML)

    # Should deduplicate (both same type, identical normalized content)
    assert len(result) == 1


def test_empty_chunks_skipped():
    """Empty or whitespace-only chunks should be filtered out."""
    chunks = [
        Chunk(
            symbol="empty",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code="   ",  # Whitespace only
            chunk_type=ChunkType.COMMENT,
            file_id=FileId(1),
            language=Language.PYTHON,
        ),
        Chunk(
            symbol="valid",
            start_line=LineNumber(2),
            end_line=LineNumber(2),
            code="# Comment",
            chunk_type=ChunkType.COMMENT,
            file_id=FileId(1),
            language=Language.PYTHON,
        ),
    ]

    result = deduplicate_chunks(chunks, Language.PYTHON)

    assert len(result) == 1
    assert result[0].symbol == "valid"


def test_dict_input_compatibility():
    """Should work with chunk dictionaries (not just Chunk objects)."""
    chunks = [
        {
            "symbol": "foo",
            "start_line": 1,
            "end_line": 3,
            "code": "def foo():\n    pass",
            "chunk_type": ChunkType.FUNCTION,
            "file_id": 1,
            "language": "python",
        },
        {
            "symbol": "foo",
            "start_line": 1,
            "end_line": 3,
            "code": "def foo():\n    pass",
            "chunk_type": ChunkType.STRUCT,
            "file_id": 1,
            "language": "python",
        },
    ]

    result = deduplicate_chunks(chunks, Language.PYTHON)

    assert len(result) == 1
    assert result[0]["chunk_type"] == ChunkType.FUNCTION


def test_language_string_compatibility():
    """Should work with language as string (not just Language enum)."""
    chunks = [
        Chunk(
            symbol="foo",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code="test",
            chunk_type=ChunkType.FUNCTION,
            file_id=FileId(1),
            language=Language.PYTHON,
        ),
    ]

    # Should not raise error with string language
    result = deduplicate_chunks(chunks, "python")

    assert len(result) == 1


def test_empty_list_returns_empty():
    """Empty input should return empty list."""
    result = deduplicate_chunks([], Language.PYTHON)
    assert result == []


def test_none_language_works():
    """Should work with None as language."""
    chunks = [
        Chunk(
            symbol="foo",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code="test",
            chunk_type=ChunkType.FUNCTION,
            file_id=FileId(1),
            language=Language.PYTHON,
        ),
    ]

    result = deduplicate_chunks(chunks, None)

    assert len(result) == 1


def test_specificity_ranking():
    """Higher specificity chunks should be preferred."""
    # Create chunks with same content but different types
    chunks = [
        Chunk(
            symbol="test_array",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code="identical content",
            chunk_type=ChunkType.ARRAY,  # Specificity 1 - should be filtered
            file_id=FileId(1),
            language=Language.PYTHON,
        ),
        Chunk(
            symbol="test_key_value",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code="identical content",
            chunk_type=ChunkType.KEY_VALUE,  # Specificity 2 - should be filtered
            file_id=FileId(1),
            language=Language.PYTHON,
        ),
        Chunk(
            symbol="test_function",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code="identical content",
            chunk_type=ChunkType.FUNCTION,  # Specificity 4 - should be kept
            file_id=FileId(1),
            language=Language.PYTHON,
        ),
    ]

    result = deduplicate_chunks(chunks, Language.PYTHON)

    # Should keep only the highest specificity (FUNCTION)
    assert len(result) == 1
    assert result[0].chunk_type == ChunkType.FUNCTION
