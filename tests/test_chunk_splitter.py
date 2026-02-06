"""Tests for chunk_splitter module."""

from chunkhound.core.types.common import ChunkType
from chunkhound.parsers.chunk_splitter import CHUNK_TYPE_TO_CONCEPT


def test_chunk_to_universal_covers_all_chunk_types() -> None:
    """Verify CHUNK_TYPE_TO_CONCEPT covers all ChunkType enum members."""
    all_types = set(ChunkType)
    mapped_types = set(CHUNK_TYPE_TO_CONCEPT.keys())
    missing = all_types - mapped_types

    assert not missing, (
        f"CHUNK_TYPE_TO_CONCEPT missing {len(missing)} types: "
        f"{sorted(t.name for t in missing)}"
    )
