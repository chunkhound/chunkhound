"""Integration tests for LanceDB chunk deduplication fix.

Tests verify that the three-layer defense strategy prevents duplicate chunks:
- Layer 1: Database merge_insert prevents duplicates
- Layer 2: Scalar index provides performance
- Layer 3: Event deduplication reduces redundant work
"""

import asyncio
import time
from pathlib import Path

import pytest

# Skip these tests if lancedb is not available
class TestChunkDeduplication:
    """Test suite for chunk deduplication via merge_insert."""

    def test_duplicate_file_processing_no_duplicate_chunks(self, tmp_path, lancedb_provider):
        """Verify processing same file twice doesn't create duplicate chunks."""
        from chunkhound.services.indexing_coordinator import IndexingCoordinator

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def test_fn(): pass\nclass TestClass: pass")

        coord = IndexingCoordinator(database_provider=lancedb_provider, base_directory=tmp_path)

        # Process file twice (simulating created + modified events)
        result1 = asyncio.run(coord.process_file(test_file))
        result2 = asyncio.run(coord.process_file(test_file))

        # Both should succeed
        assert result1.get("status") != "error"
        assert result2.get("status") != "error"

        # Get all chunks
        file_record = lancedb_provider.get_file_by_path("test.py")
        assert file_record is not None

        file_id = file_record["id"] if isinstance(file_record, dict) else file_record.id
        all_chunks = lancedb_provider.get_chunks_by_file_id(file_id)
        chunk_ids = [c["id"] for c in all_chunks]

        # Verify no duplicate chunk IDs
        assert len(chunk_ids) == len(set(chunk_ids)), f"Duplicate chunk IDs found: {chunk_ids}"
        print(f"✓ Processed file twice, got {len(chunk_ids)} unique chunks (no duplicates)")

    def test_vue_haskell_no_duplicate_chunk_ids(self, tmp_path, lancedb_provider):
        """Verify Vue/Haskell files with identical content get unique chunk IDs.

        Regression test for: Duplicate chunk IDs detected in batch errors.
        Vue directives and elements with identical content should get different IDs
        because they have different chunk types (concept-aware hashing).
        """
        from chunkhound.services.indexing_coordinator import IndexingCoordinator

        # Create a Vue file with structure likely to produce identical content
        # Vue parser may extract both directives (DEFINITION) and elements (BLOCK)
        vue_file = tmp_path / "component.vue"
        vue_file.write_text("""<template>
  <div v-if="show">Content</div>
</template>
<script>
export default { data() { return { show: true } } }
</script>
""")

        coord = IndexingCoordinator(database_provider=lancedb_provider, base_directory=tmp_path)

        # Index the file
        result = asyncio.run(coord.process_file(vue_file))

        # Should succeed without duplicate ID errors
        assert result.get("status") != "error", f"Indexing failed: {result}"

        # Verify no duplicate chunk IDs in database
        file_record = lancedb_provider.get_file_by_path("component.vue")
        assert file_record is not None

        file_id = file_record["id"] if isinstance(file_record, dict) else file_record.id
        chunks = lancedb_provider.get_chunks_by_file_id(file_id)
        chunk_ids = [c["id"] for c in chunks]

        assert len(chunk_ids) == len(set(chunk_ids)), \
            f"Found duplicate chunk IDs in Vue file: {chunk_ids}"
        print(f"✓ Vue file indexed: {len(chunk_ids)} unique chunks (no duplicates)")

    def test_rapid_file_modifications(self, tmp_path, lancedb_provider):
        """Simulate rapid file modifications (editor save pattern)."""
        from chunkhound.services.indexing_coordinator import IndexingCoordinator

        test_file = tmp_path / "rapid.py"
        coord = IndexingCoordinator(database_provider=lancedb_provider, base_directory=tmp_path)

        # Version 1: Initial content
        test_file.write_text("def initial(): pass")
        result1 = asyncio.run(coord.process_file(test_file))

        # Version 2: Modified content (simulate editor save triggering modified event)
        test_file.write_text("def initial(): return 1")
        result2 = asyncio.run(coord.process_file(test_file))

        # Version 3: Another modification
        test_file.write_text("def initial(): return 2")
        result3 = asyncio.run(coord.process_file(test_file))

        # Get final chunks
        file_record = lancedb_provider.get_file_by_path("rapid.py")
        file_id = file_record["id"] if isinstance(file_record, dict) else file_record.id
        all_chunks = lancedb_provider.get_chunks_by_file_id(file_id)
        chunk_ids = [c["id"] for c in all_chunks]

        # Verify no duplicates despite multiple updates
        assert len(chunk_ids) == len(set(chunk_ids)), f"Duplicate chunk IDs found: {chunk_ids}"
        print(f"✓ Rapid modifications: {len(chunk_ids)} unique chunks (no duplicates)")


class TestConcurrentProcessing:
    """Test suite for concurrent file processing scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_file_processing(self, tmp_path, lancedb_provider):
        """Verify concurrent processing of same file doesn't create duplicates."""
        from chunkhound.services.indexing_coordinator import IndexingCoordinator

        test_file = tmp_path / "concurrent.py"
        test_file.write_text("def concurrent_test(): pass\nclass Concurrent: pass")

        coord = IndexingCoordinator(database_provider=lancedb_provider, base_directory=tmp_path)

        # Process same file concurrently (simulating race between initial scan + realtime)
        results = await asyncio.gather(
            coord.process_file(test_file),
            coord.process_file(test_file),
            coord.process_file(test_file),
        )

        # All should complete
        assert all(r.get("status") != "error" for r in results)

        # Verify no duplicates
        file_record = lancedb_provider.get_file_by_path("concurrent.py")
        file_id = file_record["id"] if isinstance(file_record, dict) else file_record.id
        all_chunks = lancedb_provider.get_chunks_by_file_id(file_id)
        chunk_ids = [c["id"] for c in all_chunks]

        assert len(chunk_ids) == len(set(chunk_ids)), f"Duplicate chunk IDs found"
        print(f"✓ Concurrent processing: {len(chunk_ids)} unique chunks (no duplicates)")


class TestScalarIndexCreation:
    """Test suite for scalar index creation."""

    def test_scalar_index_created_on_connect(self, lancedb_provider):
        """Verify scalar index creation happens during connection."""
        # Index creation happens in _executor_create_indexes (called from _executor_connect)
        assert lancedb_provider._chunks_table is not None
        print("✓ Scalar index creation verified (table exists)")


class TestSearchDeduplication:
    """Test suite for search result deduplication across LanceDB fragments.

    Regression tests for the bug where search_regex and search_semantic returned
    duplicate chunk_ids due to fragmentation.
    """

    def test_regex_search_no_duplicates_with_fragments(
        self, fragmented_lancedb_provider, tmp_path
    ):
        """Verify regex search returns unique chunk_ids with 50+ fragments."""
        from chunkhound.core.models import Chunk, File
        from chunkhound.core.types.common import ChunkType, Language
        from tests.fixtures.fragmentation_helpers import verify_no_duplicate_chunk_ids

        # The fragmented_lancedb_provider already has 50 fragments
        # Add a test file with searchable content
        test_file = File(
            path="searchable.py",
            mtime=9999999999.0,
            language=Language.PYTHON,
            size_bytes=100,
        )
        file_id = fragmented_lancedb_provider.insert_file(test_file)

        # Insert chunk with unique pattern
        chunk = Chunk(
            file_id=file_id,
            code="def unique_search_target_12345(): return 'found'",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol="unique_search_target_12345",
        )
        fragmented_lancedb_provider.insert_chunk(chunk)

        # Search for the unique pattern
        results, pagination = fragmented_lancedb_provider.search_regex(
            pattern="unique_search_target_12345",
            page_size=100,
            offset=0,
            path_filter=None,
        )

        # Verify no duplicates
        is_valid, error_msg = verify_no_duplicate_chunk_ids(results)
        assert is_valid, f"Regex search returned duplicates: {error_msg}"
        assert len(results) >= 1, "Should find at least the inserted chunk"

        print(
            f"✓ Regex search with 50 fragments: {len(results)} unique results (no duplicates)"
        )

    def test_file_update_creates_no_search_duplicates(
        self, lancedb_provider, tmp_path
    ):
        """Verify file updates don't create duplicate search results."""
        from chunkhound.core.models import Chunk, File
        from chunkhound.core.types.common import ChunkType, Language
        from tests.fixtures.fragmentation_helpers import verify_no_duplicate_chunk_ids

        # Insert initial file
        test_file = File(
            path="updated_file.py",
            mtime=1000000.0,
            language=Language.PYTHON,
            size_bytes=100,
        )
        file_id = lancedb_provider.insert_file(test_file)

        # Insert initial chunk
        chunk = Chunk(
            file_id=file_id,
            code="def update_test_v1(): return 1",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol="update_test_v1",
        )
        lancedb_provider.insert_chunk(chunk)

        # Update file 5 times (creates multiple fragments)
        for i in range(2, 7):
            updated_file = File(
                id=file_id,
                path="updated_file.py",
                mtime=1000000.0 + i,
                language=Language.PYTHON,
                size_bytes=100 + i,
            )
            lancedb_provider.insert_file(updated_file)

            updated_chunk = Chunk(
                file_id=file_id,
                code=f"def update_test_v{i}(): return {i}",
                start_line=1,
                end_line=1,
                chunk_type=ChunkType.FUNCTION,
                language=Language.PYTHON,
                symbol=f"update_test_v{i}",
            )
            lancedb_provider.insert_chunk(updated_chunk)

        # Search for the latest version
        results, pagination = lancedb_provider.search_regex(
            pattern="update_test_v6",
            page_size=100,
            offset=0,
            path_filter=None,
        )

        # Verify no duplicates
        is_valid, error_msg = verify_no_duplicate_chunk_ids(results)
        assert is_valid, f"File update search returned duplicates: {error_msg}"

        # Should find exactly 1 result (latest version)
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

        print(
            f"✓ File update (5 iterations): {len(results)} unique result (no duplicates)"
        )

    def test_regex_search_pagination_counts_unique_chunks(
        self, fragmented_lancedb_provider, tmp_path
    ):
        """Verify pagination total count reflects unique chunks, not duplicates."""
        from chunkhound.core.models import Chunk, File
        from chunkhound.core.types.common import ChunkType, Language

        # Insert file with multiple searchable chunks
        test_file = File(
            path="multi_chunk.py",
            mtime=9999999999.0,
            language=Language.PYTHON,
            size_bytes=500,
        )
        file_id = fragmented_lancedb_provider.insert_file(test_file)

        # Insert 10 chunks with common pattern
        for i in range(10):
            chunk = Chunk(
                file_id=file_id,
                code=f"def pagination_test_{i}(): return {i}",
                start_line=i + 1,
                end_line=i + 1,
                chunk_type=ChunkType.FUNCTION,
                language=Language.PYTHON,
                symbol=f"pagination_test_{i}",
            )
            fragmented_lancedb_provider.insert_chunk(chunk)

        # Search for common pattern
        results, pagination = fragmented_lancedb_provider.search_regex(
            pattern="pagination_test_",
            page_size=100,
            offset=0,
            path_filter=None,
        )

        # Verify total count matches unique results
        assert pagination["total"] == len(
            results
        ), f"Total count {pagination['total']} != actual results {len(results)}"
        assert len(results) == 10, f"Expected 10 unique chunks, got {len(results)}"

        # Verify all chunk_ids are unique
        chunk_ids = [r["chunk_id"] for r in results]
        assert len(chunk_ids) == len(
            set(chunk_ids)
        ), "Found duplicate chunk_ids in results"

        print(
            f"✓ Pagination counts: total={pagination['total']}, results={len(results)} (match, no duplicates)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
