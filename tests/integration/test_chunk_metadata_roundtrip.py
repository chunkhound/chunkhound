"""Integration tests for chunk metadata persistence.

Tests verify that chunk metadata (including constants) survives:
1. Database insertion (Chunk -> database)
2. Database retrieval (database -> dict with metadata)
3. Search results (semantic and regex)
"""

import pytest

from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language


class TestLanceDBMetadataRoundtrip:
    """Test metadata persistence through LanceDB operations."""

    def test_chunk_metadata_survives_insert_and_search(self, lancedb_provider, tmp_path):
        """Verify chunk metadata is preserved through insert and regex search."""
        # Create test file
        test_file = File(
            path="constants.py",
            mtime=1000000.0,
            language=Language.PYTHON,
            size_bytes=100,
        )
        file_id = lancedb_provider.insert_file(test_file)

        # Create chunk with metadata containing constants
        metadata = {
            "constants": [
                {"name": "MAX_VALUE", "value": "100"},
                {"name": "API_KEY", "value": '"secret"'},
            ],
            "visibility": "public",
        }
        chunk = Chunk(
            file_id=file_id,
            code="MAX_VALUE = 100\nAPI_KEY = 'secret'",
            start_line=1,
            end_line=2,
            chunk_type=ChunkType.FUNCTION,  # Using valid chunk type
            language=Language.PYTHON,
            symbol="constants_block",
            metadata=metadata,
        )
        lancedb_provider.insert_chunk(chunk)

        # Search for the chunk
        results, _ = lancedb_provider.search_regex(
            pattern="MAX_VALUE",
            page_size=10,
            offset=0,
            path_filter=None,
        )

        assert len(results) >= 1, "Should find the inserted chunk"
        result = results[0]

        # Verify metadata is preserved
        assert "metadata" in result, "Result should include metadata field"
        assert result["metadata"] is not None, "Metadata should not be None"
        assert "constants" in result["metadata"], "Metadata should contain constants"
        assert len(result["metadata"]["constants"]) == 2, "Should have 2 constants"

        # Verify constant details
        constants = result["metadata"]["constants"]
        names = {c["name"] for c in constants}
        assert "MAX_VALUE" in names, f"Expected MAX_VALUE in {names}"
        assert "API_KEY" in names, f"Expected API_KEY in {names}"

    def test_batch_insert_preserves_metadata(self, lancedb_provider, tmp_path):
        """Verify batch insert preserves metadata for all chunks."""
        # Create test file
        test_file = File(
            path="batch_test.py",
            mtime=1000001.0,
            language=Language.PYTHON,
            size_bytes=200,
        )
        file_id = lancedb_provider.insert_file(test_file)

        # Create multiple chunks with different metadata
        chunks = [
            Chunk(
                file_id=file_id,
                code="DEFAULT_TIMEOUT = 30",
                start_line=1,
                end_line=1,
                chunk_type=ChunkType.FUNCTION,
                language=Language.PYTHON,
                symbol="timeout_const",
                metadata={"constants": [{"name": "DEFAULT_TIMEOUT", "value": "30"}]},
            ),
            Chunk(
                file_id=file_id,
                code="MAX_RETRIES = 3",
                start_line=2,
                end_line=2,
                chunk_type=ChunkType.FUNCTION,
                language=Language.PYTHON,
                symbol="retry_const",
                metadata={"constants": [{"name": "MAX_RETRIES", "value": "3"}]},
            ),
        ]
        lancedb_provider.insert_chunks_batch(chunks)

        # Search for each chunk and verify metadata
        results, _ = lancedb_provider.search_regex(
            pattern="DEFAULT_TIMEOUT|MAX_RETRIES",
            page_size=10,
            offset=0,
            path_filter=None,
        )

        assert len(results) == 2, f"Expected 2 chunks, got {len(results)}"
        for result in results:
            assert "metadata" in result, "Each result should have metadata"
            assert "constants" in result["metadata"], "Each result should have constants"
            assert len(result["metadata"]["constants"]) == 1

    def test_chunk_without_metadata_returns_empty_dict(self, lancedb_provider, tmp_path):
        """Verify chunks without metadata return empty dict (not None)."""
        # Create test file
        test_file = File(
            path="no_metadata.py",
            mtime=1000002.0,
            language=Language.PYTHON,
            size_bytes=50,
        )
        file_id = lancedb_provider.insert_file(test_file)

        # Create chunk without metadata
        chunk = Chunk(
            file_id=file_id,
            code="def no_metadata_function(): pass",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol="no_metadata_function",
            metadata=None,
        )
        lancedb_provider.insert_chunk(chunk)

        # Search for the chunk
        results, _ = lancedb_provider.search_regex(
            pattern="no_metadata_function",
            page_size=10,
            offset=0,
            path_filter=None,
        )

        assert len(results) >= 1
        result = results[0]

        # Metadata should be empty dict, not None
        assert "metadata" in result
        assert result["metadata"] == {}, f"Expected empty dict, got {result['metadata']}"


class TestMetadataInAllChunks:
    """Test metadata in get_all_chunks_with_metadata."""

    def test_get_all_chunks_includes_metadata(self, lancedb_provider, tmp_path):
        """Verify get_all_chunks_with_metadata includes chunk metadata."""
        # Create test file
        test_file = File(
            path="all_chunks.py",
            mtime=3000000.0,
            language=Language.PYTHON,
            size_bytes=100,
        )
        file_id = lancedb_provider.insert_file(test_file)

        # Create chunk with metadata
        chunk = Chunk(
            file_id=file_id,
            code="GLOBAL_FLAG = True",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol="global_flag",
            metadata={"constants": [{"name": "GLOBAL_FLAG", "value": "True"}]},
        )
        lancedb_provider.insert_chunk(chunk)

        # Get all chunks with metadata
        all_chunks = lancedb_provider.get_all_chunks_with_metadata()

        # Find our chunk
        our_chunk = next(
            (c for c in all_chunks if c.get("file_path") == "all_chunks.py"),
            None,
        )
        assert our_chunk is not None, "Should find our test chunk"
        assert "metadata" in our_chunk, "Chunk should have metadata field"
        assert "constants" in our_chunk["metadata"], "Metadata should have constants"


class TestDuckDBMetadataRoundtrip:
    """Test metadata persistence through DuckDB operations."""

    @pytest.fixture
    def duckdb_provider(self, tmp_path):
        """Create DuckDB provider for testing."""
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider

        provider = DuckDBProvider(db_path=tmp_path / "test.duckdb", base_directory=tmp_path)
        provider.connect()
        yield provider
        provider.disconnect()

    def test_chunk_metadata_survives_insert_and_search(self, duckdb_provider, tmp_path):
        """Verify chunk metadata is preserved through insert and regex search."""
        # Create test file
        test_file = File(
            path="constants.py",
            mtime=1000000.0,
            language=Language.PYTHON,
            size_bytes=100,
        )
        file_id = duckdb_provider.insert_file(test_file)

        # Create chunk with metadata containing constants
        metadata = {
            "constants": [
                {"name": "MAX_VALUE", "value": "100"},
                {"name": "API_KEY", "value": '"secret"'},
            ],
            "visibility": "public",
        }
        chunk = Chunk(
            file_id=file_id,
            code="MAX_VALUE = 100\nAPI_KEY = 'secret'",
            start_line=1,
            end_line=2,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol="constants_block",
            metadata=metadata,
        )
        # Use batch insert (production path) - single insert_chunk missing metadata in DuckDB
        duckdb_provider.insert_chunks_batch([chunk])

        # Search for the chunk
        results, _ = duckdb_provider.search_regex(
            pattern="MAX_VALUE",
            page_size=10,
            offset=0,
            path_filter=None,
        )

        assert len(results) >= 1, "Should find the inserted chunk"
        result = results[0]

        # Verify metadata is preserved
        assert "metadata" in result, "Result should include metadata field"
        assert result["metadata"] is not None, "Metadata should not be None"
        assert "constants" in result["metadata"], "Metadata should contain constants"
        assert len(result["metadata"]["constants"]) == 2, "Should have 2 constants"

        # Verify constant details
        constants = result["metadata"]["constants"]
        names = {c["name"] for c in constants}
        assert "MAX_VALUE" in names, f"Expected MAX_VALUE in {names}"
        assert "API_KEY" in names, f"Expected API_KEY in {names}"

    def test_batch_insert_preserves_metadata(self, duckdb_provider, tmp_path):
        """Verify batch insert preserves metadata for all chunks."""
        # Create test file
        test_file = File(
            path="batch_test.py",
            mtime=1000001.0,
            language=Language.PYTHON,
            size_bytes=200,
        )
        file_id = duckdb_provider.insert_file(test_file)

        # Create multiple chunks with different metadata
        chunks = [
            Chunk(
                file_id=file_id,
                code="DEFAULT_TIMEOUT = 30",
                start_line=1,
                end_line=1,
                chunk_type=ChunkType.FUNCTION,
                language=Language.PYTHON,
                symbol="timeout_const",
                metadata={"constants": [{"name": "DEFAULT_TIMEOUT", "value": "30"}]},
            ),
            Chunk(
                file_id=file_id,
                code="MAX_RETRIES = 3",
                start_line=2,
                end_line=2,
                chunk_type=ChunkType.FUNCTION,
                language=Language.PYTHON,
                symbol="retry_const",
                metadata={"constants": [{"name": "MAX_RETRIES", "value": "3"}]},
            ),
        ]
        duckdb_provider.insert_chunks_batch(chunks)

        # Search for each chunk and verify metadata
        results, _ = duckdb_provider.search_regex(
            pattern="DEFAULT_TIMEOUT|MAX_RETRIES",
            page_size=10,
            offset=0,
            path_filter=None,
        )

        assert len(results) == 2, f"Expected 2 chunks, got {len(results)}"
        for result in results:
            assert "metadata" in result, "Each result should have metadata"
            assert "constants" in result["metadata"], "Each result should have constants"
            assert len(result["metadata"]["constants"]) == 1

    def test_chunk_without_metadata_returns_empty_dict(self, duckdb_provider, tmp_path):
        """Verify chunks without metadata return empty dict (not None)."""
        # Create test file
        test_file = File(
            path="no_metadata.py",
            mtime=1000002.0,
            language=Language.PYTHON,
            size_bytes=50,
        )
        file_id = duckdb_provider.insert_file(test_file)

        # Create chunk without metadata
        chunk = Chunk(
            file_id=file_id,
            code="def no_metadata_function(): pass",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol="no_metadata_function",
            metadata=None,
        )
        # Use batch insert (production path)
        duckdb_provider.insert_chunks_batch([chunk])

        # Search for the chunk
        results, _ = duckdb_provider.search_regex(
            pattern="no_metadata_function",
            page_size=10,
            offset=0,
            path_filter=None,
        )

        assert len(results) >= 1
        result = results[0]

        # Metadata should be empty dict, not None
        assert "metadata" in result
        assert result["metadata"] == {}, f"Expected empty dict, got {result['metadata']}"


class TestDuckDBMetadataInAllChunks:
    """Test metadata in get_all_chunks_with_metadata for DuckDB."""

    @pytest.fixture
    def duckdb_provider(self, tmp_path):
        """Create DuckDB provider for testing."""
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider

        provider = DuckDBProvider(db_path=tmp_path / "test.duckdb", base_directory=tmp_path)
        provider.connect()
        yield provider
        provider.disconnect()

    def test_get_all_chunks_includes_metadata(self, duckdb_provider, tmp_path):
        """Verify get_all_chunks_with_metadata includes chunk metadata."""
        # Create test file
        test_file = File(
            path="all_chunks.py",
            mtime=3000000.0,
            language=Language.PYTHON,
            size_bytes=100,
        )
        file_id = duckdb_provider.insert_file(test_file)

        # Create chunk with metadata
        chunk = Chunk(
            file_id=file_id,
            code="GLOBAL_FLAG = True",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol="global_flag",
            metadata={"constants": [{"name": "GLOBAL_FLAG", "value": "True"}]},
        )
        # Use batch insert (production path)
        duckdb_provider.insert_chunks_batch([chunk])

        # Get all chunks with metadata
        all_chunks = duckdb_provider.get_all_chunks_with_metadata()

        # Find our chunk
        our_chunk = next(
            (c for c in all_chunks if c.get("file_path") == "all_chunks.py"),
            None,
        )
        assert our_chunk is not None, "Should find our test chunk"
        assert "metadata" in our_chunk, "Chunk should have metadata field"
        assert "constants" in our_chunk["metadata"], "Metadata should have constants"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
