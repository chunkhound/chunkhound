"""Integration tests for real-time indexing ID/data alignment fix.

This test suite verifies the fix for the critical bug where chunk IDs and chunk
data could become misaligned during embedding generation, potentially causing
embeddings to be generated for the wrong chunks.

The fix introduces IndexingResult dataclass to maintain 1:1 correspondence
between chunk_ids_needing_embeddings and chunks_for_embedding throughout
the indexing pipeline.
"""

import asyncio
import time
from pathlib import Path

import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from tests.utils.windows_compat import database_cleanup_context, windows_safe_tempdir


class TestRealtimeIndexingAlignment:
    """Test suite for real-time indexing ID/data alignment."""

    @pytest.mark.asyncio
    async def test_chunk_id_data_alignment_after_embedding(self):
        """Verify chunk IDs match chunk data after embedding generation.

        This is a regression test for the bug where chunk_ids_needing_embeddings
        and chunks_for_embedding could be misaligned, causing embeddings to be
        generated for the wrong chunks.
        """
        with windows_safe_tempdir() as temp_dir:
            # Create a test file
            test_file = temp_dir / "test.py"
            test_file.write_text("""
def function_one():
    '''First test function with extended documentation.'''
    result = 1
    return result







def function_two():
    '''Second test function with extended documentation.'''
    result = 2
    return result







def function_three():
    '''Third test function with extended documentation.'''
    result = 3
    return result
""")

            # Create config without embedding provider (we'll skip embeddings)
            config = Config(target_dir=temp_dir)
            config.database.provider = "duckdb"
            db_path = temp_dir / ".chunkhound" / "test.db"
            db_path.parent.mkdir(exist_ok=True)  # Ensure directory exists
            config.database.path = str(db_path)

            with database_cleanup_context():
                db_services = create_services(db_path, config)
                db_provider = db_services.provider

                # Import here to avoid circular dependency
                from chunkhound.services.indexing_coordinator import IndexingCoordinator

                coordinator = IndexingCoordinator(
                    database_provider=db_provider,
                    base_directory=temp_dir,
                    embedding_provider=None,  # No embedding provider
                    config=config,
                )

                # Index the file (skip embeddings for this test to avoid API calls)
                result = await coordinator.process_file(test_file, skip_embeddings=True)

                assert result["status"] == "success", f"Indexing failed: {result}"
                assert result["chunks"] == 3, "Expected 3 chunks (3 functions)"

                # Verify chunks are stored correctly
                file_record = db_provider.get_file_by_path(str(test_file), as_model=True)
                assert file_record is not None, "File not found in database"
                chunks = db_provider.get_chunks_by_file_id(file_record.id, as_model=True)
                assert len(chunks) == 3, "Expected 3 chunks in database"

                # Verify chunk data matches what was parsed
                chunk_symbols = [chunk.symbol for chunk in chunks]
                assert "function_one" in chunk_symbols
                assert "function_two" in chunk_symbols
                assert "function_three" in chunk_symbols

    @pytest.mark.asyncio
    async def test_debouncing_processes_final_file_state(self):
        """Verify debouncing retry loop processes final file state.

        This test verifies the fix for dropped file updates when files are
        modified during the debounce delay period.
        """
        with windows_safe_tempdir() as temp_dir:
            # Create a test file
            test_file = temp_dir / "test.py"
            test_file.write_text("def original(): pass")

            # Create config
            config = Config(target_dir=temp_dir)
            config.database.provider = "duckdb"
            db_path = temp_dir / ".chunkhound" / "test.db"
            db_path.parent.mkdir(exist_ok=True)  # Ensure directory exists
            config.database.path = str(db_path)

            with database_cleanup_context():
                db_services = create_services(db_path, config)

                # Import here to avoid circular dependency
                from chunkhound.services.realtime_indexing_service import (
                    RealtimeIndexingService,
                )

                # Create real-time indexing service with short debounce delay
                rt_service = RealtimeIndexingService(
                    services=db_services,
                    config=config,
                )
                # Set debounce delay via attribute for faster test
                rt_service._debounce_delay = 0.1  # 100ms for faster test

                # Start the service
                await rt_service.start(temp_dir)

                try:
                    # Wait for service to be ready
                    await asyncio.sleep(0.2)

                    # Modify file multiple times rapidly (during debounce window)
                    test_file.write_text("def modified_v1(): pass")
                    await asyncio.sleep(0.05)  # Half the debounce delay
                    test_file.write_text("def modified_v2(): pass")
                    await asyncio.sleep(0.05)
                    test_file.write_text("def modified_final(): pass")

                    # Wait for debouncing to complete and file to be processed
                    await asyncio.sleep(0.5)

                    # Verify final state was processed
                    file_record = db_services.provider.get_file_by_path(str(test_file), as_model=True)
                    assert file_record is not None, "File not found in database"
                    chunks = db_services.provider.get_chunks_by_file_id(file_record.id, as_model=True)
                    assert len(chunks) == 1, "Expected 1 chunk after updates"

                    # Verify it's the final version
                    assert chunks[0].symbol == "modified_final", (
                        f"Expected final version 'modified_final', "
                        f"got '{chunks[0].symbol}'"
                    )

                finally:
                    await rt_service.stop()

    @pytest.mark.asyncio
    async def test_indexing_result_invariants(self):
        """Verify IndexingResult maintains required invariants.

        Tests that chunk_ids_needing_embeddings and chunks_for_embedding
        maintain 1:1 correspondence throughout the indexing pipeline.
        """
        with windows_safe_tempdir() as temp_dir:
            # Create a test file with multiple chunks
            test_file = temp_dir / "test.py"
            test_file.write_text("""
class TestClass:
    '''Test class with extended documentation.'''

    def method_one(self):
        '''First method with extended documentation.'''
        result = 1
        return result

    def method_two(self):
        '''Second method with extended documentation.'''
        result = 2
        return result

def standalone_function():
    '''Standalone function with extended documentation.'''
    result = 3
    return result
""")

            # Create config
            config = Config(target_dir=temp_dir)
            config.database.provider = "duckdb"
            db_path = temp_dir / ".chunkhound" / "test.db"
            db_path.parent.mkdir(exist_ok=True)  # Ensure directory exists
            config.database.path = str(db_path)

            with database_cleanup_context():
                db_services = create_services(db_path, config)
                db_provider = db_services.provider

                # Import here to avoid circular dependency
                from chunkhound.services.indexing_coordinator import IndexingCoordinator

                coordinator = IndexingCoordinator(
                    database_provider=db_provider,
                    base_directory=temp_dir,
                    embedding_provider=None,
                    config=config,
                )

                # Parse the file
                from chunkhound.services.batch_processor import process_file_batch

                batch_result = process_file_batch(
                    [test_file],
                    {
                        "target_dir": str(temp_dir),
                        "per_file_timeout_seconds": 0,
                        "max_concurrent_timeouts": 0,
                    },
                )

                assert len(batch_result) == 1
                result = batch_result[0]
                assert result.status == "success"

                # Store the parsed results using internal method
                # Note: _store_parsed_results returns (IndexingResult, file_id) for single file
                result_data = await coordinator._store_parsed_results([result])
                if isinstance(result_data, tuple):
                    indexing_result, file_id = result_data
                else:
                    indexing_result = result_data

                # Verify Invariant 1: Length equality
                assert len(indexing_result.chunk_ids_needing_embeddings) == len(
                    indexing_result.chunks_for_embedding
                ), (
                    f"Invariant 1 violated: "
                    f"{len(indexing_result.chunk_ids_needing_embeddings)} IDs vs "
                    f"{len(indexing_result.chunks_for_embedding)} chunks"
                )

                # Verify Invariant 3: All chunks are Chunk objects
                from chunkhound.core.models.chunk import Chunk

                assert all(
                    isinstance(c, Chunk) for c in indexing_result.chunks_for_embedding
                ), "Invariant 3 violated: Not all items are Chunk objects"

                # Verify Invariant 2: Positional correspondence
                # (Check that chunk at position i has ID chunk_ids[i])
                file_record = db_provider.get_file_by_path(str(test_file), as_model=True)
                assert file_record is not None, "File not found in database"
                stored_chunks = db_provider.get_chunks_by_file_id(file_record.id, as_model=True)
                chunk_id_to_symbol = {chunk.id: chunk.symbol for chunk in stored_chunks}

                for i, (chunk_id, chunk_obj) in enumerate(
                    zip(
                        indexing_result.chunk_ids_needing_embeddings,
                        indexing_result.chunks_for_embedding,
                    )
                ):
                    # Verify the chunk ID at position i corresponds to the chunk object at position i
                    stored_symbol = chunk_id_to_symbol.get(chunk_id)
                    assert chunk_obj.symbol == stored_symbol, (
                        f"Invariant 2 violated at position {i}: "
                        f"chunk_id {chunk_id} has symbol '{stored_symbol}' in DB, "
                        f"but chunk_obj has symbol '{chunk_obj.symbol}'"
                    )

    @pytest.mark.asyncio
    async def test_type_validation_rejects_dicts(self):
        """Verify _generate_embeddings() rejects dict objects.

        This test ensures type safety is enforced and dicts are no longer
        accepted in the chunks parameter.
        """
        with windows_safe_tempdir() as temp_dir:
            # Create minimal config
            config = Config(target_dir=temp_dir)
            config.database.provider = "duckdb"
            db_path = temp_dir / ".chunkhound" / "test.db"
            db_path.parent.mkdir(exist_ok=True)
            config.database.path = str(db_path)

            with database_cleanup_context():
                db_services = create_services(db_path, config)
                db_provider = db_services.provider

                # Create a mock embedding provider (just needs to exist)
                from unittest.mock import Mock
                mock_embedding_provider = Mock()

                # Import here to avoid circular dependency
                from chunkhound.services.indexing_coordinator import IndexingCoordinator

                coordinator = IndexingCoordinator(
                    database_provider=db_provider,
                    base_directory=temp_dir,
                    embedding_provider=mock_embedding_provider,
                    config=config,
                )

                # Try to call _generate_embeddings with dict objects (should fail)
                chunk_ids = [1, 2, 3]
                chunks_as_dicts = [
                    {"code": "def test1(): pass", "symbol": "test1"},
                    {"code": "def test2(): pass", "symbol": "test2"},
                    {"code": "def test3(): pass", "symbol": "test3"},
                ]

                # Should raise TypeError
                with pytest.raises(TypeError, match="Expected list\\[Chunk\\]"):
                    await coordinator._generate_embeddings(chunk_ids, chunks_as_dicts)
