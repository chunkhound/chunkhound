"""Test suite to reproduce and verify the embedding generation bug.

This test module contains tests that are EXPECTED TO FAIL until the embedding
generation bug is fixed. These tests verify that new files get embeddings generated
and that the database maintains consistency between chunks and embeddings.
"""

import asyncio

import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services


@pytest.fixture
async def embedding_services(tmp_path):
    """Create database services with embedding provider configured."""
    db_path = tmp_path / "embedding_test.duckdb"

    # Create config with real embedding provider (if API key available)
    # Use fake args to prevent find_project_root call that fails in CI
    from types import SimpleNamespace
    fake_args = SimpleNamespace(path=tmp_path)
    config = Config(
        args=fake_args,
        database={"path": str(db_path), "provider": "duckdb"}
    )

    # Only test with real embeddings if API key is available
    if hasattr(config.embedding, 'api_key') and config.embedding.api_key:
        # Create services (embedding manager is created internally by registry)
        services = create_services(db_path, config)
        yield services
    else:
        pytest.skip("No embedding API key available for integration test")


@pytest.mark.asyncio
async def test_new_files_get_embeddings_generated_EXPECTED_FAIL(embedding_services, tmp_path):
    """REPRODUCTION TEST: New files should get embeddings generated.
    
    This test is EXPECTED TO FAIL due to the embedding generation bug.
    It verifies that newly created files get both chunks AND embeddings.
    
    BUG: New files get chunked but embeddings are not generated.
    """
    services = embedding_services

    # Get initial stats
    initial_stats = await services.indexing_coordinator.get_stats()
    initial_chunks = initial_stats.get('chunks', 0)
    initial_embeddings = initial_stats.get('embeddings', 0)

    # Create a new test file
    test_file = tmp_path / "embedding_test.py"
    test_file.write_text("""
def test_embedding_function():
    '''This function should get an embedding generated.'''
    return "embedding test content"

class TestEmbeddingClass:
    '''This class should also get embeddings.'''
    
    def method_with_embedding(self):
        '''This method needs embeddings too.'''
        return "method content for embedding"
""")

    # Process the file (this should generate both chunks and embeddings)
    result = await services.indexing_coordinator.process_file(test_file, skip_embeddings=False)

    # Verify file was processed successfully
    assert result['status'] == 'success', f"File processing failed: {result.get('error')}"
    assert result['chunks'] > 0, "Should have created chunks"

    # Wait a moment for any async embedding generation
    await asyncio.sleep(2.0)

    # Get final stats
    final_stats = await services.indexing_coordinator.get_stats()
    final_chunks = final_stats.get('chunks', 0)
    final_embeddings = final_stats.get('embeddings', 0)

    # Verify chunks were created
    chunks_created = final_chunks - initial_chunks
    assert chunks_created > 0, f"Expected chunks to be created, got {chunks_created}"

    # CRITICAL ASSERTION THAT WILL FAIL: Verify embeddings were created
    embeddings_created = final_embeddings - initial_embeddings
    assert embeddings_created > 0, f"EXPECTED FAILURE: No embeddings created for new chunks (created {embeddings_created})"

    # Database consistency check
    assert final_embeddings == final_chunks, f"EXPECTED FAILURE: Embedding count ({final_embeddings}) != chunk count ({final_chunks})"


@pytest.mark.asyncio
async def test_chunk_embedding_count_consistency_EXPECTED_FAIL(embedding_services, tmp_path):
    """REPRODUCTION TEST: Chunk count should match embedding count.
    
    This test is EXPECTED TO FAIL due to the embedding generation bug.
    It verifies database consistency between chunks and embeddings.
    """
    services = embedding_services

    # Create multiple test files
    for i in range(3):
        test_file = tmp_path / f"consistency_test_{i}.py"
        test_file.write_text(f"""
def function_{i}():
    '''Function {i} for consistency testing.'''
    return "test content {i}"

class TestClass_{i}:
    '''Class {i} for testing.'''
    
    def method_{i}(self):
        return "method {i}"
""")

        # Process each file
        result = await services.indexing_coordinator.process_file(test_file, skip_embeddings=False)
        assert result['status'] == 'success'

    # Wait for any async processing
    await asyncio.sleep(3.0)

    # Get final statistics
    stats = await services.indexing_coordinator.get_stats()
    chunk_count = stats['chunks']
    embedding_count = stats.get('embeddings', 0)

    # CRITICAL CONSISTENCY CHECK THAT WILL FAIL
    assert embedding_count == chunk_count, \
        f"EXPECTED FAILURE: Embedding count ({embedding_count}) should match chunk count ({chunk_count}). " \
        f"Missing embeddings: {chunk_count - embedding_count}"


@pytest.mark.asyncio
async def test_realtime_indexing_embeddings_EXPECTED_FAIL(tmp_path):
    """REPRODUCTION TEST: Real-time indexed files should get embeddings.

    This test is EXPECTED TO FAIL due to the embedding generation bug.
    It simulates the real-time indexing scenario that fails in practice.
    """
    # Setup config and services
    # Use fake args to prevent find_project_root call that fails in CI
    from types import SimpleNamespace

    from chunkhound.services.realtime_indexing_service import RealtimeIndexingService
    from tests.utils.windows_compat import should_use_polling
    fake_args = SimpleNamespace(path=tmp_path)
    config = Config(
        args=fake_args,
        database={"path": str(tmp_path / "realtime_test.duckdb"), "provider": "duckdb"}
    )

    # Only test with real embeddings if API key is available
    if not (hasattr(config.embedding, 'api_key') and config.embedding.api_key):
        pytest.skip("No embedding API key available for realtime test")

    services = create_services(config.database.path, config)
    # Use polling on Windows CI where watchdog's ReadDirectoryChangesW is unreliable
    force_polling = should_use_polling()
    realtime_service = RealtimeIndexingService(services, config, force_polling=force_polling)

    # Start realtime service
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    await realtime_service.start(watch_dir)

    try:
        # Get initial stats
        initial_stats = await services.indexing_coordinator.get_stats()
        initial_embeddings = initial_stats.get('embeddings', 0)

        # Create a new file (simulating user creating file)
        new_file = watch_dir / "realtime_test.py"
        new_file.write_text("""
def realtime_function():
    '''This function should get embeddings in real-time.'''
    return "realtime content"
""")

        # Wait for realtime processing (includes debouncing)
        await asyncio.sleep(3.0)

        # Get final stats
        final_stats = await services.indexing_coordinator.get_stats()
        final_embeddings = final_stats.get('embeddings', 0)

        # Check that embeddings were generated
        embeddings_generated = final_embeddings - initial_embeddings
        assert embeddings_generated > 0, \
            f"EXPECTED FAILURE: Real-time indexing should generate embeddings, got {embeddings_generated}"

        # Verify consistency
        assert final_stats['embeddings'] == final_stats['chunks'], \
            "EXPECTED FAILURE: Real-time indexing should maintain embedding/chunk consistency"

    finally:
        await realtime_service.stop()


@pytest.mark.asyncio
async def test_embedding_generation_after_skip_EXPECTED_FAIL(embedding_services, tmp_path):
    """REPRODUCTION TEST: Files processed with skip_embeddings=True should later get embeddings.
    
    This test is EXPECTED TO FAIL due to the embedding generation pipeline bug.
    It tests the two-phase processing: chunks first, embeddings later.
    """
    services = embedding_services

    # Create test file
    test_file = tmp_path / "skip_then_embed.py"
    test_file.write_text("""
def skip_then_embed_function():
    '''This should get chunked first, then embedded.'''
    return "two phase processing"
""")

    # Phase 1: Process with skip_embeddings=True
    result1 = await services.indexing_coordinator.process_file(test_file, skip_embeddings=True)
    assert result1['status'] == 'success'
    assert result1['embeddings_skipped'] == True
    assert result1['chunks'] > 0

    # Verify chunks exist but no embeddings yet
    stats_after_phase1 = await services.indexing_coordinator.get_stats()
    chunks_after_phase1 = stats_after_phase1['chunks']
    embeddings_after_phase1 = stats_after_phase1.get('embeddings', 0)

    # Phase 2: Process again with skip_embeddings=False (simulate embedding generation)
    result2 = await services.indexing_coordinator.process_file(test_file, skip_embeddings=False)
    assert result2['status'] == 'success'

    # Wait for embedding generation
    await asyncio.sleep(2.0)

    # Verify embeddings were generated
    stats_after_phase2 = await services.indexing_coordinator.get_stats()
    embeddings_after_phase2 = stats_after_phase2.get('embeddings', 0)

    # CRITICAL ASSERTION THAT MAY FAIL
    embeddings_generated = embeddings_after_phase2 - embeddings_after_phase1
    assert embeddings_generated > 0, \
        f"EXPECTED FAILURE: Should generate embeddings in phase 2, got {embeddings_generated}"


if __name__ == "__main__":
    # Run tests directly for debugging
    import subprocess
    subprocess.run(["python", "-m", "pytest", __file__, "-v", "--tb=short"])
