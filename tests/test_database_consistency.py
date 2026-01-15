"""Database consistency tests for ChunkHound.

These tests verify that the database maintains consistency between files, chunks, 
and embeddings. They should be added to the regular test suite to catch 
consistency issues early.
"""

import pytest
import asyncio
from pathlib import Path

from chunkhound.database_factory import create_services
from chunkhound.core.config.config import Config
from .test_utils import get_api_key_for_tests, get_embedding_config_for_tests, build_embedding_config_from_dict


@pytest.fixture
async def consistency_services(tmp_path):
    """Create database services for consistency testing."""
    db_path = tmp_path / "consistency_test.duckdb"

    # Get embedding config using centralized helper
    config_dict = get_embedding_config_for_tests()
    embedding_config = build_embedding_config_from_dict(config_dict)

    # Standard config creation
    config = Config(
        database={"path": str(db_path), "provider": "duckdb"},
        embedding=embedding_config
    )
    # Set target_dir after initialization since it's an excluded field
    config.target_dir = tmp_path

    # Standard service creation
    services = create_services(db_path, config)
    yield services


@pytest.mark.skipif(get_api_key_for_tests()[0] is None, reason="No API key available")
@pytest.mark.asyncio
async def test_database_embedding_chunk_consistency(consistency_services, tmp_path):
    """Test that database maintains consistency between chunks and embeddings."""
    services = consistency_services
    
    # Create test files
    test_files = []
    for i in range(3):
        test_file = tmp_path / f"consistency_{i}.py"
        test_file.write_text(f"""
def function_{i}():
    '''Function {i} for consistency testing.'''
    return "content {i}"

class Class_{i}:
    def method_{i}(self):
        return "method {i}"
""")
        test_files.append(test_file)
    
    # Process files with embeddings
    total_chunks_expected = 0
    for test_file in test_files:
        result = await services.indexing_coordinator.process_file(test_file, skip_embeddings=False)
        assert result['status'] == 'success'
        total_chunks_expected += result['chunks']
    
    # Wait for any async processing
    await asyncio.sleep(2.0)
    
    # Get database statistics
    stats = await services.indexing_coordinator.get_stats()
    
    # Verify basic counts
    assert stats['files'] == len(test_files)
    assert stats['chunks'] == total_chunks_expected
    
    # Critical consistency check
    if 'total_embeddings' in stats:
        assert stats.get('embeddings', 0) == stats['chunks'], \
            f"Embedding count ({stats.get('embeddings', 0)}) should match chunk count ({stats['chunks']})"


@pytest.mark.skipif(get_api_key_for_tests()[0] is None, reason="No API key available")
@pytest.mark.asyncio 
async def test_orphaned_embeddings_cleanup(consistency_services, tmp_path):
    """Test that no orphaned embeddings exist in the database."""
    services = consistency_services
    
    # Create and process a test file
    test_file = tmp_path / "orphan_test.py"
    test_file.write_text("""
def orphan_test():
    return "test"
""")
    
    result = await services.indexing_coordinator.process_file(test_file, skip_embeddings=False)
    assert result['status'] == 'success'
    assert result.get('embeddings_skipped', True) == False, "Should not skip embeddings"

    # Wait for async embedding processing
    await asyncio.sleep(3.0)

    # Also call generate_missing_embeddings to ensure embeddings are created
    embedding_result = await services.indexing_coordinator.generate_missing_embeddings()
    if embedding_result['status'] == 'success':
        # If we generated embeddings, wait a bit more
        await asyncio.sleep(1.0)
    
    # Query database directly to check for orphaned embeddings
    # This would need to be implemented based on the actual database schema
    db = services.provider
    
    # Get chunk count
    chunks_query = "SELECT COUNT(*) FROM chunks"
    chunk_result = db.execute_query(chunks_query)
    chunk_count = list(chunk_result[0].values())[0] if chunk_result else 0
    
    # Get embedding count (dynamically discover embeddings table)
    try:
        # Discover which embeddings table exists (embeddings_768, embeddings_1024, embeddings_1536, etc.)
        tables_query = "SHOW TABLES"
        tables_result = db.execute_query(tables_query)
        embedding_table = None

        for row in tables_result:
            table_name = row.get('name') or list(row.values())[0]
            if table_name.startswith('embeddings_'):
                embedding_table = table_name
                break

        if not embedding_table:
            pytest.skip("No embeddings table found - embeddings may not be enabled")

        # Get embedding count from discovered table
        embeddings_query = f"SELECT COUNT(*) FROM {embedding_table}"
        embedding_result = db.execute_query(embeddings_query)
        embedding_count = list(embedding_result[0].values())[0] if embedding_result else 0

        # Check for orphaned embeddings (embeddings without corresponding chunks)
        orphaned_query = f"""
            SELECT COUNT(*) as orphaned_count
            FROM {embedding_table} e
            LEFT JOIN chunks c ON e.chunk_id = c.id
            WHERE c.id IS NULL
        """
        orphaned_result = db.execute_query(orphaned_query)
        orphaned_count = orphaned_result[0]['orphaned_count'] if orphaned_result else 0

        assert orphaned_count == 0, f"Found {orphaned_count} orphaned embeddings"

        # Verify embedding/chunk consistency
        assert embedding_count == chunk_count, \
            f"Embedding count ({embedding_count}) should match chunk count ({chunk_count})"

    except Exception as e:
        # If embeddings table doesn't exist or query fails, that's also a consistency issue
        pytest.fail(f"Could not verify embedding consistency: {e}")


@pytest.mark.skipif(get_api_key_for_tests()[0] is None, reason="No API key available")
@pytest.mark.asyncio
async def test_file_deletion_cleanup(consistency_services, tmp_path):
    """Test that file deletion properly cleans up chunks and embeddings."""
    services = consistency_services
    
    # Create and process a test file
    test_file = tmp_path / "deletion_test.py"
    test_file.write_text("""
def deletion_test():
    return "will be deleted"

class DeletionClass:
    def method(self):
        return "also deleted"
""")
    
    # Process file
    result = await services.indexing_coordinator.process_file(test_file, skip_embeddings=False)
    assert result['status'] == 'success'
    chunks_created = result['chunks']
    
    # Wait for processing
    await asyncio.sleep(1.0)
    
    # Get initial stats
    initial_stats = await services.indexing_coordinator.get_stats()
    
    # Delete the file and remove from database
    chunks_removed = await services.indexing_coordinator.remove_file(str(test_file))
    assert chunks_removed == chunks_created
    
    # Wait for cleanup
    await asyncio.sleep(1.0)
    
    # Get final stats
    final_stats = await services.indexing_coordinator.get_stats()
    
    # Verify cleanup
    assert final_stats['files'] == initial_stats['files'] - 1
    assert final_stats['chunks'] == initial_stats['chunks'] - chunks_created
    
    # Verify embeddings were also cleaned up
    if 'total_embeddings' in initial_stats and 'total_embeddings' in final_stats:
        embeddings_removed = initial_stats['total_embeddings'] - final_stats['total_embeddings']
        assert embeddings_removed == chunks_created, \
            f"Should remove {chunks_created} embeddings, removed {embeddings_removed}"


@pytest.mark.skipif(get_api_key_for_tests()[0] is None, reason="No API key available")
@pytest.mark.asyncio
async def test_database_state_after_processing(consistency_services, tmp_path):
    """Test database state verification after file processing."""
    services = consistency_services
    
    # Process multiple files
    processed_files = 0
    total_chunks = 0
    
    for i in range(5):
        test_file = tmp_path / f"state_test_{i}.py"
        test_file.write_text(f"""
def state_function_{i}():
    '''Function {i}.'''
    return "state test {i}"
""")
        
        result = await services.indexing_coordinator.process_file(test_file, skip_embeddings=False)
        if result['status'] == 'success':
            processed_files += 1
            total_chunks += result['chunks']
    
    # Wait for processing
    await asyncio.sleep(2.0)
    
    # Verify database state
    stats = await services.indexing_coordinator.get_stats()
    
    assert stats['files'] >= processed_files
    assert stats['chunks'] >= total_chunks
    
    # State consistency checks
    assert stats['files'] > 0, "Should have processed files"
    assert stats['chunks'] > 0, "Should have created chunks"
    
    # If embeddings are tracked, verify consistency
    if 'total_embeddings' in stats:
        assert stats.get('embeddings', 0) <= stats['chunks'], \
            "Embeddings should not exceed chunks"


@pytest.mark.skipif(get_api_key_for_tests()[0] is None, reason="No API key available")
@pytest.mark.asyncio
async def test_concurrent_processing_consistency(consistency_services, tmp_path):
    """Test that concurrent file processing maintains database consistency."""
    services = consistency_services
    
    # Create multiple test files
    test_files = []
    for i in range(10):
        test_file = tmp_path / f"concurrent_{i}.py"
        test_file.write_text(f"""
def concurrent_function_{i}():
    return "concurrent {i}"
""")
        test_files.append(test_file)
    
    # Process files concurrently
    tasks = []
    for test_file in test_files:
        task = services.indexing_coordinator.process_file(test_file, skip_embeddings=False)
        tasks.append(task)
    
    # Wait for all processing to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count successful processing
    successful_results = [r for r in results if isinstance(r, dict) and r.get('status') == 'success']
    total_chunks_expected = sum(r['chunks'] for r in successful_results)
    
    # Wait for any async processing
    await asyncio.sleep(3.0)
    
    # Verify final state
    stats = await services.indexing_coordinator.get_stats()
    
    assert stats['chunks'] >= total_chunks_expected
    
    # Critical consistency check for concurrent processing
    if 'embeddings' in stats:
        assert stats['embeddings'] == stats['chunks'], \
            f"Concurrent processing broke embedding consistency: " \
            f"{stats['embeddings']} embeddings vs {stats['chunks']} chunks"


if __name__ == "__main__":
    # Run consistency tests
    import subprocess
    subprocess.run(["python", "-m", "pytest", __file__, "-v", "--tb=short"])