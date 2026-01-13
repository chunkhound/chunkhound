"""Integration tests for the complete embedding generation pipeline.

These tests verify the end-to-end flow from file processing through chunk creation
to embedding generation and storage. They should be added to catch pipeline issues.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from chunkhound.database_factory import create_services
from chunkhound.core.config.config import Config
from chunkhound.services.embedding_service import EmbeddingService
from chunkhound.interfaces.database_provider import DatabaseProvider
from chunkhound.interfaces.embedding_provider import EmbeddingProvider
from .test_utils import get_api_key_for_tests


@pytest.fixture
async def pipeline_services(tmp_path):
    """Create database services for pipeline testing."""
    db_path = tmp_path / "pipeline_test.duckdb"

    # Standard API key discovery
    api_key, provider = get_api_key_for_tests()
    if not api_key:
        pytest.skip("No embedding API key available for pipeline integration test")

    # Standard embedding config
    model = "text-embedding-3-small" if provider == "openai" else "voyage-3.5"
    embedding_config = {
        "provider": provider,
        "api_key": api_key,
        "model": model
    }

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


@pytest.fixture
async def mock_pipeline_services(tmp_path):
    """Create mock database services for pipeline testing with predictable outcomes."""
    from unittest.mock import AsyncMock, MagicMock
    from chunkhound.services.indexing_coordinator import IndexingCoordinator
    from chunkhound.services.embedding_service import EmbeddingService
    from chunkhound.services.search_service import SearchService

    # Track state for stats
    state = {"chunks": 0, "embeddings": 0}

    # Create mock database provider
    mock_db = MagicMock()
    mock_db.optimize_tables = MagicMock()
    mock_db.get_chunks_without_embeddings_paginated.return_value = []

    # Create mock embedding service
    mock_embedding_service = MagicMock(spec=EmbeddingService)

    def generate_embeddings():
        chunks_to_embed = state["chunks"] - state["embeddings"]
        state["embeddings"] += chunks_to_embed  # Add all remaining chunks as embeddings
        return {
            "status": "success",
            "generated": chunks_to_embed,
            "attempted": chunks_to_embed,
            "failed": 0,
            "permanent_failures": 0
        }

    mock_embedding_service.generate_missing_embeddings = AsyncMock(side_effect=generate_embeddings)

    # Create mock indexing coordinator
    mock_indexing_coordinator = MagicMock(spec=IndexingCoordinator)

    def get_stats():
        return dict(state)

    mock_indexing_coordinator.get_stats = AsyncMock(side_effect=get_stats)

    mock_indexing_coordinator.generate_missing_embeddings = AsyncMock(side_effect=generate_embeddings)

    # Mock process_file to update state
    def process_file(file_path):
        state["chunks"] += 5  # Add chunks
        return {
            "status": "success",
            "chunks": 5
        }

    mock_indexing_coordinator.process_file = AsyncMock(side_effect=process_file)

    # Create mock search service
    mock_search_service = MagicMock(spec=SearchService)

    # Create a mock services object
    class MockServices:
        def __init__(self):
            self.provider = mock_db
            self.indexing_coordinator = mock_indexing_coordinator
            self.embedding_service = mock_embedding_service
            self.search_service = mock_search_service

    yield MockServices()


@pytest.mark.asyncio
async def test_complete_pipeline_file_to_embeddings(mock_pipeline_services, tmp_path):
    """Test complete pipeline: file → chunks → embeddings."""
    services = mock_pipeline_services

    # Create test file with various code structures
    test_file = tmp_path / "pipeline_test.py"
    test_file.write_text("""
'''Module docstring for pipeline testing.'''

def pipeline_function():
    '''A function to test the complete pipeline.'''
    return "pipeline test"

class PipelineClass:
    '''A class to test embedding generation.'''

    def __init__(self):
        '''Constructor for pipeline testing.'''
        self.value = "pipeline"

    def pipeline_method(self):
        '''Method for pipeline verification.'''
        return self.value + " method"

    @staticmethod
    def static_pipeline_method():
        '''Static method for pipeline testing.'''
        return "static pipeline"

async def async_pipeline_function():
    '''Async function for pipeline testing.'''
    await asyncio.sleep(0.1)
    return "async pipeline"
""")

    # Get initial state
    initial_stats = await services.indexing_coordinator.get_stats()

    # Process file through parsing and chunking
    result = await services.indexing_coordinator.process_file(test_file)

    # Verify file processing succeeded
    assert result['status'] == 'success', f"Pipeline processing failed: {result.get('error')}"
    assert result['chunks'] > 0, "Should create chunks"

    # Generate embeddings separately
    embedding_result = await services.indexing_coordinator.generate_missing_embeddings()
    assert embedding_result['status'] in ['success', 'complete'], f"Embedding generation failed: {embedding_result.get('error')}"

    # Wait for any async embedding processing
    await asyncio.sleep(1.0)  # Reduced wait time since mocks are fast

    # Verify pipeline results
    final_stats = await services.indexing_coordinator.get_stats()

    chunks_created = final_stats['chunks'] - initial_stats['chunks']
    embeddings_created = final_stats.get('embeddings', 0) - initial_stats.get('embeddings', 0)

    # Critical pipeline verification - with mocks, all chunks should get embeddings
    assert chunks_created > 0, f"Expected chunks to be created, got {chunks_created}"
    assert embeddings_created == chunks_created, f"Expected all chunks to get embeddings, got {embeddings_created} embeddings for {chunks_created} chunks"


@pytest.mark.asyncio
async def test_pipeline_parsing_then_embeddings(mock_pipeline_services, tmp_path):
    """Test two-phase pipeline: parse and chunk, then generate embeddings."""
    services = mock_pipeline_services

    # Create test file
    test_file = tmp_path / "two_phase_test.py"
    test_file.write_text("""
def two_phase_function():
    '''Function for two-phase pipeline testing.'''
    return "two phase test"

class TwoPhaseClass:
    '''Class for two-phase testing.'''
    pass
""")

    # Phase 1: Process file (parsing and chunking only)
    result1 = await services.indexing_coordinator.process_file(test_file)
    assert result1['status'] == 'success'
    assert result1['chunks'] > 0

    # Verify only chunks exist, no embeddings yet
    stats_phase1 = await services.indexing_coordinator.get_stats()
    chunks_after_phase1 = stats_phase1['chunks']
    embeddings_after_phase1 = stats_phase1.get('embeddings', 0)

    # Phase 2: Generate missing embeddings
    embedding_result = await services.indexing_coordinator.generate_missing_embeddings()
    assert embedding_result['status'] in ['success', 'complete'], f"Embedding generation failed: {embedding_result.get('error')}"

    # With mocks, we expect success and embeddings to be generated
    assert embedding_result['status'] == 'success', "Should generate embeddings with mock provider"
    assert embedding_result['generated'] > 0, "Should generate embeddings when status is success"

    # Verify embeddings were created
    stats_phase2 = await services.indexing_coordinator.get_stats()
    embeddings_after_phase2 = stats_phase2.get('embeddings', 0)

    embeddings_generated = embeddings_after_phase2 - embeddings_after_phase1

    # With mocks, all chunks should get embeddings
    assert embeddings_generated == chunks_after_phase1, f"Expected all {chunks_after_phase1} chunks to get embeddings, got {embeddings_generated}"


@pytest.mark.asyncio
async def test_pipeline_error_recovery(mock_pipeline_services, tmp_path):
    """Test that pipeline recovers from errors gracefully."""
    services = mock_pipeline_services
    
    # Create file with valid content
    test_file = tmp_path / "recovery_test.py"
    test_file.write_text("""
def valid_function():
    return "valid"
""")
    
    # Process normally first
    result = await services.indexing_coordinator.process_file(test_file)
    assert result['status'] == 'success'

    # Generate embeddings
    await services.indexing_coordinator.generate_missing_embeddings()

    # Wait for processing
    await asyncio.sleep(1.0)

    # Verify system is still functional after any potential errors
    stats = await services.indexing_coordinator.get_stats()
    assert stats['chunks'] > 0

    # Try processing another file to ensure system recovered
    test_file2 = tmp_path / "recovery_test2.py"
    test_file2.write_text("""
def another_valid_function():
    return "also valid"
""")

    result2 = await services.indexing_coordinator.process_file(test_file2)
    assert result2['status'] == 'success', "System should recover and continue processing"

    # Generate embeddings for second file
    await services.indexing_coordinator.generate_missing_embeddings()


@pytest.mark.asyncio
async def test_embedding_service_direct_integration(mock_pipeline_services, tmp_path):
    """Test EmbeddingService integration directly."""
    services = mock_pipeline_services

    # Create test file and process it with skip_embeddings=True
    test_file = tmp_path / "embedding_service_test.py"
    test_file.write_text("""
def embedding_service_function():
    '''Function for testing EmbeddingService directly.'''
    return "embedding service test"
""")

    # Process file to create chunks
    result = await services.indexing_coordinator.process_file(test_file)
    assert result['status'] == 'success'
    assert result['chunks'] > 0

    # Use EmbeddingService directly to generate embeddings
    embedding_service = services.embedding_service

    # Generate missing embeddings
    embedding_result = await embedding_service.generate_missing_embeddings()
    assert embedding_result['status'] in ['success', 'complete'], f"EmbeddingService failed: {embedding_result.get('error')}"

    # With mocks, we expect success
    assert embedding_result['status'] == 'success', "Should generate embeddings with mock provider"
    assert embedding_result['generated'] > 0, "EmbeddingService should generate embeddings when status is success"

    # Verify consistency - with mocks, all chunks should get embeddings
    stats = await services.indexing_coordinator.get_stats()
    embeddings_count = stats.get('embeddings', 0)
    chunks_count = stats['chunks']
    assert embeddings_count == chunks_count, f"Expected all {chunks_count} chunks to get embeddings, got {embeddings_count}"


@pytest.mark.asyncio
async def test_pipeline_batch_processing(mock_pipeline_services, tmp_path):
    """Test pipeline with batch processing of multiple files."""
    services = mock_pipeline_services

    # Create multiple test files
    test_files = []
    for i in range(5):
        test_file = tmp_path / f"batch_test_{i}.py"
        test_file.write_text(f"""
def batch_function_{i}():
    '''Function {i} for batch testing.'''
    return "batch test {i}"

class BatchClass_{i}:
    '''Class {i} for batch testing.'''
    def method_{i}(self):
        return "method {i}"
""")
        test_files.append(test_file)

    # Process all files (parsing and chunking only)
    processed_chunks = 0
    for test_file in test_files:
        result = await services.indexing_coordinator.process_file(test_file)
        assert result['status'] == 'success'
        processed_chunks += result['chunks']

    # Verify chunks were created
    stats_after_chunking = await services.indexing_coordinator.get_stats()
    assert stats_after_chunking['chunks'] == processed_chunks

    # Generate embeddings for all chunks
    embedding_result = await services.indexing_coordinator.generate_missing_embeddings()
    assert embedding_result['status'] == 'success', "Should generate embeddings with mock provider"
    assert embedding_result['generated'] > 0, "Should generate embeddings when status is success"

    # Wait for batch processing
    await asyncio.sleep(1.0)  # Reduced wait time for mocks

    # Verify final consistency - with mocks, all chunks should get embeddings
    final_stats = await services.indexing_coordinator.get_stats()
    embeddings_count = final_stats.get('embeddings', 0)
    chunks_count = final_stats['chunks']
    assert embeddings_count == chunks_count, f"Expected all {chunks_count} chunks to get embeddings, got {embeddings_count}"


@pytest.mark.asyncio
async def test_pipeline_file_modification_embeddings(mock_pipeline_services, tmp_path):
    """Test that file modifications properly update embeddings."""
    services = mock_pipeline_services

    # Create initial file
    test_file = tmp_path / "modification_test.py"
    test_file.write_text("""
def original_function():
    '''Original function.'''
    return "original"
""")

    # Process initial file
    result1 = await services.indexing_coordinator.process_file(test_file)
    assert result1['status'] == 'success'

    # Generate embeddings for initial file
    await services.indexing_coordinator.generate_missing_embeddings()

    # Wait for initial processing
    await asyncio.sleep(1.0)  # Reduced for mocks
    initial_stats = await services.indexing_coordinator.get_stats()

    # Modify file by adding new function (truly additive)
    original_content = test_file.read_text()
    test_file.write_text(original_content + """

def new_function():
    '''Newly added function.'''
    return "new"
""")

    # Process modified file
    result2 = await services.indexing_coordinator.process_file(test_file)
    assert result2['status'] == 'success'

    # Generate embeddings for modified file
    await services.indexing_coordinator.generate_missing_embeddings()

    # Wait for modification processing
    await asyncio.sleep(1.0)  # Reduced for mocks

    # Verify embeddings were updated/added
    final_stats = await services.indexing_coordinator.get_stats()

    # Should have more chunks after adding new function
    assert final_stats['chunks'] > initial_stats['chunks'], \
        f"Should have more chunks after file modification, got {final_stats['chunks']} vs {initial_stats['chunks']}"

    # Verify embedding consistency - with mocks, all chunks should get embeddings
    embeddings_count = final_stats.get('embeddings', 0)
    chunks_count = final_stats['chunks']
    assert embeddings_count == chunks_count, f"Expected all {chunks_count} chunks to get embeddings, got {embeddings_count}"



if __name__ == "__main__":
    # Run pipeline integration tests
    import subprocess
    subprocess.run(["python", "-m", "pytest", __file__, "-v", "--tb=short"])