"""Integration tests for LanceDB embedding storage.

Tests that embeddings are properly generated and stored when using
the LanceDB backend, addressing the bug where .search() failed to find
chunks with NULL embedding columns.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

def test_lancedb_embeddings_stored_during_indexing(lancedb_provider, tmp_path):
    """Verify embeddings are stored in LanceDB during indexing with mock provider."""
    from chunkhound.core.models import Chunk
    from chunkhound.core.types.common import ChunkType, Language

    # Need to insert a file first for foreign key
    from chunkhound.core.models import File

    test_file = File(
        path="test.py",
        mtime=1234567890.0,
        language=Language.PYTHON,
        size_bytes=100,
    )
    file_id = lancedb_provider.insert_file(test_file)

    # Insert test chunks
    chunks = [
        Chunk(
            file_id=file_id,
            code="def hello(): pass",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol="hello",
        ),
        Chunk(
            file_id=file_id,
            code="def world(): return 42",
            start_line=2,
            end_line=2,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol="world",
        ),
    ]

    # Insert chunks (returns chunk IDs)
    chunk_ids = lancedb_provider.insert_chunks_batch(chunks)
    assert len(chunk_ids) == 2, "Should insert 2 chunks"

    # Verify chunks exist
    stats = lancedb_provider.get_stats()
    assert stats["chunks"] == 2, "Should have 2 chunks in database"

    # Create mock embeddings data
    embedding_dim = 8  # Small dimension for testing
    embeddings_data = [
        {
            "chunk_id": chunk_ids[0],
            "provider": "test",
            "model": "test-model",
            "dims": embedding_dim,
            "embedding": [0.1] * embedding_dim,
        },
        {
            "chunk_id": chunk_ids[1],
            "provider": "test",
            "model": "test-model",
            "dims": embedding_dim,
            "embedding": [0.2] * embedding_dim,
        },
    ]

    # Store embeddings
    stored_count = lancedb_provider.insert_embeddings_batch(embeddings_data)
    assert stored_count == 2, f"Should store 2 embeddings, got {stored_count}"

    # Verify embeddings are retrievable
    for i, chunk_id in enumerate(chunk_ids):
        embedding = lancedb_provider.get_embedding_by_chunk_id(chunk_id, "test", "test-model")
        assert embedding is not None, f"Embedding for chunk {chunk_id} should exist"
        assert embedding.vector is not None, f"Embedding vector should not be None"
        assert len(embedding.vector) == embedding_dim, f"Embedding should have {embedding_dim} dimensions"


def test_lancedb_indexing_flow_creates_chunks(lancedb_provider, tmp_path):
    """Test that LanceDB indexing flow creates chunks correctly.

    Note: This test verifies chunk creation only. For embedding storage tests,
    see test_lancedb_embeddings_stored_during_indexing and
    test_lancedb_embedding_update_finds_chunks which test the embedding
    insertion path directly without needing to inject mock providers.
    """
    from chunkhound.services.indexing_coordinator import IndexingCoordinator

    # Create a test Python file
    test_file = tmp_path / "test_module.py"
    test_file.write_text("""
def greet(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y
""")

    # Create coordinator without embedding provider (skip embeddings)
    coord = IndexingCoordinator(
        database_provider=lancedb_provider,
        base_directory=tmp_path,
        embedding_provider=None,  # No embeddings - just test chunk creation
    )

    # Process the file
    result = asyncio.run(
        coord.process_directory(tmp_path, patterns=["**/*.py"], exclude_patterns=[])
    )

    assert result["files_processed"] == 1, "Should process 1 file"

    # Verify chunks were created
    stats = lancedb_provider.get_stats()
    assert stats["chunks"] > 0, "Should have chunks in database"

    # The embedding tests (test_lancedb_embedding_update_finds_chunks) verify
    # that the fix for NULL embedding column search works correctly


def test_lancedb_embedding_update_finds_chunks(lancedb_provider, tmp_path):
    """Test that embedding update correctly finds chunks with NULL embeddings.

    This specifically tests the fix for the bug where .search() failed to
    find chunks with NULL embedding columns.
    """
    from chunkhound.core.models import Chunk
    from chunkhound.core.types.common import ChunkType, Language

    # Need to insert a file first for foreign key
    from chunkhound.core.models import File

    test_file = File(
        path="test.py",
        mtime=1234567890.0,
        language=Language.PYTHON,
        size_bytes=100,
    )
    file_id = lancedb_provider.insert_file(test_file)

    # Insert chunks with NULL embeddings (default state)
    chunks = []
    for i in range(10):
        chunks.append(
            Chunk(
                file_id=file_id,
                code=f"def func_{i}(): return {i}",
                start_line=i + 1,  # Line numbers must be positive (start at 1)
                end_line=i + 1,
                chunk_type=ChunkType.FUNCTION,
                language=Language.PYTHON,
                symbol=f"func_{i}",
            )
        )

    chunk_ids = lancedb_provider.insert_chunks_batch(chunks)
    assert len(chunk_ids) == 10, "Should insert 10 chunks"

    # Verify chunks were inserted with NULL embeddings
    stats_before = lancedb_provider.get_stats()
    assert stats_before["chunks"] == 10

    # Now try to add embeddings (this is where the bug manifested)
    embedding_dim = 16
    embeddings_data = [
        {
            "chunk_id": cid,
            "provider": "test",
            "model": "test-model",
            "dims": embedding_dim,
            "embedding": [float(i) / 100] * embedding_dim,
        }
        for i, cid in enumerate(chunk_ids)
    ]

    # This should succeed with the fix
    stored_count = lancedb_provider.insert_embeddings_batch(embeddings_data)

    # THE KEY ASSERTION: All embeddings should be stored
    assert stored_count == 10, (
        f"Should store all 10 embeddings, but only stored {stored_count}. "
        "This indicates the chunk lookup in insert_embeddings_batch failed."
    )

    # Verify each embedding is retrievable
    for chunk_id in chunk_ids:
        emb = lancedb_provider.get_embedding_by_chunk_id(chunk_id, "test", "test-model")
        assert emb is not None, f"Embedding for chunk {chunk_id} should be retrievable"
        assert emb.vector is not None, f"Embedding vector should not be None"


def test_lancedb_find_similar_chunks_basic(lancedb_provider, tmp_path):
    """Test that find_similar_chunks returns similar chunks ranked by score."""
    from chunkhound.core.models import Chunk, File
    from chunkhound.core.types.common import ChunkType, Language

    # Insert test file
    test_file = File(
        path="test.py",
        mtime=1234567890.0,
        language=Language.PYTHON,
        size_bytes=100,
    )
    file_id = lancedb_provider.insert_file(test_file)

    # Insert test chunks
    chunks = [
        Chunk(
            file_id=file_id,
            code="def authenticate_user(username, password): pass",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol="authenticate_user",
        ),
        Chunk(
            file_id=file_id,
            code="def login_user(credentials): pass",
            start_line=2,
            end_line=2,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol="login_user",
        ),
        Chunk(
            file_id=file_id,
            code="def calculate_taxes(income): pass",
            start_line=3,
            end_line=3,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol="calculate_taxes",
        ),
    ]

    chunk_ids = lancedb_provider.insert_chunks_batch(chunks)

    # Add embeddings with different similarity levels
    # chunk 0 and 1 are similar (auth-related), chunk 2 is different
    embedding_dim = 8
    embeddings_data = [
        {
            "chunk_id": chunk_ids[0],
            "provider": "test",
            "model": "test-model",
            "dims": embedding_dim,
            "embedding": [0.9, 0.8, 0.7, 0.6, 0.1, 0.1, 0.1, 0.1],  # Auth-related
        },
        {
            "chunk_id": chunk_ids[1],
            "provider": "test",
            "model": "test-model",
            "dims": embedding_dim,
            "embedding": [0.8, 0.9, 0.6, 0.7, 0.1, 0.1, 0.1, 0.1],  # Similar to chunk 0
        },
        {
            "chunk_id": chunk_ids[2],
            "provider": "test",
            "model": "test-model",
            "dims": embedding_dim,
            "embedding": [0.1, 0.1, 0.1, 0.1, 0.9, 0.8, 0.7, 0.6],  # Different domain
        },
    ]

    lancedb_provider.insert_embeddings_batch(embeddings_data)

    # Find similar chunks to chunk 0
    similar = lancedb_provider.find_similar_chunks(
        chunk_id=chunk_ids[0],
        provider="test",
        model="test-model",
        limit=10,
    )

    # Should return chunks 1 and 2 (excluding source chunk 0)
    assert len(similar) == 2, f"Should find 2 similar chunks, got {len(similar)}"

    # Results should be ranked by similarity score
    assert similar[0]["chunk_id"] == chunk_ids[1], "Most similar should be chunk 1"
    assert similar[1]["chunk_id"] == chunk_ids[2], "Least similar should be chunk 2"

    # Scores should be in descending order
    assert similar[0]["score"] > similar[1]["score"], "Scores should be ranked"


def test_lancedb_find_similar_chunks_excludes_source(lancedb_provider, tmp_path):
    """Test that source chunk is excluded from results."""
    from chunkhound.core.models import Chunk, File
    from chunkhound.core.types.common import ChunkType, Language

    # Insert test file
    test_file = File(
        path="test.py",
        mtime=1234567890.0,
        language=Language.PYTHON,
        size_bytes=100,
    )
    file_id = lancedb_provider.insert_file(test_file)

    # Insert chunks
    chunks = [
        Chunk(
            file_id=file_id,
            code=f"def func_{i}(): pass",
            start_line=i + 1,
            end_line=i + 1,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol=f"func_{i}",
        )
        for i in range(3)
    ]

    chunk_ids = lancedb_provider.insert_chunks_batch(chunks)

    # Add identical embeddings (all should be equally similar)
    embedding_dim = 8
    embeddings_data = [
        {
            "chunk_id": cid,
            "provider": "test",
            "model": "test-model",
            "dims": embedding_dim,
            "embedding": [0.5] * embedding_dim,
        }
        for cid in chunk_ids
    ]

    lancedb_provider.insert_embeddings_batch(embeddings_data)

    # Find similar chunks to chunk 0
    similar = lancedb_provider.find_similar_chunks(
        chunk_id=chunk_ids[0],
        provider="test",
        model="test-model",
        limit=10,
    )

    # Should return 2 chunks (excluding source)
    assert len(similar) == 2, "Should exclude source chunk"

    # Source chunk should NOT be in results
    result_ids = [r["chunk_id"] for r in similar]
    assert chunk_ids[0] not in result_ids, "Source chunk should be excluded"
    assert chunk_ids[1] in result_ids, "Chunk 1 should be included"
    assert chunk_ids[2] in result_ids, "Chunk 2 should be included"


def test_lancedb_find_similar_chunks_no_embedding(lancedb_provider, tmp_path):
    """Test that empty list is returned when chunk has no embedding."""
    from chunkhound.core.models import Chunk, File
    from chunkhound.core.types.common import ChunkType, Language

    # Insert test file
    test_file = File(
        path="test.py",
        mtime=1234567890.0,
        language=Language.PYTHON,
        size_bytes=100,
    )
    file_id = lancedb_provider.insert_file(test_file)

    # Insert chunk WITHOUT embedding
    chunks = [
        Chunk(
            file_id=file_id,
            code="def test(): pass",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol="test",
        )
    ]

    chunk_ids = lancedb_provider.insert_chunks_batch(chunks)

    # Try to find similar chunks (should return empty list)
    similar = lancedb_provider.find_similar_chunks(
        chunk_id=chunk_ids[0],
        provider="test",
        model="test-model",
        limit=10,
    )

    assert similar == [], "Should return empty list when no embedding exists"


def test_lancedb_find_similar_chunks_threshold(lancedb_provider, tmp_path):
    """Test that similarity threshold filters results correctly."""
    from chunkhound.core.models import Chunk, File
    from chunkhound.core.types.common import ChunkType, Language

    # Insert test file
    test_file = File(
        path="test.py",
        mtime=1234567890.0,
        language=Language.PYTHON,
        size_bytes=100,
    )
    file_id = lancedb_provider.insert_file(test_file)

    # Insert chunks
    chunks = [
        Chunk(
            file_id=file_id,
            code=f"def func_{i}(): pass",
            start_line=i + 1,
            end_line=i + 1,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol=f"func_{i}",
        )
        for i in range(3)
    ]

    chunk_ids = lancedb_provider.insert_chunks_batch(chunks)

    # Add embeddings with different similarities
    embedding_dim = 8
    embeddings_data = [
        {
            "chunk_id": chunk_ids[0],
            "provider": "test",
            "model": "test-model",
            "dims": embedding_dim,
            "embedding": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        {
            "chunk_id": chunk_ids[1],
            "provider": "test",
            "model": "test-model",
            "dims": embedding_dim,
            "embedding": [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Very similar
        },
        {
            "chunk_id": chunk_ids[2],
            "provider": "test",
            "model": "test-model",
            "dims": embedding_dim,
            "embedding": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Not similar
        },
    ]

    lancedb_provider.insert_embeddings_batch(embeddings_data)

    # Find with high threshold (should filter out dissimilar chunks)
    similar = lancedb_provider.find_similar_chunks(
        chunk_id=chunk_ids[0],
        provider="test",
        model="test-model",
        limit=10,
        threshold=0.8,  # High threshold - only very similar chunks
    )

    # Should only return highly similar chunk 1
    assert len(similar) >= 1, "Should find at least one similar chunk"
    assert all(r["score"] >= 0.8 for r in similar), "All results should meet threshold"


def test_lancedb_find_similar_chunks_limit(lancedb_provider, tmp_path):
    """Test that result limit is respected."""
    from chunkhound.core.models import Chunk, File
    from chunkhound.core.types.common import ChunkType, Language

    # Insert test file
    test_file = File(
        path="test.py",
        mtime=1234567890.0,
        language=Language.PYTHON,
        size_bytes=100,
    )
    file_id = lancedb_provider.insert_file(test_file)

    # Insert many chunks
    chunks = [
        Chunk(
            file_id=file_id,
            code=f"def func_{i}(): pass",
            start_line=i + 1,
            end_line=i + 1,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol=f"func_{i}",
        )
        for i in range(10)
    ]

    chunk_ids = lancedb_provider.insert_chunks_batch(chunks)

    # Add similar embeddings to all
    embedding_dim = 8
    embeddings_data = [
        {
            "chunk_id": cid,
            "provider": "test",
            "model": "test-model",
            "dims": embedding_dim,
            "embedding": [0.5 + i * 0.01 for i in range(embedding_dim)],
        }
        for cid in chunk_ids
    ]

    lancedb_provider.insert_embeddings_batch(embeddings_data)

    # Find with small limit
    limit = 3
    similar = lancedb_provider.find_similar_chunks(
        chunk_id=chunk_ids[0],
        provider="test",
        model="test-model",
        limit=limit,
    )

    assert len(similar) == limit, f"Should respect limit of {limit}"


def test_lancedb_find_similar_chunks_wrong_provider(lancedb_provider, tmp_path):
    """Test that empty list is returned for non-existent provider/model."""
    from chunkhound.core.models import Chunk, File
    from chunkhound.core.types.common import ChunkType, Language

    # Insert test file
    test_file = File(
        path="test.py",
        mtime=1234567890.0,
        language=Language.PYTHON,
        size_bytes=100,
    )
    file_id = lancedb_provider.insert_file(test_file)

    # Insert chunk with embedding
    chunks = [
        Chunk(
            file_id=file_id,
            code="def test(): pass",
            start_line=1,
            end_line=1,
            chunk_type=ChunkType.FUNCTION,
            language=Language.PYTHON,
            symbol="test",
        )
    ]

    chunk_ids = lancedb_provider.insert_chunks_batch(chunks)

    # Add embedding with specific provider/model
    embedding_dim = 8
    embeddings_data = [
        {
            "chunk_id": chunk_ids[0],
            "provider": "test",
            "model": "test-model",
            "dims": embedding_dim,
            "embedding": [0.5] * embedding_dim,
        }
    ]

    lancedb_provider.insert_embeddings_batch(embeddings_data)

    # Try to find with wrong provider/model
    similar = lancedb_provider.find_similar_chunks(
        chunk_id=chunk_ids[0],
        provider="wrong-provider",
        model="wrong-model",
        limit=10,
    )

    assert similar == [], "Should return empty list for non-existent provider/model"
