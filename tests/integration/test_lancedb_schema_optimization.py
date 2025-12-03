"""Integration tests for LanceDB proactive schema creation optimization.

Tests that verify the fix for the table recreation performance issue where
chunks table was created with variable-size schema then recreated when first
embeddings arrived, causing O(n) migration overhead.
"""

import pyarrow as pa
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.embeddings import EmbeddingManager
from chunkhound.providers.database.lancedb_provider import LanceDBProvider


def test_schema_created_with_fixed_dimensions_when_provider_available(tmp_path):
    """Verify table created with fixed-size schema when embedding provider available.

    This is the optimization: when embedding_manager has a provider with known
    dimensions, the chunks table should be created with fixed-size schema from
    the start, avoiding the need for migration later.
    """
    pytest.importorskip("lancedb")

    # Create embedding manager with mock provider
    em = EmbeddingManager()
    mock_provider = MagicMock()
    mock_provider.name = "test"
    mock_provider.model = "test-model"
    mock_provider.dims = 768
    mock_provider.distance = "cosine"
    em.register_provider(mock_provider, set_default=True)

    # Create provider with embedding_manager
    config = DatabaseConfig(path=tmp_path, provider="lancedb")
    db_path = config.get_db_path()

    provider = LanceDBProvider(
        str(db_path),
        base_directory=tmp_path,
        embedding_manager=em
    )
    provider.connect()

    try:
        # Verify schema has fixed-size embedding field
        schema = provider._chunks_table.schema
        embedding_field = next(f for f in schema if f.name == "embedding")

        assert pa.types.is_fixed_size_list(embedding_field.type), \
            "Embedding field should be fixed-size list when provider available"
        assert embedding_field.type.list_size == 768, \
            f"Expected 768 dims from mock provider, got {embedding_field.type.list_size}"
    finally:
        provider.disconnect()


def test_schema_fallback_to_variable_size_without_provider(tmp_path):
    """Verify table created with variable-size schema when no embedding provider.

    This ensures backward compatibility: when no embedding_manager is provided,
    the system falls back to the original behavior of creating variable-size
    schema (which will be migrated later).
    """
    pytest.importorskip("lancedb")

    # Create provider WITHOUT embedding_manager
    config = DatabaseConfig(path=tmp_path, provider="lancedb")
    db_path = config.get_db_path()

    provider = LanceDBProvider(
        str(db_path),
        base_directory=tmp_path,
        embedding_manager=None  # No embedding manager
    )
    provider.connect()

    try:
        # Verify schema has variable-size embedding field
        schema = provider._chunks_table.schema
        embedding_field = next(f for f in schema if f.name == "embedding")

        assert not pa.types.is_fixed_size_list(embedding_field.type), \
            "Embedding field should be variable-size list when no provider configured"
    finally:
        provider.disconnect()


def test_no_migration_when_schema_matches_dimensions(tmp_path):
    """Verify no migration occurs when schema created with correct dimensions.

    This is the key performance benefit: when the schema is created with the
    correct dimensions upfront, inserting embeddings should NOT trigger the
    migration, as verified by the schema remaining unchanged.
    """
    pytest.importorskip("lancedb")

    # Setup: Create DB with fixed schema matching embedding dimensions
    em = EmbeddingManager()
    mock_provider = MagicMock()
    mock_provider.name = "test"
    mock_provider.model = "test-model"
    mock_provider.dims = 768
    em.register_provider(mock_provider, set_default=True)

    config = DatabaseConfig(path=tmp_path, provider="lancedb")
    db_path = config.get_db_path()

    provider = LanceDBProvider(
        str(db_path),
        base_directory=tmp_path,
        embedding_manager=em
    )
    provider.connect()

    try:
        # Verify schema created with fixed-size from the start
        initial_schema = provider._chunks_table.schema
        initial_embedding_field = next(f for f in initial_schema if f.name == "embedding")
        assert pa.types.is_fixed_size_list(initial_embedding_field.type), \
            "Initial schema should be fixed-size when provider configured"
        assert initial_embedding_field.type.list_size == 768, \
            "Initial schema should have correct dimensions from provider"

        # Insert file and chunks
        test_file = File(
            path="test.py",
            mtime=1234567890.0,
            language=Language.PYTHON,
            size_bytes=100,
        )
        file_id = provider.insert_file(test_file)

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
        chunk_ids = provider.insert_chunks_batch(chunks)

        # Insert embeddings (should NOT trigger migration)
        embeddings = [{
            "chunk_id": chunk_ids[0],
            "provider": "test",
            "model": "test-model",
            "embedding": [0.1] * 768,
            "dims": 768,
        }]

        provider.insert_embeddings_batch(embeddings)

        # Verify schema unchanged (no migration occurred)
        final_schema = provider._chunks_table.schema
        final_embedding_field = next(f for f in final_schema if f.name == "embedding")

        # Schema should remain the same (no recreation)
        assert pa.types.is_fixed_size_list(final_embedding_field.type), \
            "Schema should still be fixed-size (no migration occurred)"
        assert final_embedding_field.type.list_size == 768, \
            "Schema dimensions should remain 768 (no migration occurred)"

        # Verify embeddings were successfully inserted
        chunks_result = provider.get_chunks_by_file_id(file_id)
        assert len(chunks_result) == 1, "Chunk should exist"

    finally:
        provider.disconnect()


def test_migration_still_works_for_existing_databases(tmp_path):
    """Verify migration still works for databases created before the optimization.

    This ensures backward compatibility: databases created with the old code
    (variable-size schema) should still migrate correctly when embeddings arrive.
    """
    pytest.importorskip("lancedb")

    # Step 1: Create DB WITHOUT embedding manager (simulates old behavior)
    config = DatabaseConfig(path=tmp_path, provider="lancedb")
    db_path = config.get_db_path()

    provider = LanceDBProvider(
        str(db_path),
        base_directory=tmp_path,
        embedding_manager=None
    )
    provider.connect()

    # Verify initial schema is variable-size
    initial_schema = provider._chunks_table.schema
    initial_embedding_field = next(f for f in initial_schema if f.name == "embedding")
    assert not pa.types.is_fixed_size_list(initial_embedding_field.type), \
        "Initial schema should be variable-size when no provider configured"

    # Insert file and chunks
    test_file = File(
        path="test.py",
        mtime=1234567890.0,
        language=Language.PYTHON,
        size_bytes=100,
    )
    file_id = provider.insert_file(test_file)

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
    chunk_ids = provider.insert_chunks_batch(chunks)
    provider.disconnect()

    # Step 2: Reconnect with embedding manager and insert embeddings
    # This simulates the scenario where embedding provider is added later
    em = EmbeddingManager()
    mock_provider = MagicMock()
    mock_provider.name = "test"
    mock_provider.model = "test-model"
    mock_provider.dims = 768
    em.register_provider(mock_provider, set_default=True)

    provider = LanceDBProvider(
        str(db_path),
        base_directory=tmp_path,
        embedding_manager=em
    )
    provider.connect()

    try:
        # Insert embeddings (SHOULD trigger migration due to variable-size schema)
        embeddings = [{
            "chunk_id": chunk_ids[0],
            "provider": "test",
            "model": "test-model",
            "embedding": [0.1] * 768,
            "dims": 768,
        }]

        provider.insert_embeddings_batch(embeddings)

        # Verify migration occurred by checking schema is now fixed-size
        final_schema = provider._chunks_table.schema
        final_embedding_field = next(f for f in final_schema if f.name == "embedding")

        assert pa.types.is_fixed_size_list(final_embedding_field.type), \
            "Schema should be fixed-size after migration"
        assert final_embedding_field.type.list_size == 768, \
            f"Schema should have correct dimensions (768) after migration, got {final_embedding_field.type.list_size}"

        # Verify data integrity: chunk still exists with embedding
        chunks_result = provider.get_chunks_by_file_id(file_id)
        assert len(chunks_result) == 1, "Chunk should still exist after migration"

    finally:
        provider.disconnect()
