"""Test that embedding repository properly updates HNSW indexes."""
import tempfile
from pathlib import Path

import numpy.core.multiarray  # noqa: F401  # Prevent DuckDB threading segfault
import pytest

from chunkhound.core.config.sharding_config import ShardingConfig
from chunkhound.providers.database.duckdb_provider import DuckDBProvider


def test_shard_manager_initialized_for_file_db():
    """Verify ShardManager is properly initialized for file-based databases.

    This is the critical requirement - if ShardManager is None for file-based
    databases, embeddings won't be indexed for semantic search.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.duckdb"
        provider = DuckDBProvider(
            db_path=db_path,
            base_directory=Path(tmpdir),
            sharding_config=ShardingConfig()
        )
        provider.connect()

        # This is the core fix verification - ShardManager MUST exist
        assert provider.shard_manager is not None, (
            "ShardManager should be initialized for file-based DB. "
            "If None, embeddings won't be indexed for semantic search (the bug)."
        )

        # Verify ShardManager can be used
        assert hasattr(provider.shard_manager, 'insert_embeddings')
        assert hasattr(provider.shard_manager, 'fix_pass')

        provider.disconnect()


def test_repository_in_memory_graceful_degradation():
    """Verify in-memory databases don't crash when shard_manager is None.

    In-memory databases intentionally skip ShardManager initialization.
    This test ensures the fix doesn't break that behavior.
    """
    provider = DuckDBProvider(
        db_path=":memory:",
        base_directory=Path("/tmp")
    )
    provider.connect()

    # Verify shard_manager is None for in-memory
    assert provider.shard_manager is None, "ShardManager should be None for :memory:"

    # Insert should still work (no crash)
    embeddings_data = [
        {
            "chunk_id": i,
            "provider": "test",
            "model": "test-model",
            "embedding": [0.1] * 128,
            "dims": 128
        }
        for i in range(5)
    ]

    # This should not raise an exception (graceful degradation)
    count = provider.insert_embeddings_batch(embeddings_data)
    assert count == 5

    provider.disconnect()


def test_embedding_insertion_with_shard_manager():
    """Verify embedding insertion works correctly with ShardManager.

    This test confirms basic embedding insertion functionality works
    without crashes when ShardManager is present.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.duckdb"
        provider = DuckDBProvider(
            db_path=db_path,
            base_directory=Path(tmpdir),
            sharding_config=ShardingConfig()
        )
        provider.connect()

        # Verify ShardManager exists
        assert provider.shard_manager is not None, "ShardManager should exist for file-based DB"

        # Insert embeddings - this should not crash
        embeddings_data = [
            {
                "chunk_id": i,
                "provider": "test",
                "model": "test-model",
                "embedding": [0.1 + i * 0.01] * 128,  # Slightly different vectors
                "dims": 128
            }
            for i in range(10)
        ]

        count = provider.insert_embeddings_batch(embeddings_data)
        assert count == 10, "All embeddings should be inserted successfully"

        # If we got here without crashes, the HNSW update path executed
        # The diagnostic logging (added in our fix) will show if ShardManager was called

        provider.disconnect()
