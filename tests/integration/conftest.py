"""Shared fixtures for integration tests."""

import pytest


@pytest.fixture
def lancedb_provider(tmp_path):
    """Create LanceDB provider with proper path transformation.

    This is the standard fixture for all LanceDB integration tests.
    Uses DatabaseConfig.get_db_path() to ensure correct .lancedb suffix.

    Usage:
        def test_something(lancedb_provider, tmp_path):
            # lancedb_provider is already connected and ready
            chunks = lancedb_provider.get_chunks_by_file_id(file_id)
    """
    # Skip if lancedb not installed
    pytest.importorskip("lancedb")

    from chunkhound.core.config.database_config import DatabaseConfig
    from chunkhound.providers.database.lancedb_provider import LanceDBProvider

    # Use DatabaseConfig for proper path transformation
    config = DatabaseConfig(path=tmp_path, provider="lancedb")
    db_path = config.get_db_path()  # Returns: tmp_path/lancedb.lancedb

    provider = LanceDBProvider(str(db_path), base_directory=tmp_path)
    provider.connect()

    yield provider

    provider.disconnect()


@pytest.fixture
def fragmented_lancedb_provider(lancedb_provider, tmp_path):
    """LanceDB provider with 50 fragments (below compaction threshold).

    Simulates real-world database state with accumulated fragments
    from multiple file updates. This is where the deduplication bug
    manifests in production.

    Usage:
        def test_search_with_fragments(fragmented_lancedb_provider):
            results, _ = fragmented_lancedb_provider.search_regex("pattern")
            # Verify no duplicate chunk_ids
    """
    from tests.fixtures.fragmentation_helpers import create_fragmented_state

    # Create 50 fragments (below 100 compaction threshold)
    create_fragmented_state(lancedb_provider, num_fragments=50, base_path=tmp_path)

    yield lancedb_provider


@pytest.fixture
def heavily_fragmented_lancedb_provider(lancedb_provider, tmp_path):
    """LanceDB provider with 100+ fragments (at compaction threshold).

    Tests edge case behavior at the compaction threshold boundary.

    Usage:
        def test_search_with_heavy_fragmentation(heavily_fragmented_lancedb_provider):
            results, _ = heavily_fragmented_lancedb_provider.search_regex("pattern")
    """
    from tests.fixtures.fragmentation_helpers import create_fragmented_state

    # Create 120 fragments (above 100 compaction threshold)
    create_fragmented_state(lancedb_provider, num_fragments=120, base_path=tmp_path)

    yield lancedb_provider


@pytest.fixture
def provider_with_duplicate_chunks(lancedb_provider):
    """LanceDB provider with deliberately created duplicate chunks across fragments.

    Directly simulates the bug condition for regression testing.

    Usage:
        def test_deduplication_works(provider_with_duplicate_chunks):
            results, _ = provider_with_duplicate_chunks.search_regex("test_function")
            chunk_ids = [r["chunk_id"] for r in results]
            assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates
    """
    from tests.fixtures.fragmentation_helpers import (
        create_file_with_duplicates_across_fragments,
    )

    # Create file and update it 5 times (creates 5+ fragments with same content)
    create_file_with_duplicates_across_fragments(
        lancedb_provider,
        file_path="test_duplicate.py",
        num_updates=5,
    )

    yield lancedb_provider
