"""Unit tests for ShardManager heartbeat functionality."""

import pytest
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

from chunkhound.core.config.sharding_config import ShardingConfig
from chunkhound.providers.database.shard_manager import (
    ShardManager,
    HEARTBEAT_INTERVAL_INSERT_ROUTING,
    HEARTBEAT_INTERVAL_MERGE_ROUTING,
)


class TestShardManagerHeartbeat:
    """Test heartbeat callback functionality during long operations."""

    @pytest.fixture
    def mock_db(self, tmp_path):
        """Create mock database provider."""
        db = Mock()
        db.connection = Mock()
        return db

    @pytest.fixture
    def shard_manager(self, tmp_path, mock_db):
        """Create ShardManager with heartbeat tracking."""
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()

        config = ShardingConfig(
            split_threshold=100,
            merge_threshold=20,
        )

        heartbeat_mock = Mock()
        manager = ShardManager(
            db_provider=mock_db,
            shard_dir=shard_dir,
            config=config,
            heartbeat_callback=heartbeat_mock,
        )
        return manager

    def test_heartbeat_frequency_constants(self):
        """Verify heartbeat interval constants defined correctly."""
        assert HEARTBEAT_INTERVAL_INSERT_ROUTING == 100
        assert HEARTBEAT_INTERVAL_MERGE_ROUTING == 10_000
        # Verify merge interval is 100x insert interval
        assert HEARTBEAT_INTERVAL_MERGE_ROUTING == 100 * HEARTBEAT_INTERVAL_INSERT_ROUTING
