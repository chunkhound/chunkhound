"""Tests for exploration strategies.

Tests verify that both WideCoverageStrategy and BFSExplorationStrategy:
1. Implement the ExplorationStrategy protocol
2. Return compatible result formats
3. Can be swapped in research pipelines
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from chunkhound.services.research.shared.exploration import (
    ExplorationStrategy,
    WideCoverageStrategy,
    BFSExplorationStrategy,
)


class TestExplorationStrategyProtocol:
    """Test that implementations conform to the ExplorationStrategy protocol."""

    def test_wide_coverage_implements_protocol(self) -> None:
        """WideCoverageStrategy should implement ExplorationStrategy."""
        # Create mock dependencies
        llm_manager = MagicMock()
        embedding_manager = MagicMock()
        db_services = MagicMock()
        config = MagicMock()
        config.depth_exploration_enabled = True

        strategy = WideCoverageStrategy(
            llm_manager=llm_manager,
            embedding_manager=embedding_manager,
            db_services=db_services,
            config=config,
        )

        assert isinstance(strategy, ExplorationStrategy)
        assert hasattr(strategy, "name")
        assert hasattr(strategy, "explore")
        assert strategy.name == "wide_coverage"

    def test_bfs_exploration_implements_protocol(self) -> None:
        """BFSExplorationStrategy should implement ExplorationStrategy."""
        llm_manager = MagicMock()
        embedding_manager = MagicMock()
        db_services = MagicMock()

        strategy = BFSExplorationStrategy(
            llm_manager=llm_manager,
            embedding_manager=embedding_manager,
            db_services=db_services,
        )

        assert isinstance(strategy, ExplorationStrategy)
        assert hasattr(strategy, "name")
        assert hasattr(strategy, "explore")
        assert strategy.name == "bfs"


class TestWideCoverageStrategy:
    """Tests for WideCoverageStrategy."""

    @pytest.fixture
    def mock_strategy(self) -> WideCoverageStrategy:
        """Create a strategy with mocked dependencies."""
        llm_manager = MagicMock()
        embedding_manager = MagicMock()
        db_services = MagicMock()
        config = MagicMock()
        config.depth_exploration_enabled = True

        strategy = WideCoverageStrategy(
            llm_manager=llm_manager,
            embedding_manager=embedding_manager,
            db_services=db_services,
            config=config,
        )
        return strategy

    @pytest.mark.asyncio
    async def test_explore_returns_tuple(self, mock_strategy: WideCoverageStrategy) -> None:
        """explore() should return (chunks, stats, files) tuple."""
        # Mock internal services
        mock_strategy._depth_exploration.explore_coverage_depth = AsyncMock(
            return_value=(
                [{"chunk_id": "1", "content": "test"}],
                {"chunks_added": 1},
            )
        )
        mock_strategy._gap_detection.detect_and_fill_gaps = AsyncMock(
            return_value=(
                [{"chunk_id": "1", "content": "test"}, {"chunk_id": "2", "content": "more"}],
                {"gaps_found": 1, "gaps_filled": 1},
            )
        )
        # Mock file reader to return empty dict (no files found)
        mock_strategy._file_reader.read_files_with_budget = AsyncMock(
            return_value={}
        )

        initial_chunks = [{"chunk_id": "0", "content": "initial"}]
        result = await mock_strategy.explore(
            root_query="test query",
            initial_chunks=initial_chunks,
            phase1_threshold=0.5,
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        chunks, stats, files = result
        assert isinstance(chunks, list)
        assert isinstance(stats, dict)
        assert isinstance(files, dict)

    @pytest.mark.asyncio
    async def test_explore_empty_chunks(self, mock_strategy: WideCoverageStrategy) -> None:
        """explore() with empty initial chunks should return empty results."""
        mock_strategy._depth_exploration.explore_coverage_depth = AsyncMock(
            return_value=([], {"chunks_added": 0})
        )
        mock_strategy._gap_detection.detect_and_fill_gaps = AsyncMock(
            return_value=([], {"gaps_found": 0})
        )
        # Mock file reader to return empty dict (no files found)
        mock_strategy._file_reader.read_files_with_budget = AsyncMock(
            return_value={}
        )

        result = await mock_strategy.explore(
            root_query="test",
            initial_chunks=[],
            phase1_threshold=0.5,
        )

        chunks, stats, files = result
        assert chunks == []
        assert files == {}


class TestBFSExplorationStrategy:
    """Tests for BFSExplorationStrategy."""

    @pytest.fixture
    def mock_strategy(self) -> BFSExplorationStrategy:
        """Create a strategy with mocked dependencies."""
        llm_manager = MagicMock()
        llm_manager.get_utility_provider.return_value.estimate_tokens.return_value = 100

        embedding_manager = MagicMock()
        # Mock rerank provider with proper return values
        embedding_manager.get_provider.return_value.get_max_rerank_batch_size.return_value = 100
        embedding_manager.get_provider.return_value.rerank = AsyncMock(return_value=[])

        db_services = MagicMock()
        db_services.provider.get_base_directory.return_value = MagicMock()

        strategy = BFSExplorationStrategy(
            llm_manager=llm_manager,
            embedding_manager=embedding_manager,
            db_services=db_services,
        )
        return strategy

    @pytest.mark.asyncio
    async def test_explore_returns_tuple(self, mock_strategy: BFSExplorationStrategy) -> None:
        """explore() should return (chunks, stats, files) tuple."""
        # Mock unified search
        mock_strategy._unified_search.unified_search = AsyncMock(
            return_value=[{"chunk_id": "new1", "content": "found"}]
        )

        # Mock question generator
        mock_strategy._question_generator.generate_follow_up_questions = AsyncMock(
            return_value=[]
        )

        initial_chunks = [{"chunk_id": "0", "content": "initial", "file_path": "test.py"}]
        result = await mock_strategy.explore(
            root_query="test query",
            initial_chunks=initial_chunks,
            phase1_threshold=0.5,
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        chunks, stats, files = result
        assert isinstance(chunks, list)
        assert isinstance(stats, dict)
        assert isinstance(files, dict)

    @pytest.mark.asyncio
    async def test_explore_empty_chunks(self, mock_strategy: BFSExplorationStrategy) -> None:
        """explore() with empty initial chunks should return early."""
        result = await mock_strategy.explore(
            root_query="test",
            initial_chunks=[],
            phase1_threshold=0.5,
        )

        chunks, stats, files = result
        assert chunks == []
        assert stats["nodes_explored"] == 0
        assert files == {}

    def test_aggregate_chunks_deduplicates(self, mock_strategy: BFSExplorationStrategy) -> None:
        """_aggregate_chunks should deduplicate by chunk_id."""
        from chunkhound.services.research.shared.exploration.bfs_exploration_strategy import (
            BFSExplorationNode,
        )

        node1 = BFSExplorationNode(query="q1", chunks=[{"chunk_id": "a"}, {"chunk_id": "b"}])
        node2 = BFSExplorationNode(query="q2", chunks=[{"chunk_id": "b"}, {"chunk_id": "c"}])

        result = mock_strategy._aggregate_chunks([node1, node2])

        chunk_ids = [c["chunk_id"] for c in result]
        assert len(chunk_ids) == 3
        assert set(chunk_ids) == {"a", "b", "c"}


class TestBFSExplorationStrategyInternals:
    """Tests for BFS internal helper methods."""

    @pytest.fixture
    def mock_strategy(self) -> BFSExplorationStrategy:
        """Create strategy with mocked dependencies."""
        llm_manager = MagicMock()
        embedding_manager = MagicMock()
        db_services = MagicMock()

        strategy = BFSExplorationStrategy(
            llm_manager=llm_manager,
            embedding_manager=embedding_manager,
            db_services=db_services,
        )
        return strategy

    @pytest.mark.asyncio
    async def test_traverse_bfs_tree_returns_all_nodes(
        self, mock_strategy: BFSExplorationStrategy
    ) -> None:
        """_traverse_bfs_tree should return all visited nodes including root."""
        from chunkhound.services.research.shared.exploration.bfs_exploration_strategy import (
            BFSExplorationNode,
        )
        from chunkhound.services.research.shared.models import ResearchContext

        # Mock _process_node to return empty children (no traversal)
        mock_strategy._process_node = AsyncMock(return_value=[])

        root = BFSExplorationNode(
            query="test",
            depth=0,
            node_id=1,
            chunks=[{"chunk_id": "0"}],
        )
        context = ResearchContext(root_query="test")
        global_data: dict[str, Any] = {
            "files_fully_read": set(),
            "chunk_ranges": {},
            "chunks": [],
        }

        nodes = await mock_strategy._traverse_bfs_tree(
            root=root,
            context=context,
            global_explored_data=global_data,
            path_filter=None,
            constants_context="",
            log_prefix="Test",
        )

        assert len(nodes) == 1
        assert nodes[0] == root

    @pytest.mark.asyncio
    async def test_traverse_bfs_tree_handles_node_errors(
        self, mock_strategy: BFSExplorationStrategy
    ) -> None:
        """_traverse_bfs_tree should gracefully handle node processing errors."""
        from chunkhound.services.research.shared.exploration.bfs_exploration_strategy import (
            BFSExplorationNode,
        )
        from chunkhound.services.research.shared.models import ResearchContext

        # Mock _process_node to raise error
        mock_strategy._process_node = AsyncMock(side_effect=Exception("Test error"))

        root = BFSExplorationNode(query="test", depth=0, node_id=1)
        context = ResearchContext(root_query="test")
        global_data: dict[str, Any] = {
            "files_fully_read": set(),
            "chunk_ranges": {},
            "chunks": [],
        }

        nodes = await mock_strategy._traverse_bfs_tree(
            root=root,
            context=context,
            global_explored_data=global_data,
            path_filter=None,
            constants_context="",
            log_prefix="Test",
        )

        # Should still return root despite error
        assert len(nodes) == 1
        assert nodes[0] == root

    @pytest.mark.asyncio
    async def test_rerank_files_in_batches_single_batch(
        self, mock_strategy: BFSExplorationStrategy
    ) -> None:
        """_rerank_files_in_batches should handle single batch case."""
        from chunkhound.interfaces.embedding_provider import RerankResult

        # Mock provider to return results
        mock_provider = mock_strategy._embedding_manager.get_provider()
        mock_provider.get_max_rerank_batch_size.return_value = 100
        mock_provider.rerank = AsyncMock(
            return_value=[
                RerankResult(index=0, score=0.9),
                RerankResult(index=1, score=0.7),
            ]
        )

        results = await mock_strategy._rerank_files_in_batches(
            root_query="test",
            file_paths=["file1.py", "file2.py"],
            file_documents=["doc1", "doc2"],
        )

        assert len(results) == 2
        assert results[0] == ("file1.py", 0.9)
        assert results[1] == ("file2.py", 0.7)
        # Should call rerank once
        assert mock_provider.rerank.call_count == 1

    @pytest.mark.asyncio
    async def test_rerank_files_in_batches_multiple_batches(
        self, mock_strategy: BFSExplorationStrategy
    ) -> None:
        """_rerank_files_in_batches should split into multiple batches."""
        from chunkhound.interfaces.embedding_provider import RerankResult

        # Mock provider with small batch size
        mock_provider = mock_strategy._embedding_manager.get_provider()
        mock_provider.get_max_rerank_batch_size.return_value = 2
        mock_provider.name = "test_provider"
        mock_provider.rerank = AsyncMock(
            side_effect=[
                # Batch 1: indices 0-1
                [RerankResult(index=0, score=0.9), RerankResult(index=1, score=0.8)],
                # Batch 2: indices 0-1 (relative to batch)
                [RerankResult(index=0, score=0.7), RerankResult(index=1, score=0.6)],
            ]
        )

        results = await mock_strategy._rerank_files_in_batches(
            root_query="test",
            file_paths=["f1.py", "f2.py", "f3.py", "f4.py"],
            file_documents=["d1", "d2", "d3", "d4"],
        )

        # Should adjust indices correctly
        assert len(results) == 4
        assert results[0] == ("f1.py", 0.9)  # Batch 1, idx 0 → global 0
        assert results[1] == ("f2.py", 0.8)  # Batch 1, idx 1 → global 1
        assert results[2] == ("f3.py", 0.7)  # Batch 2, idx 0 → global 2
        assert results[3] == ("f4.py", 0.6)  # Batch 2, idx 1 → global 3

        # Should call rerank twice (2 batches)
        assert mock_provider.rerank.call_count == 2

    @pytest.mark.asyncio
    async def test_rerank_files_in_batches_handles_failures(
        self, mock_strategy: BFSExplorationStrategy
    ) -> None:
        """_rerank_files_in_batches should handle batch failures gracefully."""
        from chunkhound.interfaces.embedding_provider import RerankResult

        # Mock provider with batch failure
        mock_provider = mock_strategy._embedding_manager.get_provider()
        mock_provider.get_max_rerank_batch_size.return_value = 2
        mock_provider.name = "test_provider"
        mock_provider.rerank = AsyncMock(
            side_effect=[
                # Batch 1: success
                [RerankResult(index=0, score=0.9), RerankResult(index=1, score=0.8)],
                # Batch 2: failure
                Exception("Rerank failed"),
            ]
        )

        results = await mock_strategy._rerank_files_in_batches(
            root_query="test",
            file_paths=["f1.py", "f2.py", "f3.py", "f4.py"],
            file_documents=["d1", "d2", "d3", "d4"],
        )

        # Should return partial results (batch 1 only)
        assert len(results) == 2
        assert results[0] == ("f1.py", 0.9)
        assert results[1] == ("f2.py", 0.8)

    @pytest.mark.asyncio
    async def test_rerank_files_in_batches_all_batches_fail(
        self, mock_strategy: BFSExplorationStrategy
    ) -> None:
        """_rerank_files_in_batches should raise if all batches fail."""
        # Mock provider to always fail
        mock_provider = mock_strategy._embedding_manager.get_provider()
        mock_provider.get_max_rerank_batch_size.return_value = 100
        mock_provider.rerank = AsyncMock(side_effect=Exception("Rerank failed"))

        with pytest.raises(Exception, match="Rerank failed"):
            await mock_strategy._rerank_files_in_batches(
                root_query="test",
                file_paths=["f1.py"],
                file_documents=["d1"],
            )

    @pytest.mark.asyncio
    async def test_rerank_files_in_batches_filters_invalid_indices(
        self, mock_strategy: BFSExplorationStrategy
    ) -> None:
        """_rerank_files_in_batches should filter out-of-bounds indices."""
        from chunkhound.interfaces.embedding_provider import RerankResult

        # Mock provider to return invalid index
        mock_provider = mock_strategy._embedding_manager.get_provider()
        mock_provider.get_max_rerank_batch_size.return_value = 100
        mock_provider.rerank = AsyncMock(
            return_value=[
                RerankResult(index=0, score=0.9),
                RerankResult(index=99, score=0.8),  # Invalid: > len(documents)
                RerankResult(index=1, score=0.7),
            ]
        )

        results = await mock_strategy._rerank_files_in_batches(
            root_query="test",
            file_paths=["f1.py", "f2.py"],
            file_documents=["d1", "d2"],
        )

        # Should filter out index 99
        assert len(results) == 2
        assert results[0] == ("f1.py", 0.9)
        assert results[1] == ("f2.py", 0.7)


class TestStrategySwapping:
    """Test that strategies can be swapped in research services."""

    def test_pluggable_service_accepts_strategy(self) -> None:
        """PluggableResearchService should accept exploration strategy."""
        from chunkhound.services.research.v1.pluggable_research_service import (
            PluggableResearchService,
        )

        # Create mock dependencies
        db_services = MagicMock()
        embedding_manager = MagicMock()
        llm_manager = MagicMock()

        # Create a custom strategy (e.g., WideCoverageStrategy used in pluggable service)
        custom_strategy = MagicMock(spec=ExplorationStrategy)
        custom_strategy.name = "wide_coverage"

        # Should accept the strategy without error (now required, not optional)
        service = PluggableResearchService(
            database_services=db_services,
            embedding_manager=embedding_manager,
            llm_manager=llm_manager,
            exploration_strategy=custom_strategy,
        )

        assert service._exploration_strategy == custom_strategy
