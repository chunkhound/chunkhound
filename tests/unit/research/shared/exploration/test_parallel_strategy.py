"""Tests for ParallelExplorationStrategy.

Tests verify that:
1. ParallelExplorationStrategy implements the ExplorationStrategy protocol
2. Both BFS and WideCoverage strategies run concurrently
3. Results are properly merged and deduplicated
4. Unified elbow detection is applied after merging
5. File reading happens after elbow filtering
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from chunkhound.services.research.shared.exploration import (
    ExplorationStrategy,
    ParallelExplorationStrategy,
    get_unified_score,
)


class TestParallelExplorationStrategyProtocol:
    """Test that ParallelExplorationStrategy conforms to ExplorationStrategy."""

    def test_implements_protocol(self) -> None:
        """ParallelExplorationStrategy should implement ExplorationStrategy."""
        bfs_strategy = MagicMock()
        wide_strategy = MagicMock()
        file_reader = MagicMock()
        llm_manager = MagicMock()

        strategy = ParallelExplorationStrategy(
            bfs_strategy=bfs_strategy,
            wide_strategy=wide_strategy,
            file_reader=file_reader,
            llm_manager=llm_manager,
        )

        assert isinstance(strategy, ExplorationStrategy)
        assert hasattr(strategy, "name")
        assert hasattr(strategy, "explore")
        assert strategy.name == "parallel"


class TestParallelExplorationStrategy:
    """Tests for ParallelExplorationStrategy."""

    @pytest.fixture
    def mock_strategy(self) -> ParallelExplorationStrategy:
        """Create a strategy with mocked dependencies."""
        bfs_strategy = MagicMock()
        wide_strategy = MagicMock()
        file_reader = MagicMock()
        llm_manager = MagicMock()

        strategy = ParallelExplorationStrategy(
            bfs_strategy=bfs_strategy,
            wide_strategy=wide_strategy,
            file_reader=file_reader,
            llm_manager=llm_manager,
        )
        return strategy

    @pytest.mark.asyncio
    async def test_explore_returns_tuple(
        self, mock_strategy: ParallelExplorationStrategy
    ) -> None:
        """explore() should return (chunks, stats, files) tuple."""
        # Mock BFS explore_raw
        mock_strategy._bfs.explore_raw = AsyncMock(
            return_value=(
                [{"chunk_id": "1", "content": "bfs", "rerank_score": 0.9}],
                {"nodes_explored": 1, "chunks_total": 1},
            )
        )
        # Mock Wide explore_raw
        mock_strategy._wide.explore_raw = AsyncMock(
            return_value=(
                [{"chunk_id": "2", "content": "wide", "rerank_score": 0.8}],
                {"chunks_before": 1, "chunks_after": 1},
            )
        )
        # Mock file reader
        mock_strategy._file_reader.read_files_with_budget = AsyncMock(
            return_value={"test.py": "content"}
        )

        result = await mock_strategy.explore(
            root_query="test query",
            initial_chunks=[{"chunk_id": "0", "content": "initial"}],
            phase1_threshold=0.5,
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        chunks, stats, files = result
        assert isinstance(chunks, list)
        assert isinstance(stats, dict)
        assert isinstance(files, dict)

    @pytest.mark.asyncio
    async def test_both_strategies_called_concurrently(
        self, mock_strategy: ParallelExplorationStrategy
    ) -> None:
        """Both BFS and Wide strategies should be called."""
        mock_strategy._bfs.explore_raw = AsyncMock(
            return_value=([], {"nodes_explored": 0})
        )
        mock_strategy._wide.explore_raw = AsyncMock(
            return_value=([], {"chunks_before": 0, "chunks_after": 0})
        )
        mock_strategy._file_reader.read_files_with_budget = AsyncMock(
            return_value={}
        )

        initial_chunks = [{"chunk_id": "0"}]
        await mock_strategy.explore(
            root_query="test",
            initial_chunks=initial_chunks,
            phase1_threshold=0.5,
        )

        mock_strategy._bfs.explore_raw.assert_called_once()
        mock_strategy._wide.explore_raw.assert_called_once()

    @pytest.mark.asyncio
    async def test_chunks_deduplicated_by_id(
        self, mock_strategy: ParallelExplorationStrategy
    ) -> None:
        """Duplicate chunk_ids should be deduplicated, keeping highest score."""
        # BFS returns chunk 1 with score 0.7
        mock_strategy._bfs.explore_raw = AsyncMock(
            return_value=(
                [{"chunk_id": "1", "content": "bfs_version", "rerank_score": 0.7}],
                {"nodes_explored": 1},
            )
        )
        # Wide returns same chunk 1 with higher score 0.9
        mock_strategy._wide.explore_raw = AsyncMock(
            return_value=(
                [{"chunk_id": "1", "content": "wide_version", "rerank_score": 0.9}],
                {"chunks_after": 1},
            )
        )
        mock_strategy._file_reader.read_files_with_budget = AsyncMock(
            return_value={}
        )

        chunks, stats, _ = await mock_strategy.explore(
            root_query="test",
            initial_chunks=[{"chunk_id": "0"}],
            phase1_threshold=0.5,
        )

        # Should only have 1 chunk (deduplicated)
        assert len(chunks) == 1
        # Should keep the higher-scored version (wide)
        assert chunks[0]["content"] == "wide_version"
        assert chunks[0]["rerank_score"] == 0.9

    @pytest.mark.asyncio
    async def test_stats_from_both_strategies_preserved(
        self, mock_strategy: ParallelExplorationStrategy
    ) -> None:
        """Stats from both strategies should be preserved in output."""
        mock_strategy._bfs.explore_raw = AsyncMock(
            return_value=(
                [{"chunk_id": "1", "rerank_score": 0.9}],
                {"nodes_explored": 5, "depth_reached": 1},
            )
        )
        mock_strategy._wide.explore_raw = AsyncMock(
            return_value=(
                [{"chunk_id": "2", "rerank_score": 0.8}],
                {"chunks_before": 1, "chunks_after": 2, "gaps_filled": 1},
            )
        )
        mock_strategy._file_reader.read_files_with_budget = AsyncMock(
            return_value={}
        )

        _, stats, _ = await mock_strategy.explore(
            root_query="test",
            initial_chunks=[{"chunk_id": "0"}],
            phase1_threshold=0.5,
        )

        assert "bfs" in stats
        assert stats["bfs"]["nodes_explored"] == 5
        assert "wide" in stats
        assert stats["wide"]["gaps_filled"] == 1
        assert "merged_chunks" in stats
        assert "elbow_filter" in stats

    @pytest.mark.asyncio
    async def test_file_reading_after_elbow_filter(
        self, mock_strategy: ParallelExplorationStrategy
    ) -> None:
        """File reading should happen after elbow filtering."""
        mock_strategy._bfs.explore_raw = AsyncMock(
            return_value=(
                [{"chunk_id": "1", "file_path": "a.py", "rerank_score": 0.9}],
                {},
            )
        )
        mock_strategy._wide.explore_raw = AsyncMock(
            return_value=(
                [{"chunk_id": "2", "file_path": "b.py", "rerank_score": 0.8}],
                {},
            )
        )
        mock_strategy._file_reader.read_files_with_budget = AsyncMock(
            return_value={"a.py": "content_a", "b.py": "content_b"}
        )

        chunks, _, files = await mock_strategy.explore(
            root_query="test",
            initial_chunks=[{"chunk_id": "0"}],
            phase1_threshold=0.5,
        )

        # File reader should be called with filtered chunks
        mock_strategy._file_reader.read_files_with_budget.assert_called_once()
        call_args = mock_strategy._file_reader.read_files_with_budget.call_args
        # Should be called with max_tokens=None (unlimited after elbow)
        assert call_args.kwargs["max_tokens"] is None

    @pytest.mark.asyncio
    async def test_empty_initial_chunks_returns_early(
        self, mock_strategy: ParallelExplorationStrategy
    ) -> None:
        """Empty initial chunks should return early without calling strategies."""
        chunks, stats, files = await mock_strategy.explore(
            root_query="test",
            initial_chunks=[],
            phase1_threshold=0.5,
        )

        assert chunks == []
        assert files == {}
        assert stats["merged_chunks"] == 0
        # Strategies should not be called
        mock_strategy._bfs.explore_raw.assert_not_called()
        mock_strategy._wide.explore_raw.assert_not_called()

    @pytest.mark.asyncio
    async def test_strategy_exception_handled_gracefully(
        self, mock_strategy: ParallelExplorationStrategy
    ) -> None:
        """If one strategy fails, the other's results should still be used."""
        # BFS fails
        mock_strategy._bfs.explore_raw = AsyncMock(
            side_effect=RuntimeError("BFS failed")
        )
        # Wide succeeds
        mock_strategy._wide.explore_raw = AsyncMock(
            return_value=(
                [{"chunk_id": "1", "rerank_score": 0.9}],
                {"chunks_after": 1},
            )
        )
        mock_strategy._file_reader.read_files_with_budget = AsyncMock(
            return_value={}
        )

        chunks, stats, _ = await mock_strategy.explore(
            root_query="test",
            initial_chunks=[{"chunk_id": "0"}],
            phase1_threshold=0.5,
        )

        # Should still get Wide's chunks
        assert len(chunks) == 1
        # BFS stats should have error
        assert "error" in stats["bfs"]


class TestGetUnifiedScore:
    """Tests for get_unified_score helper."""

    def test_prefers_rerank_score(self) -> None:
        """Should prefer rerank_score over score."""
        chunk = {"rerank_score": 0.9, "score": 0.5}
        assert get_unified_score(chunk) == 0.9

    def test_falls_back_to_score(self) -> None:
        """Should fall back to score if rerank_score is missing."""
        chunk = {"score": 0.5}
        assert get_unified_score(chunk) == 0.5

    def test_returns_zero_if_no_scores(self) -> None:
        """Should return 0.0 if no score fields present."""
        chunk = {"content": "test"}
        assert get_unified_score(chunk) == 0.0

    def test_handles_none_rerank_score(self) -> None:
        """Should fall back to score if rerank_score is None."""
        chunk = {"rerank_score": None, "score": 0.5}
        assert get_unified_score(chunk) == 0.5

    def test_converts_to_float(self) -> None:
        """Should convert score to float."""
        chunk = {"rerank_score": "0.9"}  # String value
        result = get_unified_score(chunk)
        assert isinstance(result, float)
        assert result == 0.9
