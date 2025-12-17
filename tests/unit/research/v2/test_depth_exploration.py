"""Unit tests for v2 Depth Exploration Service.

Tests the depth exploration logic (Phase 1.5) without requiring real API calls,
using mock LLM and embedding providers.
"""

import pytest

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.v2.depth_exploration import DepthExplorationService
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider


@pytest.fixture
def fake_llm_provider():
    """Create fake LLM provider for exploration query generation."""
    return FakeLLMProvider(
        responses={
            # Exploration query generation (structured JSON)
            "DIFFERENT ASPECTS": '{"queries": ["How does the initialization work?", "What error handling patterns are used?"]}',
            "ROOT QUERY": '{"queries": ["What is the data flow?", "How are dependencies managed?"]}',
        }
    )


@pytest.fixture
def fake_embedding_provider():
    """Create fake embedding provider for unified search."""
    return FakeEmbeddingProvider(dims=1536)


@pytest.fixture
def llm_manager(fake_llm_provider, monkeypatch):
    """Create LLM manager with fake provider."""

    def mock_create_provider(self, config):
        return fake_llm_provider

    monkeypatch.setattr(LLMManager, "_create_provider", mock_create_provider)

    utility_config = {"provider": "fake", "model": "fake-gpt"}
    synthesis_config = {"provider": "fake", "model": "fake-gpt"}
    return LLMManager(utility_config, synthesis_config)


@pytest.fixture
def embedding_manager(fake_embedding_provider, monkeypatch):
    """Create embedding manager with fake provider."""

    class MockEmbeddingManager:
        def get_provider(self):
            return fake_embedding_provider

    return MockEmbeddingManager()


@pytest.fixture
def db_services(monkeypatch):
    """Create mock database services."""

    class MockProvider:
        def get_base_directory(self):
            from pathlib import Path

            return Path("/fake/base")

    class MockDatabaseServices:
        provider = MockProvider()

    return MockDatabaseServices()


@pytest.fixture
def research_config():
    """Create research configuration for testing."""
    return ResearchConfig(
        depth_exploration_enabled=True,
        max_exploration_files=3,
        exploration_queries_per_file=2,
        min_gaps=1,
        max_gaps=5,
        target_tokens=10000,
        max_chunks_per_file_repr=5,
        max_tokens_per_file_repr=2000,
    )


@pytest.fixture
def depth_exploration_service(
    llm_manager, embedding_manager, db_services, research_config
):
    """Create depth exploration service with mocked dependencies."""
    return DepthExplorationService(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=research_config,
    )


class TestGroupChunksByFile:
    """Test chunk grouping by file path."""

    def test_groups_chunks_correctly(self, depth_exploration_service):
        """Should group chunks by file_path."""
        chunks = [
            {"chunk_id": "c1", "file_path": "file1.py", "content": "code1"},
            {"chunk_id": "c2", "file_path": "file1.py", "content": "code2"},
            {"chunk_id": "c3", "file_path": "file2.py", "content": "code3"},
        ]

        grouped = depth_exploration_service._group_chunks_by_file(chunks)

        assert len(grouped) == 2
        assert len(grouped["file1.py"]) == 2
        assert len(grouped["file2.py"]) == 1

    def test_skips_chunks_without_file_path(self, depth_exploration_service):
        """Should skip chunks without file_path."""
        chunks = [
            {"chunk_id": "c1", "file_path": "file1.py", "content": "code1"},
            {"chunk_id": "c2", "content": "code2"},  # No file_path
            {"chunk_id": "c3", "file_path": "", "content": "code3"},  # Empty path
        ]

        grouped = depth_exploration_service._group_chunks_by_file(chunks)

        assert len(grouped) == 1
        assert "file1.py" in grouped

    def test_empty_chunks_returns_empty_dict(self, depth_exploration_service):
        """Should return empty dict for empty chunks."""
        grouped = depth_exploration_service._group_chunks_by_file([])
        assert grouped == {}


class TestSelectTopFiles:
    """Test top-K file selection by score."""

    def test_selects_top_k_files(self, depth_exploration_service):
        """Should select top K files by average rerank score."""
        file_to_chunks = {
            "low.py": [{"rerank_score": 0.3}],
            "medium.py": [{"rerank_score": 0.5}],
            "high.py": [{"rerank_score": 0.9}],
        }

        top_files = depth_exploration_service._select_top_files(file_to_chunks, 2)

        assert len(top_files) == 2
        assert top_files[0] == "high.py"
        assert top_files[1] == "medium.py"

    def test_averages_scores_for_multiple_chunks(self, depth_exploration_service):
        """Should average scores when file has multiple chunks."""
        file_to_chunks = {
            "file1.py": [
                {"rerank_score": 0.8},
                {"rerank_score": 0.6},
            ],  # avg 0.7
            "file2.py": [{"rerank_score": 0.75}],  # avg 0.75
        }

        top_files = depth_exploration_service._select_top_files(file_to_chunks, 2)

        assert top_files[0] == "file2.py"  # 0.75 > 0.7
        assert top_files[1] == "file1.py"

    def test_handles_missing_rerank_score(self, depth_exploration_service):
        """Should handle chunks without rerank_score."""
        file_to_chunks = {
            "file1.py": [{"content": "code"}],  # No rerank_score
            "file2.py": [{"rerank_score": 0.5}],
        }

        top_files = depth_exploration_service._select_top_files(file_to_chunks, 2)

        assert len(top_files) == 2
        assert top_files[0] == "file2.py"  # Has score

    def test_respects_max_files_limit(self, depth_exploration_service):
        """Should not return more than max_files."""
        file_to_chunks = {
            f"file{i}.py": [{"rerank_score": 0.5}] for i in range(10)
        }

        top_files = depth_exploration_service._select_top_files(file_to_chunks, 3)

        assert len(top_files) == 3


class TestGlobalDedup:
    """Test global deduplication across exploration results."""

    def test_deduplicates_by_chunk_id(self, depth_exploration_service):
        """Should deduplicate chunks by chunk_id."""
        results = [
            [
                {"chunk_id": "c1", "content": "code1", "rerank_score": 0.8},
                {"chunk_id": "c2", "content": "code2", "rerank_score": 0.7},
            ],
            [
                {"chunk_id": "c1", "content": "code1", "rerank_score": 0.9},  # Dup
                {"chunk_id": "c3", "content": "code3", "rerank_score": 0.6},
            ],
        ]

        deduped = depth_exploration_service._global_dedup(results)

        chunk_ids = [c["chunk_id"] for c in deduped]
        assert len(chunk_ids) == 3
        assert set(chunk_ids) == {"c1", "c2", "c3"}

    def test_keeps_higher_score_on_conflict(self, depth_exploration_service):
        """Should keep chunk with higher rerank_score on conflict."""
        results = [
            [{"chunk_id": "c1", "content": "code1", "rerank_score": 0.5}],
            [{"chunk_id": "c1", "content": "code1", "rerank_score": 0.9}],
        ]

        deduped = depth_exploration_service._global_dedup(results)

        assert len(deduped) == 1
        assert deduped[0]["rerank_score"] == 0.9

    def test_handles_id_field_fallback(self, depth_exploration_service):
        """Should fall back to 'id' if 'chunk_id' not present."""
        results = [
            [{"id": "c1", "content": "code1", "rerank_score": 0.8}],
            [{"id": "c2", "content": "code2", "rerank_score": 0.7}],
        ]

        deduped = depth_exploration_service._global_dedup(results)

        assert len(deduped) == 2

    def test_skips_chunks_without_id(self, depth_exploration_service):
        """Should skip chunks without any ID field."""
        results = [
            [
                {"chunk_id": "c1", "content": "code1"},
                {"content": "code2"},  # No ID
            ],
        ]

        deduped = depth_exploration_service._global_dedup(results)

        assert len(deduped) == 1
        assert deduped[0]["chunk_id"] == "c1"


class TestMergeCoverage:
    """Test merging coverage and exploration chunks."""

    def test_merges_without_duplicates(self, depth_exploration_service):
        """Should merge coverage and exploration chunks."""
        covered = [
            {"chunk_id": "c1", "content": "code1", "rerank_score": 0.8},
            {"chunk_id": "c2", "content": "code2", "rerank_score": 0.7},
        ]
        exploration = [
            {"chunk_id": "c3", "content": "code3", "rerank_score": 0.9},
        ]

        merged = depth_exploration_service._merge_coverage(covered, exploration)

        assert len(merged) == 3
        chunk_ids = {c["chunk_id"] for c in merged}
        assert chunk_ids == {"c1", "c2", "c3"}

    def test_overwrites_with_higher_score(self, depth_exploration_service):
        """Should overwrite coverage chunk if exploration has higher score."""
        covered = [
            {"chunk_id": "c1", "content": "code1", "rerank_score": 0.5},
        ]
        exploration = [
            {"chunk_id": "c1", "content": "code1", "rerank_score": 0.9},
        ]

        merged = depth_exploration_service._merge_coverage(covered, exploration)

        assert len(merged) == 1
        assert merged[0]["rerank_score"] == 0.9

    def test_keeps_coverage_chunk_if_higher_score(self, depth_exploration_service):
        """Should keep coverage chunk if it has higher score."""
        covered = [
            {"chunk_id": "c1", "content": "code1", "rerank_score": 0.9},
        ]
        exploration = [
            {"chunk_id": "c1", "content": "code1", "rerank_score": 0.5},
        ]

        merged = depth_exploration_service._merge_coverage(covered, exploration)

        assert len(merged) == 1
        assert merged[0]["rerank_score"] == 0.9


class TestExploreCoverageDepthEmptyInput:
    """Test explore_coverage_depth with edge cases."""

    @pytest.mark.asyncio
    async def test_returns_original_on_empty_chunks(self, depth_exploration_service):
        """Should return original chunks when input is empty."""
        chunks, stats = await depth_exploration_service.explore_coverage_depth(
            root_query="test query",
            covered_chunks=[],
            phase1_threshold=0.5,
        )

        assert chunks == []
        assert stats["files_explored"] == 0
        assert stats["queries_generated"] == 0
        assert stats["chunks_added"] == 0
