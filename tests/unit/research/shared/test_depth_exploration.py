"""Unit tests for Depth Exploration Service.

Tests the depth exploration logic (Phase 1.5) without requiring real API calls,
using mock LLM and embedding providers.

Tests focus on synchronous helper methods:
- _group_chunks_by_file(): Group chunks by file path
- _select_top_files(): Score calculation and file selection
- _global_dedup(): Cross-query deduplication (via deduplicate_chunks)
- _merge_coverage(): Merging coverage and exploration chunks
- _generate_exploration_queries(): LLM response mocking
"""

import pytest

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.services.research.shared.depth_exploration import DepthExplorationService
from tests.fixtures.fake_providers import FakeLLMProvider


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def fake_llm_provider():
    """Create fake LLM provider for exploration query generation."""
    return FakeLLMProvider(
        responses={
            # Exploration query generation (structured JSON)
            "different aspects": '{"queries": ["How does initialization work?", "What error handling patterns are used?"]}',
            "root query": '{"queries": ["What is the data flow?", "How are dependencies managed?"]}',
        }
    )


@pytest.fixture
def mock_llm_manager(fake_llm_provider):
    """Create LLM manager that returns fake provider."""

    class MockLLMManager:
        def __init__(self, provider):
            self._provider = provider

        def get_utility_provider(self):
            return self._provider

        def get_synthesis_provider(self):
            return self._provider

    return MockLLMManager(fake_llm_provider)


@pytest.fixture
def mock_embedding_manager():
    """Create mock embedding manager."""

    class MockEmbeddingManager:
        pass

    return MockEmbeddingManager()


@pytest.fixture
def mock_db_services():
    """Create mock database services."""
    from pathlib import Path

    class MockProvider:
        def get_base_directory(self):
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
    )


@pytest.fixture
def depth_service(mock_llm_manager, mock_embedding_manager, mock_db_services, research_config):
    """Create depth exploration service with mocked dependencies."""
    return DepthExplorationService(
        llm_manager=mock_llm_manager,
        embedding_manager=mock_embedding_manager,
        db_services=mock_db_services,
        config=research_config,
    )


# -----------------------------------------------------------------------------
# Tests: _group_chunks_by_file
# -----------------------------------------------------------------------------


class TestGroupChunksByFile:
    """Test chunk grouping by file path."""

    def test_groups_chunks_correctly(self, depth_service):
        """Should group chunks by file_path."""
        chunks = [
            {"chunk_id": "c1", "file_path": "file1.py", "content": "code1"},
            {"chunk_id": "c2", "file_path": "file1.py", "content": "code2"},
            {"chunk_id": "c3", "file_path": "file2.py", "content": "code3"},
        ]

        grouped = depth_service._group_chunks_by_file(chunks)

        assert len(grouped) == 2
        assert len(grouped["file1.py"]) == 2
        assert len(grouped["file2.py"]) == 1

    def test_skips_chunks_without_file_path(self, depth_service):
        """Should skip chunks without file_path."""
        chunks = [
            {"chunk_id": "c1", "file_path": "file1.py", "content": "code1"},
            {"chunk_id": "c2", "content": "code2"},  # No file_path
            {"chunk_id": "c3", "file_path": "", "content": "code3"},  # Empty path
        ]

        grouped = depth_service._group_chunks_by_file(chunks)

        assert len(grouped) == 1
        assert "file1.py" in grouped

    def test_empty_chunks_returns_empty_dict(self, depth_service):
        """Should return empty dict for empty chunks."""
        grouped = depth_service._group_chunks_by_file([])
        assert grouped == {}

    def test_preserves_chunk_data(self, depth_service):
        """Should preserve all chunk data in grouped output."""
        chunks = [
            {
                "chunk_id": "c1",
                "file_path": "file1.py",
                "content": "def foo(): pass",
                "start_line": 1,
                "end_line": 2,
                "rerank_score": 0.9,
            }
        ]

        grouped = depth_service._group_chunks_by_file(chunks)

        chunk = grouped["file1.py"][0]
        assert chunk["chunk_id"] == "c1"
        assert chunk["start_line"] == 1
        assert chunk["rerank_score"] == 0.9


# -----------------------------------------------------------------------------
# Tests: _select_top_files
# -----------------------------------------------------------------------------


class TestSelectTopFiles:
    """Test top-K file selection by score."""

    def test_selects_top_k_files(self, depth_service):
        """Should select top K files by average rerank score."""
        file_to_chunks = {
            "low.py": [{"rerank_score": 0.3}],
            "medium.py": [{"rerank_score": 0.5}],
            "high.py": [{"rerank_score": 0.9}],
        }

        top_files = depth_service._select_top_files(file_to_chunks, 2)

        assert len(top_files) == 2
        assert top_files[0] == "high.py"
        assert top_files[1] == "medium.py"

    def test_averages_scores_for_multiple_chunks(self, depth_service):
        """Should average scores when file has multiple chunks."""
        file_to_chunks = {
            "file1.py": [
                {"rerank_score": 0.8},
                {"rerank_score": 0.6},
            ],  # avg 0.7
            "file2.py": [{"rerank_score": 0.75}],  # avg 0.75
        }

        top_files = depth_service._select_top_files(file_to_chunks, 2)

        assert top_files[0] == "file2.py"  # 0.75 > 0.7
        assert top_files[1] == "file1.py"

    def test_handles_missing_rerank_score(self, depth_service):
        """Should handle chunks without rerank_score (defaults to 0.0)."""
        file_to_chunks = {
            "file1.py": [{"content": "code"}],  # No rerank_score
            "file2.py": [{"rerank_score": 0.5}],
        }

        top_files = depth_service._select_top_files(file_to_chunks, 2)

        assert len(top_files) == 2
        assert top_files[0] == "file2.py"  # Has score

    def test_respects_max_files_limit(self, depth_service):
        """Should not return more than max_files."""
        file_to_chunks = {f"file{i}.py": [{"rerank_score": 0.5}] for i in range(10)}

        top_files = depth_service._select_top_files(file_to_chunks, 3)

        assert len(top_files) == 3

    def test_empty_input_returns_empty_list(self, depth_service):
        """Should return empty list for empty input."""
        top_files = depth_service._select_top_files({}, 5)
        assert top_files == []

    def test_zero_max_files_returns_empty(self, depth_service):
        """Should return empty list when max_files is 0."""
        file_to_chunks = {"file.py": [{"rerank_score": 0.9}]}

        top_files = depth_service._select_top_files(file_to_chunks, 0)

        assert top_files == []

    def test_fewer_files_than_max_returns_all(self, depth_service):
        """Should return all files when fewer than max."""
        file_to_chunks = {
            "file1.py": [{"rerank_score": 0.9}],
            "file2.py": [{"rerank_score": 0.8}],
        }

        top_files = depth_service._select_top_files(file_to_chunks, 10)

        assert len(top_files) == 2

    def test_sorts_by_score_descending(self, depth_service):
        """Should sort files by score in descending order."""
        file_to_chunks = {
            "a.py": [{"rerank_score": 0.3}],
            "b.py": [{"rerank_score": 0.9}],
            "c.py": [{"rerank_score": 0.6}],
            "d.py": [{"rerank_score": 0.1}],
        }

        top_files = depth_service._select_top_files(file_to_chunks, 4)

        assert top_files == ["b.py", "c.py", "a.py", "d.py"]

    def test_handles_mixed_scores(self, depth_service):
        """Should handle files with varying chunk counts and scores."""
        file_to_chunks = {
            "single.py": [{"rerank_score": 0.9}],  # avg 0.9
            "multiple.py": [
                {"rerank_score": 0.7},
                {"rerank_score": 0.8},
                {"rerank_score": 0.9},
            ],  # avg 0.8
            "none.py": [{}, {}, {}],  # avg 0.0
        }

        top_files = depth_service._select_top_files(file_to_chunks, 3)

        assert top_files[0] == "single.py"  # 0.9
        assert top_files[1] == "multiple.py"  # 0.8
        assert top_files[2] == "none.py"  # 0.0


# -----------------------------------------------------------------------------
# Tests: _global_dedup
# -----------------------------------------------------------------------------


class TestGlobalDedup:
    """Test global deduplication across exploration results."""

    def test_deduplicates_by_chunk_id(self, depth_service):
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

        deduped = depth_service._global_dedup(results)

        chunk_ids = [c["chunk_id"] for c in deduped]
        assert len(chunk_ids) == 3
        assert set(chunk_ids) == {"c1", "c2", "c3"}

    def test_keeps_higher_score_on_conflict(self, depth_service):
        """Should keep chunk with higher rerank_score on conflict."""
        results = [
            [{"chunk_id": "c1", "content": "code1", "rerank_score": 0.5}],
            [{"chunk_id": "c1", "content": "code1", "rerank_score": 0.9}],
        ]

        deduped = depth_service._global_dedup(results)

        assert len(deduped) == 1
        assert deduped[0]["rerank_score"] == 0.9

    def test_handles_id_field_fallback(self, depth_service):
        """Should fall back to 'id' if 'chunk_id' not present."""
        results = [
            [{"id": "c1", "content": "code1", "rerank_score": 0.8}],
            [{"id": "c2", "content": "code2", "rerank_score": 0.7}],
        ]

        deduped = depth_service._global_dedup(results)

        assert len(deduped) == 2

    def test_skips_chunks_without_id(self, depth_service):
        """Should skip chunks without any ID field."""
        results = [
            [
                {"chunk_id": "c1", "content": "code1"},
                {"content": "code2"},  # No ID
            ],
        ]

        deduped = depth_service._global_dedup(results)

        assert len(deduped) == 1
        assert deduped[0]["chunk_id"] == "c1"

    def test_empty_results_return_empty(self, depth_service):
        """Empty input should return empty output."""
        deduped = depth_service._global_dedup([])
        assert deduped == []

    def test_handles_empty_sublists(self, depth_service):
        """Empty result sublists should be handled."""
        results = [
            [],
            [{"chunk_id": "c1", "rerank_score": 0.8}],
            [],
        ]

        deduped = depth_service._global_dedup(results)

        assert len(deduped) == 1


# -----------------------------------------------------------------------------
# Tests: _merge_coverage
# -----------------------------------------------------------------------------


class TestMergeCoverage:
    """Test merging coverage and exploration chunks."""

    def test_merges_without_duplicates(self, depth_service):
        """Should merge coverage and exploration chunks."""
        covered = [
            {"chunk_id": "c1", "content": "code1", "rerank_score": 0.8},
            {"chunk_id": "c2", "content": "code2", "rerank_score": 0.7},
        ]
        exploration = [
            {"chunk_id": "c3", "content": "code3", "rerank_score": 0.9},
        ]

        merged = depth_service._merge_coverage(covered, exploration)

        assert len(merged) == 3
        chunk_ids = {c["chunk_id"] for c in merged}
        assert chunk_ids == {"c1", "c2", "c3"}

    def test_overwrites_with_higher_score(self, depth_service):
        """Should overwrite coverage chunk if exploration has higher score."""
        covered = [
            {"chunk_id": "c1", "content": "code1", "rerank_score": 0.5},
        ]
        exploration = [
            {"chunk_id": "c1", "content": "code1", "rerank_score": 0.9},
        ]

        merged = depth_service._merge_coverage(covered, exploration)

        assert len(merged) == 1
        assert merged[0]["rerank_score"] == 0.9

    def test_keeps_coverage_chunk_if_higher_score(self, depth_service):
        """Should keep coverage chunk if it has higher score."""
        covered = [
            {"chunk_id": "c1", "content": "code1", "rerank_score": 0.9},
        ]
        exploration = [
            {"chunk_id": "c1", "content": "code1", "rerank_score": 0.5},
        ]

        merged = depth_service._merge_coverage(covered, exploration)

        assert len(merged) == 1
        assert merged[0]["rerank_score"] == 0.9

    def test_empty_exploration_returns_coverage(self, depth_service):
        """Should return coverage when exploration is empty."""
        covered = [{"chunk_id": "c1", "rerank_score": 0.8}]

        merged = depth_service._merge_coverage(covered, [])

        assert len(merged) == 1
        assert merged[0]["chunk_id"] == "c1"

    def test_empty_coverage_returns_exploration(self, depth_service):
        """Should return exploration when coverage is empty."""
        exploration = [{"chunk_id": "c1", "rerank_score": 0.8}]

        merged = depth_service._merge_coverage([], exploration)

        assert len(merged) == 1
        assert merged[0]["chunk_id"] == "c1"


# -----------------------------------------------------------------------------
# Tests: _generate_exploration_queries (async)
# -----------------------------------------------------------------------------


class TestGenerateExplorationQueries:
    """Test LLM-based exploration query generation."""

    @pytest.mark.asyncio
    async def test_returns_queries_from_llm(self, depth_service):
        """Should return queries from LLM structured response."""
        file_chunks = [
            {"chunk_id": "c1", "content": "def main(): pass", "file_path": "test.py"}
        ]

        queries = await depth_service._generate_exploration_queries(
            root_query="How does the system work?",
            file_chunks=file_chunks,
            file_path="test.py",
        )

        # FakeLLMProvider matches "different aspects" (case-insensitive)
        assert isinstance(queries, list)
        assert len(queries) <= 2  # exploration_queries_per_file=2

    @pytest.mark.asyncio
    async def test_returns_empty_on_llm_failure(self, depth_service, fake_llm_provider):
        """Should return empty list on LLM failure."""
        # Make LLM raise exception
        original_complete = fake_llm_provider.complete_structured

        async def failing_complete(*args, **kwargs):
            raise RuntimeError("LLM API error")

        fake_llm_provider.complete_structured = failing_complete

        try:
            queries = await depth_service._generate_exploration_queries(
                root_query="test query",
                file_chunks=[{"chunk_id": "c1", "content": "code"}],
                file_path="test.py",
            )

            assert queries == []
        finally:
            fake_llm_provider.complete_structured = original_complete

    @pytest.mark.asyncio
    async def test_handles_empty_queries_response(self, depth_service, fake_llm_provider):
        """Should handle LLM returning empty queries array."""
        # Override response to return empty queries
        fake_llm_provider._responses["different aspects"] = '{"queries": []}'

        queries = await depth_service._generate_exploration_queries(
            root_query="test query",
            file_chunks=[{"chunk_id": "c1", "content": "code"}],
            file_path="test.py",
        )

        assert queries == []


# -----------------------------------------------------------------------------
# Tests: explore_coverage_depth (main entry point)
# -----------------------------------------------------------------------------


class TestExploreCoverageDepthEmptyInput:
    """Test explore_coverage_depth with edge cases."""

    @pytest.mark.asyncio
    async def test_returns_original_on_empty_chunks(self, depth_service):
        """Should return original chunks when input is empty."""
        chunks, stats = await depth_service.explore_coverage_depth(
            root_query="test query",
            covered_chunks=[],
            phase1_threshold=0.5,
        )

        assert chunks == []
        assert stats["files_explored"] == 0
        assert stats["queries_generated"] == 0
        assert stats["chunks_added"] == 0

    @pytest.mark.asyncio
    async def test_stats_includes_expected_fields(self, depth_service):
        """Stats should include all expected fields."""
        chunks, stats = await depth_service.explore_coverage_depth(
            root_query="test",
            covered_chunks=[],
            phase1_threshold=0.5,
        )

        expected_fields = ["files_explored", "queries_generated", "chunks_added"]
        for field in expected_fields:
            assert field in stats


# -----------------------------------------------------------------------------
# Tests: Edge Cases
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_select_files_with_identical_scores(self, depth_service):
        """All identical scores should be handled deterministically."""
        file_to_chunks = {f"file{i}.py": [{"rerank_score": 0.8}] for i in range(5)}

        top_files = depth_service._select_top_files(file_to_chunks, 3)

        # Should return exactly 3 files
        assert len(top_files) == 3
        # All should be from original set
        for f in top_files:
            assert f in file_to_chunks

    def test_dedup_preserves_all_metadata(self, depth_service):
        """Dedup should preserve all chunk metadata."""
        results = [
            [
                {
                    "chunk_id": "c1",
                    "content": "code",
                    "file_path": "test.py",
                    "start_line": 1,
                    "end_line": 10,
                    "rerank_score": 0.9,
                    "custom_field": "value",
                }
            ]
        ]

        deduped = depth_service._global_dedup(results)

        chunk = deduped[0]
        assert chunk["chunk_id"] == "c1"
        assert chunk["file_path"] == "test.py"
        assert chunk["start_line"] == 1
        assert chunk["custom_field"] == "value"

    def test_merge_with_many_overlapping_chunks(self, depth_service):
        """Merge should handle many overlapping chunks efficiently."""
        covered = [
            {"chunk_id": f"c{i}", "rerank_score": 0.5 + i * 0.01} for i in range(50)
        ]
        # Overlap with half of coverage
        exploration = [
            {"chunk_id": f"c{i}", "rerank_score": 0.9} for i in range(25)
        ]

        merged = depth_service._merge_coverage(covered, exploration)

        # Should have 50 unique chunks
        chunk_ids = {c["chunk_id"] for c in merged}
        assert len(chunk_ids) == 50
        # First 25 should have exploration's higher score (0.9)
        for chunk in merged:
            if int(chunk["chunk_id"][1:]) < 25:
                assert chunk["rerank_score"] == 0.9

    def test_group_chunks_with_path_variations(self, depth_service):
        """Should handle different path formats."""
        chunks = [
            {"chunk_id": "c1", "file_path": "/abs/path/file.py"},
            {"chunk_id": "c2", "file_path": "relative/file.py"},
            {"chunk_id": "c3", "file_path": "./dotted/file.py"},
        ]

        grouped = depth_service._group_chunks_by_file(chunks)

        assert len(grouped) == 3
        assert "/abs/path/file.py" in grouped
        assert "relative/file.py" in grouped
        assert "./dotted/file.py" in grouped
