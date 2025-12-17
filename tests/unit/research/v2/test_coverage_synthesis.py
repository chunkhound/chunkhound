"""Unit tests for v2 Coverage Synthesis Engine.

Tests the synthesis phase logic without requiring real API calls,
using mock LLM and embedding providers.
"""

import pytest

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.v2.coverage_synthesis import CoverageSynthesisEngine
from tests.fixtures.fake_providers import FakeLLMProvider, FakeEmbeddingProvider


@pytest.fixture
def fake_llm_provider():
    """Create fake LLM provider for synthesis."""
    return FakeLLMProvider(
        responses={
            # Compression responses
            "compress": "## Summary\nThe code implements authentication with session management.\n\nKey components: AuthService, SessionManager, TokenValidator.",
            "code": "## Analysis\nThis cluster handles user authentication and token validation.",
            # Final synthesis
            "synthesis": "## Architecture Overview\nThe system uses a layered authentication architecture with JWT tokens.\n\n## Implementation Details\nAuthService validates credentials, SessionManager persists sessions, and TokenValidator ensures token integrity.\n\n## Data Flow\nUser credentials → AuthService → SessionManager → Token generation → Client",
            "research": "## Deep Analysis\nComprehensive research findings with detailed technical analysis.",
        }
    )


@pytest.fixture
def fake_embedding_provider():
    """Create fake embedding provider for reranking."""
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
def db_services(tmp_path, monkeypatch):
    """Create mock database services with temporary directory."""

    class MockProvider:
        def get_base_directory(self):
            return tmp_path

    class MockDatabaseServices:
        provider = MockProvider()

    return MockDatabaseServices()


@pytest.fixture
def research_config():
    """Create research configuration for testing."""
    return ResearchConfig(
        target_tokens=10000,
        max_chunks_per_file_repr=5,
        max_tokens_per_file_repr=2000,
        max_boundary_expansion_lines=300,
        max_compression_iterations=3,
        min_cluster_size=2,
        shard_budget=20_000,  # Minimum valid value
        gap_similarity_threshold=0.3,
        min_gaps=1,
        max_gaps=5,
        max_symbols=10,  # Maximum is 20
        query_expansion_enabled=True,
    )


@pytest.fixture
def synthesis_engine(llm_manager, embedding_manager, db_services, research_config):
    """Create coverage synthesis engine with mocked dependencies."""
    return CoverageSynthesisEngine(
        llm_manager=llm_manager,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=research_config,
    )


class TestCompoundRerankFiles:
    """Test compound reranking with ROOT + gap queries."""

    @pytest.mark.asyncio
    async def test_reranks_against_all_queries(self, synthesis_engine):
        """Should rerank files against root query + gap queries."""
        root_query = "How does authentication work?"
        gap_queries = ["How is caching implemented?", "What error handling exists?"]
        all_chunks = [
            {
                "chunk_id": "c1",
                "file_path": "auth.py",
                "content": "def authenticate(): pass",
                "rerank_score": 0.9,
                "embedding": [0.1] * 1536,
            },
            {
                "chunk_id": "c2",
                "file_path": "cache.py",
                "content": "class Cache: pass",
                "rerank_score": 0.8,
                "embedding": [0.2] * 1536,
            },
        ]

        file_priorities = await synthesis_engine._compound_rerank_files(
            root_query, all_chunks, gap_queries
        )

        # Should have priority scores for both files
        assert "auth.py" in file_priorities
        assert "cache.py" in file_priorities
        assert isinstance(file_priorities["auth.py"], float)
        assert isinstance(file_priorities["cache.py"], float)

    @pytest.mark.asyncio
    async def test_averages_scores_across_queries(self, synthesis_engine):
        """Should average rerank scores across all compound queries."""
        root_query = "authentication"
        gap_queries = ["caching"]
        all_chunks = [
            {
                "chunk_id": "c1",
                "file_path": "test.py",
                "content": "code",
                "rerank_score": 0.9,
                "embedding": [0.1] * 1536,
            }
        ]

        file_priorities = await synthesis_engine._compound_rerank_files(
            root_query, all_chunks, gap_queries
        )

        # Should average scores from 2 queries (root + 1 gap)
        assert "test.py" in file_priorities
        # Score should be average of rerank results (cosine similarity can be negative)
        assert -1.0 <= file_priorities["test.py"] <= 1.0

    @pytest.mark.asyncio
    async def test_handles_empty_gap_queries(self, synthesis_engine):
        """Should work with just root query when no gap queries."""
        root_query = "authentication"
        gap_queries = []
        all_chunks = [
            {
                "chunk_id": "c1",
                "file_path": "test.py",
                "content": "code",
                "rerank_score": 0.9,
                "embedding": [0.1] * 1536,
            }
        ]

        file_priorities = await synthesis_engine._compound_rerank_files(
            root_query, all_chunks, gap_queries
        )

        assert "test.py" in file_priorities

    @pytest.mark.asyncio
    async def test_groups_chunks_by_file(self, synthesis_engine):
        """Should group chunks from same file together."""
        root_query = "authentication"
        gap_queries = []
        all_chunks = [
            {
                "chunk_id": "c1",
                "file_path": "auth.py",
                "content": "chunk1",
                "rerank_score": 0.9,
                "embedding": [0.1] * 1536,
            },
            {
                "chunk_id": "c2",
                "file_path": "auth.py",
                "content": "chunk2",
                "rerank_score": 0.8,
                "embedding": [0.2] * 1536,
            },
            {
                "chunk_id": "c3",
                "file_path": "cache.py",
                "content": "chunk3",
                "rerank_score": 0.7,
                "embedding": [0.3] * 1536,
            },
        ]

        file_priorities = await synthesis_engine._compound_rerank_files(
            root_query, all_chunks, gap_queries
        )

        # Should have 2 files
        assert len(file_priorities) == 2
        assert "auth.py" in file_priorities
        assert "cache.py" in file_priorities

    @pytest.mark.asyncio
    async def test_falls_back_on_rerank_failure(self, synthesis_engine, monkeypatch):
        """Should use fallback scores when reranking fails."""
        # Mock rerank to raise exception
        async def failing_rerank(query, documents, top_k=None):
            raise Exception("Rerank failed")

        fake_provider = synthesis_engine._embedding_manager.get_provider()
        monkeypatch.setattr(fake_provider, "rerank", failing_rerank)

        root_query = "test"
        gap_queries = []
        all_chunks = [
            {
                "chunk_id": "c1",
                "file_path": "test.py",
                "content": "code",
                "rerank_score": 0.9,
                "embedding": [0.1] * 1536,
            }
        ]

        file_priorities = await synthesis_engine._compound_rerank_files(
            root_query, all_chunks, gap_queries
        )

        # Should still return priorities with fallback scores
        assert "test.py" in file_priorities
        assert file_priorities["test.py"] == 0.5  # Fallback score


@pytest.mark.skip(reason="_allocate_budget removed - v2 reads ALL files, compression handles size")
class TestAllocateBudget:
    """Test token budget allocation (DEPRECATED).

    These tests are for the removed _allocate_budget method.
    V2 uses a different approach:
    - Reads ALL prioritized files (no budget filtering during file reading)
    - Uses recursive compression (_compress_to_budget) to fit output budget
    - Budget only constrains the final synthesis output, not input files

    See TestCompressToBudget for the new compression approach tests.
    """

    @pytest.mark.asyncio
    async def test_includes_files_within_budget(self, synthesis_engine, tmp_path):
        """Should include full files until budget limit."""
        pass

    @pytest.mark.asyncio
    async def test_respects_token_budget(self, synthesis_engine, tmp_path):
        """Should not exceed target token budget."""
        pass

    @pytest.mark.asyncio
    async def test_prioritizes_by_score(self, synthesis_engine, tmp_path):
        """Should include higher priority files first."""
        pass

    @pytest.mark.asyncio
    async def test_returns_budget_stats(self, synthesis_engine, tmp_path):
        """Should return comprehensive budget statistics."""
        pass

    @pytest.mark.asyncio
    async def test_handles_missing_files(self, synthesis_engine, tmp_path):
        """Should handle files that don't exist gracefully."""
        pass


class TestEstimateTokens:
    """Test token estimation helper."""

    def test_estimates_token_count(self, synthesis_engine):
        """Should estimate tokens from text."""
        text = "This is a test string" * 100

        tokens = synthesis_engine._estimate_tokens(text)

        assert tokens > 0
        assert isinstance(tokens, int)

    def test_empty_string_returns_zero(self, synthesis_engine):
        """Should return 0 for empty string."""
        tokens = synthesis_engine._estimate_tokens("")

        assert tokens == 0

    def test_longer_text_more_tokens(self, synthesis_engine):
        """Should estimate more tokens for longer text."""
        short = "short"
        long = "long" * 1000

        short_tokens = synthesis_engine._estimate_tokens(short)
        long_tokens = synthesis_engine._estimate_tokens(long)

        assert long_tokens > short_tokens


class TestBuildCitations:
    """Test citation building from chunks and files."""

    def test_builds_citations_from_chunks(self, synthesis_engine):
        """Should extract file_path and line numbers from chunks."""
        all_chunks = [
            {
                "chunk_id": "c1",
                "file_path": "auth.py",
                "start_line": 10,
                "end_line": 20,
            },
            {
                "chunk_id": "c2",
                "file_path": "cache.py",
                "start_line": 5,
                "end_line": 15,
            },
        ]
        files = {"auth.py": "content", "cache.py": "content"}

        citations = synthesis_engine._build_citations(all_chunks, files)

        assert len(citations) == 2
        assert citations[0]["file_path"] == "auth.py"
        assert citations[0]["start_line"] == 10
        assert citations[0]["end_line"] == 20

    def test_filters_chunks_to_included_files(self, synthesis_engine):
        """Should only include citations for files in synthesis."""
        all_chunks = [
            {"chunk_id": "c1", "file_path": "included.py", "start_line": 1, "end_line": 5},
            {"chunk_id": "c2", "file_path": "excluded.py", "start_line": 1, "end_line": 5},
        ]
        files = {"included.py": "content"}  # Only one file included

        citations = synthesis_engine._build_citations(all_chunks, files)

        # Should only have citation for included file
        assert len(citations) == 1
        assert citations[0]["file_path"] == "included.py"

    def test_handles_empty_inputs(self, synthesis_engine):
        """Should handle empty chunks and files."""
        citations = synthesis_engine._build_citations([], {})

        assert citations == []


class TestExpandBoundaries:
    """Test boundary expansion (placeholder implementation)."""

    @pytest.mark.asyncio
    async def test_returns_content_as_is(self, synthesis_engine):
        """Should currently return budgeted files unchanged."""
        budgeted_files = {
            "test.py": "def foo(): pass",
            "auth.py": "class Auth: pass",
        }

        expanded = await synthesis_engine._expand_boundaries(budgeted_files)

        # Current implementation returns as-is
        assert expanded == budgeted_files

    @pytest.mark.asyncio
    async def test_preserves_all_files(self, synthesis_engine):
        """Should preserve all files through expansion."""
        budgeted_files = {
            "file1.py": "content1",
            "file2.py": "content2",
            "file3.py": "content3",
        }

        expanded = await synthesis_engine._expand_boundaries(budgeted_files)

        assert len(expanded) == len(budgeted_files)
        for file_path in budgeted_files:
            assert file_path in expanded
