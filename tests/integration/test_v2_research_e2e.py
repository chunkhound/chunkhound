"""Integration tests for v2 Coverage-First Research pipeline.

Tests the complete v2 research workflow including:
- Factory instantiation (v1 vs v2)
- Phase execution order
- Path filter propagation
- Return format validation
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.core.types.common import Language
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.mcp_server.tools import deep_research_impl
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.embedding_service import EmbeddingService
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.services.research.factory import ResearchServiceFactory
from chunkhound.services.research.v1.bfs_research_service import BFSResearchService
from chunkhound.services.research.v2.coverage_research_service import (
    CoverageResearchService,
)
from chunkhound.services.search_service import SearchService
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider


@pytest.fixture
def mock_db_services():
    """Create mock database services with search methods.

    Mocks the minimal DatabaseServices interface needed by research services:
    - search_chunks_by_embedding: Vector search
    - search_regex: Regex search
    - get_chunks_by_ids: Bulk chunk retrieval
    """
    mock_db = MagicMock()

    # Mock search methods to return empty results by default
    mock_db.search_chunks_by_embedding = AsyncMock(return_value=([], 0))
    mock_db.search_regex = MagicMock(return_value=([], 0))
    mock_db.get_chunks_by_ids = MagicMock(return_value=[])
    mock_db.get_all_files = MagicMock(return_value=[])
    mock_db.get_file_by_id = MagicMock(return_value=None)

    return mock_db


@pytest.fixture
def mock_embedding_manager():
    """Create mock embedding manager with generate and rerank methods.

    Mocks the EmbeddingManager interface for semantic search:
    - generate_embedding: Returns fake embedding vector
    - generate_embeddings_batch: Returns batch of fake embeddings
    - rerank: Returns reranked results with scores
    """
    mock_embedding = MagicMock()

    # Mock embedding generation
    mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 128)
    mock_embedding.generate_embeddings_batch = AsyncMock(
        return_value=[[0.1] * 128, [0.2] * 128]
    )

    # Mock reranking (returns input with scores)
    async def mock_rerank(query, documents, top_n=None):
        results = []
        for i, doc in enumerate(documents[:top_n] if top_n else documents):
            results.append({**doc, "rerank_score": 0.9 - (i * 0.1)})
        return results

    mock_embedding.rerank = AsyncMock(side_effect=mock_rerank)
    mock_embedding.supports_reranking = True

    return mock_embedding


@pytest.fixture
def mock_llm_manager():
    """Create mock LLM manager with structured output support.

    Mocks the LLMManager interface for LLM operations:
    - complete: Returns text completion
    - complete_structured: Returns structured JSON output
    - get_utility_provider: Returns mock utility provider
    - get_synthesis_provider: Returns mock synthesis provider
    """
    mock_llm = MagicMock()

    # Mock utility provider (fast, cheap)
    mock_utility = MagicMock()
    mock_utility.complete = AsyncMock(return_value="Utility response")
    mock_utility.complete_structured = AsyncMock(
        return_value={
            "queries": ["expanded query 1", "expanded query 2"],
            "questions": ["question 1", "question 2"],
        }
    )
    mock_utility.model = "gpt-4o-mini"

    # Mock synthesis provider (high-quality)
    mock_synthesis = MagicMock()
    mock_synthesis.complete = AsyncMock(
        return_value="# Answer\n\nThis is a synthesized answer with citations."
    )
    mock_synthesis.complete_structured = AsyncMock(
        return_value={"answer": "Synthesized answer", "citations": []}
    )
    mock_synthesis.model = "gpt-4o"

    # Wire up manager
    mock_llm.get_utility_provider = MagicMock(return_value=mock_utility)
    mock_llm.get_synthesis_provider = MagicMock(return_value=mock_synthesis)

    return mock_llm


@pytest.fixture
def v1_research_config():
    """Create Config with v1 (BFS) algorithm."""
    return Config(
        research=ResearchConfig(algorithm="v1"),
    )


@pytest.fixture
def v2_research_config():
    """Create Config with v2 (coverage-first) algorithm."""
    return Config(
        research=ResearchConfig(
            algorithm="v2",
            query_expansion_enabled=True,
            num_expanded_queries=2,
        ),
    )


@pytest.mark.integration
class TestV2ResearchFactory:
    """Test ResearchServiceFactory instantiation logic."""

    def test_factory_creates_v2_service(
        self,
        v2_research_config,
        mock_db_services,
        mock_embedding_manager,
        mock_llm_manager,
    ):
        """Verify factory creates CoverageResearchService when algorithm=v2.

        This tests that the factory correctly routes to v2 implementation
        based on config.research.algorithm setting.
        """
        service = ResearchServiceFactory.create(
            config=v2_research_config,
            db_services=mock_db_services,
            embedding_manager=mock_embedding_manager,
            llm_manager=mock_llm_manager,
        )

        assert isinstance(service, CoverageResearchService)
        assert service._config.algorithm == "v2"

    def test_factory_creates_v1_service_by_default(
        self,
        v1_research_config,
        mock_db_services,
        mock_embedding_manager,
        mock_llm_manager,
    ):
        """Verify factory creates BFSResearchService when algorithm=v1 (default).

        This tests backward compatibility - v1 remains the default algorithm.
        """
        service = ResearchServiceFactory.create(
            config=v1_research_config,
            db_services=mock_db_services,
            embedding_manager=mock_embedding_manager,
            llm_manager=mock_llm_manager,
        )

        assert isinstance(service, BFSResearchService)

    def test_factory_propagates_path_filter(
        self,
        v2_research_config,
        mock_db_services,
        mock_embedding_manager,
        mock_llm_manager,
    ):
        """Verify factory propagates path_filter to service instance.

        Path filter is used to limit research scope to specific directories.
        """
        service = ResearchServiceFactory.create(
            config=v2_research_config,
            db_services=mock_db_services,
            embedding_manager=mock_embedding_manager,
            llm_manager=mock_llm_manager,
            path_filter="src/",
        )

        assert service._path_filter == "src/"


@pytest.mark.integration
@pytest.mark.asyncio
class TestV2ResearchEndToEnd:
    """End-to-end tests for v2 research pipeline."""

    async def test_v2_deep_research_returns_expected_format(
        self,
        v2_research_config,
        mock_db_services,
        mock_embedding_manager,
        mock_llm_manager,
    ):
        """Verify deep_research returns correct structure with all required fields.

        Tests that the v2 service returns:
        - answer: Synthesized text
        - metadata: Contains phase timings and statistics
        - gap_queries: Optional list of gap queries filled
        """
        service = CoverageResearchService(
            database_services=mock_db_services,
            embedding_manager=mock_embedding_manager,
            llm_manager=mock_llm_manager,
            config=v2_research_config,
        )

        # Mock sub-services to return minimal valid results
        # Phase 1: Coverage
        with patch.object(
            service._unified_search,
            "unified_search",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = [
                {
                    "chunk_id": 1,
                    "code": "def test(): pass",
                    "rerank_score": 0.95,
                    "file_id": 1,
                    "start_line": 1,
                    "end_line": 1,
                }
            ]

            # Phase 2: Gap detection (no gaps found)
            with patch.object(
                service._gap_detection_service,
                "detect_and_fill_gaps",
                new_callable=AsyncMock,
            ) as mock_gaps:
                mock_gaps.return_value = (
                    mock_search.return_value,  # Same chunks (no gaps)
                    {
                        "gaps_found": 0,
                        "gaps_filled": 0,
                        "chunks_added": 0,
                        "gap_queries": [],
                    },
                )

                # Phase 3: Synthesis
                with patch.object(
                    service._synthesis_engine,
                    "synthesize",
                    new_callable=AsyncMock,
                ) as mock_synthesis:
                    mock_synthesis.return_value = (
                        "This is a test answer.",
                        [],
                        {"final_tokens": 1000, "token_budget": {}},
                    )

                    # Execute research
                    result = await service.deep_research("test query")

        # Verify structure
        assert "answer" in result
        assert "metadata" in result
        assert "gap_queries" in result

        # Verify answer content
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

        # Verify metadata contains phase timings
        metadata = result["metadata"]
        assert "phase_timings" in metadata
        assert "phase1_ms" in metadata["phase_timings"]
        assert "phase2_ms" in metadata["phase_timings"]
        assert "phase3_ms" in metadata["phase_timings"]
        assert "total_ms" in metadata["phase_timings"]

        # Verify phase timing values are reasonable
        assert metadata["phase_timings"]["phase1_ms"] >= 0
        assert metadata["phase_timings"]["total_ms"] > 0

    async def test_v2_research_respects_path_filter(
        self,
        v2_research_config,
        mock_db_services,
        mock_embedding_manager,
        mock_llm_manager,
    ):
        """Verify path_filter is propagated to search operations.

        Tests that when a path_filter is set, it's passed through to
        all search operations (unified_search, gap filling).
        """
        service = CoverageResearchService(
            database_services=mock_db_services,
            embedding_manager=mock_embedding_manager,
            llm_manager=mock_llm_manager,
            config=v2_research_config,
            path_filter="src/core/",
        )

        # Mock unified search to capture path_filter argument
        with patch.object(
            service._unified_search,
            "unified_search",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = []

            # Mock gap detection to avoid Phase 2
            with patch.object(
                service._gap_detection_service,
                "detect_and_fill_gaps",
                new_callable=AsyncMock,
            ) as mock_gaps:
                mock_gaps.return_value = (
                    [],
                    {
                        "gaps_found": 0,
                        "gaps_filled": 0,
                        "chunks_added": 0,
                        "gap_queries": [],
                    },
                )

                # Mock synthesis to avoid Phase 3
                with patch.object(
                    service._synthesis_engine,
                    "synthesize",
                    new_callable=AsyncMock,
                ) as mock_synthesis:
                    mock_synthesis.return_value = ("answer", [], {})

                    await service.deep_research("test query")

        # Verify path_filter was passed to unified_search
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args.kwargs
        assert "path_filter" in call_kwargs
        assert call_kwargs["path_filter"] == "src/core/"

    async def test_v2_phases_execute_in_order(self, tmp_path: Path):
        """Verify v2 research pipeline executes all phases without mocks.

        Uses FakeLLMProvider + FakeEmbeddingProvider + in-memory DB.
        Verifies output structure indicates all phases ran.
        """
        # 1. Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text(
            "def alpha():\n"
            "    '''Test function for research.'''\n"
            "    return 'test'\n",
            encoding="utf-8",
        )

        # 2. Setup embedding manager with fake provider
        embedding_manager = EmbeddingManager()
        embedding_manager.register_provider(FakeEmbeddingProvider(), set_default=True)

        # 3. Patch LLMManager to use FakeLLMProvider
        def _fake_create_provider(self, config):
            return FakeLLMProvider()

        original_create_provider = LLMManager._create_provider
        LLMManager._create_provider = _fake_create_provider  # type: ignore[assignment]
        try:
            llm_manager = LLMManager(
                {"provider": "fake", "model": "fake-gpt"},
                {"provider": "fake", "model": "fake-gpt"},
            )
        finally:
            LLMManager._create_provider = original_create_provider  # type: ignore[assignment]

        # 4. Create in-memory database and index test file
        db = DuckDBProvider(":memory:", base_directory=tmp_path)
        db.connect()
        parser = create_parser_for_language(Language.PYTHON)
        coordinator = IndexingCoordinator(
            db,
            tmp_path,
            embedding_manager.get_default_provider(),
            {Language.PYTHON: parser},
        )
        await coordinator.process_file(test_file)

        # 5. Create services bundle
        services = DatabaseServices(
            provider=db,
            indexing_coordinator=coordinator,
            search_service=SearchService(db, embedding_manager.get_default_provider()),
            embedding_service=EmbeddingService(
                db, embedding_manager.get_default_provider()
            ),
        )

        # 6. Create v2 config and disable relevance threshold for small test corpus
        v2_config = Config(
            research=ResearchConfig(algorithm="v2", query_expansion_enabled=True),
        )

        import chunkhound.services.research.shared.models as models_mod

        original_threshold = models_mod.RELEVANCE_THRESHOLD
        models_mod.RELEVANCE_THRESHOLD = None
        try:
            # 7. Run v2 research
            result = await deep_research_impl(
                services=services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                query="alpha function",
                progress=None,
                path=None,
                config=v2_config,
            )
        finally:
            models_mod.RELEVANCE_THRESHOLD = original_threshold

        # 8. Verify all phases ran via output structure
        assert "answer" in result
        assert "metadata" in result
        metadata = result["metadata"]
        assert "phase_timings" in metadata
        assert metadata["phase_timings"]["phase1_ms"] >= 0
        assert metadata["phase_timings"]["phase2_ms"] >= 0
        assert metadata["phase_timings"]["phase3_ms"] >= 0

    async def test_v2_phase_timings_recorded(
        self,
        v2_research_config,
        mock_db_services,
        mock_embedding_manager,
        mock_llm_manager,
    ):
        """Verify phase timings are recorded in metadata.

        Tests that each phase execution time is tracked and returned
        in metadata.phase_timings with millisecond precision.
        """
        service = CoverageResearchService(
            database_services=mock_db_services,
            embedding_manager=mock_embedding_manager,
            llm_manager=mock_llm_manager,
            config=v2_research_config,
        )

        # Mock all phases with minimal delay
        with patch.object(
            service._unified_search,
            "unified_search",
            new_callable=AsyncMock,
            return_value=[{"chunk_id": 1, "rerank_score": 0.9}],
        ):
            with patch.object(
                service._gap_detection_service,
                "detect_and_fill_gaps",
                new_callable=AsyncMock,
                return_value=(
                    [{"chunk_id": 1}],
                    {
                        "gaps_found": 0,
                        "gaps_filled": 0,
                        "chunks_added": 0,
                        "gap_queries": [],
                    },
                ),
            ):
                with patch.object(
                    service._synthesis_engine,
                    "synthesize",
                    new_callable=AsyncMock,
                    return_value=("answer", [], {"final_tokens": 1000}),
                ):
                    result = await service.deep_research("test query")

        # Verify timings exist and are positive
        timings = result["metadata"]["phase_timings"]
        assert timings["phase1_ms"] >= 0
        assert timings["phase2_ms"] >= 0
        assert timings["phase3_ms"] >= 0
        assert timings["total_ms"] >= timings["phase1_ms"]
        assert timings["total_ms"] >= timings["phase2_ms"]
        assert timings["total_ms"] >= timings["phase3_ms"]

    async def test_v2_handles_query_expansion_disabled(
        self,
        mock_db_services,
        mock_embedding_manager,
        mock_llm_manager,
    ):
        """Verify v2 works correctly when query expansion is disabled.

        Tests that when query_expansion_enabled=False, the service
        uses only the original query without LLM expansion.
        """
        config = Config(
            research=ResearchConfig(
                algorithm="v2",
                query_expansion_enabled=False,
            ),
        )

        service = CoverageResearchService(
            database_services=mock_db_services,
            embedding_manager=mock_embedding_manager,
            llm_manager=mock_llm_manager,
            config=config,
        )

        # Mock unified search to capture expanded_queries argument
        with patch.object(
            service._unified_search,
            "unified_search",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = []

            # Mock other phases
            with patch.object(
                service._gap_detection_service,
                "detect_and_fill_gaps",
                new_callable=AsyncMock,
                return_value=(
                    [],
                    {
                        "gaps_found": 0,
                        "gaps_filled": 0,
                        "chunks_added": 0,
                        "gap_queries": [],
                    },
                ),
            ):
                with patch.object(
                    service._synthesis_engine,
                    "synthesize",
                    new_callable=AsyncMock,
                    return_value=("answer", [], {}),
                ):
                    await service.deep_research("test query")

        # Verify expanded_queries contains only original query
        call_kwargs = mock_search.call_args.kwargs
        assert "expanded_queries" in call_kwargs
        assert call_kwargs["expanded_queries"] == ["test query"]
        assert len(call_kwargs["expanded_queries"]) == 1

    async def test_v2_gap_queries_in_result(
        self,
        v2_research_config,
        mock_db_services,
        mock_embedding_manager,
        mock_llm_manager,
    ):
        """Verify gap_queries field is properly propagated in result.

        Tests that gap_queries are correctly returned when gaps are detected
        and filled during Phase 2. Gap queries are propagated from
        gap_detection_service through the result metadata.
        """
        service = CoverageResearchService(
            database_services=mock_db_services,
            embedding_manager=mock_embedding_manager,
            llm_manager=mock_llm_manager,
            config=v2_research_config,
        )

        # Mock with gaps found and filled
        gap_queries = ["What about error handling?", "How does caching work?"]

        with patch.object(
            service._unified_search,
            "unified_search",
            new_callable=AsyncMock,
            return_value=[{"chunk_id": 1, "rerank_score": 0.9}],
        ):
            with patch.object(
                service._gap_detection_service,
                "detect_and_fill_gaps",
                new_callable=AsyncMock,
            ) as mock_gaps:
                mock_gaps.return_value = (
                    [{"chunk_id": 1}, {"chunk_id": 2}],
                    {
                        "gaps_found": 2,
                        "gaps_filled": 2,
                        "chunks_added": 1,
                        "gap_queries": gap_queries,
                    },
                )

                with patch.object(
                    service._synthesis_engine,
                    "synthesize",
                    new_callable=AsyncMock,
                    return_value=("answer", [], {}),
                ):
                    result = await service.deep_research("test query")

        # Verify gap_queries are properly propagated
        assert "gap_queries" in result
        assert result["gap_queries"] == gap_queries
