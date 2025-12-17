"""Sociable tests for empty result propagation in v2 Coverage Research Service.

Tests edge cases where different phases return empty or zero results,
using real service implementations with fake external dependencies.

Sociable Testing Approach:
- Real services: UnifiedSearch, GapDetectionService, CoverageSynthesisEngine
- Real database: LanceDB (empty or with indexed content)
- Fake providers: FakeLLMProvider, FakeEmbeddingProvider (avoid API calls)
"""

import asyncio
import os

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.v2.coverage_research_service import (
    CoverageResearchService,
)
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider


# =============================================================================
# Fake Provider Fixtures (external dependencies only)
# =============================================================================


@pytest.fixture
def fake_llm_provider():
    """Configure FakeLLMProvider with v2 research responses.

    This provider returns deterministic responses for all LLM calls:
    - Query expansion: Returns expanded query list
    - Gap detection: Returns empty gaps (for empty result tests)
    - Synthesis: Returns formatted answer (must be >100 chars)
    - Aspect generation: Returns empty aspects
    """
    # Synthesis response must be at least 100 characters to pass validation
    synthesis_response = """## Architecture Overview

The authentication system implements a secure login flow with the following key components:

### Components
- Authentication module handles credential validation
- Session management tracks active user sessions
- Password hashing provides secure storage

### Flow
1. User submits credentials
2. System validates against database
3. Session token generated on success
"""

    return FakeLLMProvider(
        responses={
            # Query expansion (structured output)
            "expand": '{"queries": ["authentication implementation", "auth flow design"]}',
            # Gap detection (structured output) - empty gaps for testing
            "gap": '{"gaps": []}',
            "detect": '{"gaps": []}',
            # Synthesis (text completion) - must be >100 chars
            "synthesis": synthesis_response,
            "answer": synthesis_response,
            "analyz": synthesis_response,  # For analyze prompts
            # Aspect generation (for depth exploration)
            "aspect": '{"aspects": []}',
            "question": '{"questions": []}',
            # Compression prompts
            "compress": "Compressed summary: Authentication validates credentials and manages sessions.",
            "summar": "Summary: The code handles user authentication and session management.",
            # Default catch-all for any prompt
            "default": synthesis_response,
        }
    )


@pytest.fixture
def fake_embedding_provider():
    """Create FakeEmbeddingProvider with 1536 dimensions."""
    return FakeEmbeddingProvider(dims=1536)


# =============================================================================
# Real Manager Fixtures (wired to fake providers)
# =============================================================================


@pytest.fixture
def real_llm_manager(fake_llm_provider, monkeypatch):
    """Create real LLMManager that uses FakeLLMProvider.

    The LLMManager is real, but its internal provider creation
    is patched to return our fake provider.
    """

    def mock_create_provider(self, config):
        return fake_llm_provider

    monkeypatch.setattr(LLMManager, "_create_provider", mock_create_provider)

    return LLMManager(
        utility_config={"provider": "fake", "model": "fake-utility"},
        synthesis_config={"provider": "fake", "model": "fake-synthesis"},
    )


@pytest.fixture
def real_embedding_manager(fake_embedding_provider):
    """Create real EmbeddingManager wired to fake provider.

    Uses the standard register_provider interface.
    """
    manager = EmbeddingManager()
    manager.register_provider(fake_embedding_provider, set_default=True)
    return manager


# =============================================================================
# Configuration Fixtures
# =============================================================================


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
        shard_budget=20_000,
        gap_similarity_threshold=0.3,
        min_gaps=1,
        max_gaps=5,
        max_symbols=10,
        query_expansion_enabled=True,
    )


@pytest.fixture
def config(research_config):
    """Create full Config object with research settings."""
    return Config(research=research_config)


# =============================================================================
# Real Database Fixtures
# =============================================================================


@pytest.fixture
def empty_lancedb_services(tmp_path, real_embedding_manager, fake_embedding_provider):
    """Create real DatabaseServices with an empty LanceDB.

    Manually constructs services to avoid registry configuration issues.
    The empty database naturally produces empty search results.
    """
    pytest.importorskip("lancedb")

    from chunkhound.core.config.database_config import DatabaseConfig
    from chunkhound.database_factory import DatabaseServices
    from chunkhound.providers.database.lancedb_provider import LanceDBProvider
    from chunkhound.services.embedding_service import EmbeddingService
    from chunkhound.services.indexing_coordinator import IndexingCoordinator
    from chunkhound.services.search_service import SearchService

    # Configure LanceDB path properly
    db_config = DatabaseConfig(path=tmp_path, provider="lancedb")
    db_path = db_config.get_db_path()

    # Create provider directly
    provider = LanceDBProvider(str(db_path), base_directory=tmp_path)
    provider.connect()

    # Create minimal services directly (without registry)
    embedding_service = EmbeddingService(provider, fake_embedding_provider)
    search_service = SearchService(provider, fake_embedding_provider)
    indexing_coordinator = IndexingCoordinator(
        database_provider=provider,
        embedding_provider=fake_embedding_provider,
        base_directory=tmp_path,
    )

    services = DatabaseServices(
        provider=provider,
        indexing_coordinator=indexing_coordinator,
        search_service=search_service,
        embedding_service=embedding_service,
    )

    yield services

    provider.disconnect()


@pytest.fixture
def lancedb_services_with_chunks(tmp_path, real_embedding_manager, fake_embedding_provider):
    """Create real DatabaseServices with one indexed file.

    Manually constructs services to avoid registry configuration issues.
    """
    pytest.importorskip("lancedb")

    from chunkhound.core.config.database_config import DatabaseConfig
    from chunkhound.database_factory import DatabaseServices
    from chunkhound.providers.database.lancedb_provider import LanceDBProvider
    from chunkhound.services.embedding_service import EmbeddingService
    from chunkhound.services.indexing_coordinator import IndexingCoordinator
    from chunkhound.services.search_service import SearchService

    # Configure LanceDB path properly
    db_config = DatabaseConfig(path=tmp_path, provider="lancedb")
    db_path = db_config.get_db_path()

    # Create provider directly
    provider = LanceDBProvider(str(db_path), base_directory=tmp_path)
    provider.connect()

    # Create minimal services directly (without registry)
    embedding_service = EmbeddingService(provider, fake_embedding_provider)
    search_service = SearchService(provider, fake_embedding_provider)
    indexing_coordinator = IndexingCoordinator(
        database_provider=provider,
        embedding_provider=fake_embedding_provider,
        base_directory=tmp_path,
    )

    services = DatabaseServices(
        provider=provider,
        indexing_coordinator=indexing_coordinator,
        search_service=search_service,
        embedding_service=embedding_service,
    )

    # Index a small file to provide Phase 1 results
    test_file = tmp_path / "auth.py"
    test_file.write_text(
        '''"""Authentication module."""


def authenticate(username: str, password: str) -> bool:
    """Authenticate a user with username and password.

    Args:
        username: The user's username
        password: The user's password

    Returns:
        True if authentication succeeds, False otherwise
    """
    # Check credentials against database
    if username == "admin" and password == "secret":
        return True
    return False


def logout(session_id: str) -> None:
    """Log out a user session.

    Args:
        session_id: The session identifier to invalidate
    """
    # Invalidate the session
    pass
'''
    )

    # Use real indexing to populate database
    asyncio.run(indexing_coordinator.process_file(test_file))

    yield services

    provider.disconnect()


# =============================================================================
# Sociable Service Fixtures
# =============================================================================


@pytest.fixture
def coverage_research_service_empty(
    empty_lancedb_services,
    real_embedding_manager,
    real_llm_manager,
    config,
):
    """Sociable test: Empty database → Phase 1 returns no chunks.

    Real services execute against empty LanceDB.
    Expected behavior:
    - Phase 1: Returns empty chunks (real search finds nothing)
    - Phase 2: Receives empty, returns empty with zero stats
    - Phase 3: Raises ValueError (real synthesis code path)
    """
    return CoverageResearchService(
        database_services=empty_lancedb_services,
        embedding_manager=real_embedding_manager,
        llm_manager=real_llm_manager,
        config=config,
        tool_name="code_research",
        progress=None,
        path_filter=None,
    )


@pytest.fixture
def coverage_research_service_no_new_gaps(
    lancedb_services_with_chunks,
    real_embedding_manager,
    real_llm_manager,
    config,
):
    """Sociable test: Phase 1 returns chunks, Phase 2 adds nothing new.

    Real services execute with indexed content.
    Expected behavior:
    - Phase 1: Returns chunks from indexed file
    - Phase 2: Detects gaps but may add 0 NEW chunks (depends on search)
    - Phase 3: Synthesizes successfully with available chunks
    """
    return CoverageResearchService(
        database_services=lancedb_services_with_chunks,
        embedding_manager=real_embedding_manager,
        llm_manager=real_llm_manager,
        config=config,
        tool_name="code_research",
        progress=None,
        path_filter=None,
    )


# =============================================================================
# Test Classes
# =============================================================================


class TestEmptyResultPropagation:
    """Test empty result handling throughout the research pipeline.

    These tests use real services with an empty database to verify
    that empty results are handled gracefully throughout the pipeline.

    Note: With real services, empty results may still proceed to synthesis
    (the LLM generates a response even without code context). This is
    different from mocked tests which could short-circuit early.
    """

    @pytest.mark.asyncio
    async def test_phase1_empty_still_completes_or_errors_gracefully(
        self, coverage_research_service_empty, fake_llm_provider
    ):
        """Phase 1 returns zero chunks -> Pipeline handles gracefully.

        With real services and empty database:
        - Phase 1: Returns empty chunks list (real search finds nothing)
        - Phase 2: Returns zero stats (no chunks to analyze)
        - Phase 3: Either completes with generic answer OR raises RuntimeError

        The pipeline should not crash unexpectedly.
        """
        query = "How does authentication work?"

        # Reset LLM usage stats
        fake_llm_provider._requests_made = 0
        fake_llm_provider._tokens_used = 0

        # Pipeline should either complete or raise a handled error
        try:
            result = await coverage_research_service_empty.deep_research(query)
            # If we get here, synthesis completed with empty context
            assert "answer" in result
            assert "metadata" in result
        except (ValueError, RuntimeError) as e:
            # Expected - synthesis may fail with empty context
            # Just verify it's a graceful failure, not an unhandled crash
            assert str(e)  # Has an error message

    @pytest.mark.asyncio
    async def test_phase1_chunks_phase2_synthesis_succeeds(
        self, coverage_research_service_no_new_gaps, fake_llm_provider
    ):
        """Phase 1 returns chunks -> Phase 3 synthesizes successfully.

        Expected behavior:
        - Phase 1: Returns chunks from indexed file
        - Phase 2: Processes coverage (may add or not add new chunks)
        - Phase 3: Synthesizes successfully
        - Final result: Success with proper metadata
        """
        query = "How does authentication work?"

        # Should complete successfully
        result = await coverage_research_service_no_new_gaps.deep_research(query)

        # Verify result structure
        assert "answer" in result
        assert "metadata" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

        # Verify metadata exists
        metadata = result["metadata"]
        assert "phase1_chunks" in metadata
        assert metadata["phase1_chunks"] >= 0

    @pytest.mark.asyncio
    async def test_phase1_empty_produces_metadata(
        self, coverage_research_service_empty, fake_llm_provider
    ):
        """Phase 1 returns zero chunks -> Metadata reflects empty state.

        With real services, we can call internal phase methods to verify
        that the phases correctly report empty state.
        """
        query = "How does authentication work?"

        # Reset LLM usage stats
        fake_llm_provider._requests_made = 0

        # Call Phase 1 directly to inspect results
        phase1_result = await coverage_research_service_empty._phase1_coverage(query)

        # Verify Phase 1 reports empty
        assert len(phase1_result.chunks) == 0
        assert len(phase1_result.symbols) == 0

    @pytest.mark.asyncio
    async def test_empty_database_doesnt_crash(
        self, coverage_research_service_empty
    ):
        """Empty database -> Service handles gracefully without crashing.

        The key assertion is that the service doesn't throw an unhandled
        exception. Handled errors (ValueError, RuntimeError) are acceptable.
        """
        query = "How does the nonexistent feature work?"

        # Should not throw an unexpected exception
        try:
            result = await coverage_research_service_empty.deep_research(query)
            # Success is fine
            assert "answer" in result
        except (ValueError, RuntimeError):
            # These are expected graceful failures
            pass
        except Exception as e:
            # Any other exception type is unexpected
            pytest.fail(f"Unexpected exception type: {type(e).__name__}: {e}")


class TestPhase2GapStatsEdgeCases:
    """Test gap statistics accuracy in edge cases.

    These tests directly call internal phase methods to verify
    the stats returned by each phase.
    """

    @pytest.mark.asyncio
    async def test_phase2_stats_when_phase1_empty(
        self, coverage_research_service_empty
    ):
        """Verify Phase 2 gap stats structure when Phase 1 returns empty chunks.

        Expected stats with real services:
        - gaps_found: 0 (no chunks to analyze)
        - gaps_filled: 0
        - chunks_added: 0
        """
        query = "test query"

        # Manually call phase1 and phase2 to inspect stats
        service = coverage_research_service_empty

        # Phase 1
        phase1_result = await service._phase1_coverage(query)
        assert len(phase1_result.chunks) == 0
        assert len(phase1_result.symbols) == 0

        # Phase 2
        all_chunks, gap_stats = await service._phase2_gap_detection(query, phase1_result)

        # Verify gap stats structure for empty input
        assert gap_stats["gaps_found"] == 0
        assert gap_stats["chunks_added"] == 0
        assert gap_stats["gaps_filled"] == 0

        # Verify all_chunks is still empty
        assert len(all_chunks) == 0

    @pytest.mark.asyncio
    async def test_phase2_stats_when_phase1_has_chunks(
        self, coverage_research_service_no_new_gaps
    ):
        """Verify Phase 2 processes chunks from Phase 1.

        Expected behavior:
        - Phase 1: Returns chunks from indexed file
        - Phase 2: Processes the chunks (may detect gaps)
        - Core stats are properly populated
        """
        query = "How does authentication work?"

        # Manually call phase1 and phase2 to inspect stats
        service = coverage_research_service_no_new_gaps

        # Phase 1 - should have chunks from indexed file
        phase1_result = await service._phase1_coverage(query)
        assert len(phase1_result.chunks) >= 1  # Should have at least 1 chunk

        # Phase 2
        all_chunks, gap_stats = await service._phase2_gap_detection(query, phase1_result)

        # Verify core gap stats exist
        assert "gaps_found" in gap_stats
        assert "chunks_added" in gap_stats
        assert "gaps_filled" in gap_stats

        # Verify chunks are preserved or enhanced
        assert len(all_chunks) >= len(phase1_result.chunks)


class TestPhase3SynthesisEmptyHandling:
    """Test Phase 3 synthesis behavior with empty inputs.

    These tests verify that the real synthesis engine handles
    empty chunks gracefully. Note: Real services may complete
    synthesis even with empty context (LLM generates a response).
    """

    @pytest.mark.asyncio
    async def test_synthesis_handles_empty_chunks_gracefully(
        self, coverage_research_service_empty
    ):
        """Verify Phase 3 handles empty chunks gracefully.

        With real services, synthesis may:
        - Complete successfully (LLM generates generic answer)
        - Raise RuntimeError (LLM output validation failure)
        - Raise ValueError (early empty check)

        All are acceptable graceful outcomes.
        """
        query = "test query"

        # Pipeline should handle empty gracefully
        try:
            result = await coverage_research_service_empty.deep_research(query)
            # Success means LLM generated a valid response
            assert "answer" in result
        except (ValueError, RuntimeError):
            # These are expected graceful failures
            pass
        except Exception as e:
            # Unexpected exception type
            pytest.fail(f"Unexpected exception: {type(e).__name__}: {e}")

    @pytest.mark.asyncio
    async def test_synthesis_succeeds_with_chunks(
        self, coverage_research_service_no_new_gaps
    ):
        """Verify Phase 3 synthesizes successfully when chunks are available.

        Expected behavior:
        - Returns answer string
        - Returns metadata with synthesis stats
        """
        query = "How does authentication work?"

        result = await coverage_research_service_no_new_gaps.deep_research(query)

        # Verify answer is returned
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

        # Verify metadata exists
        assert "metadata" in result
        metadata = result["metadata"]
        assert "phase_timings" in metadata
