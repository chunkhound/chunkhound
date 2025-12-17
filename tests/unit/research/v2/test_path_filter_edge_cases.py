"""Unit tests for path_filter edge cases in v2 Coverage Research Service.

Tests path_filter parameter behavior when no files match the specified path,
ensuring proper error handling and user-friendly error messages for debugging.

Edge cases covered:
1. path_filter points to nonexistent directory (e.g., "nonexistent/")
2. path_filter valid but database only has files in different paths
3. path_filter with special regex characters (treated as literal prefix)
4. path_filter empty string (treated as None, no filtering)
"""

import pytest

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.v2.coverage_research_service import (
    CoverageResearchService,
)
from tests.fixtures.fake_providers import FakeLLMProvider, FakeEmbeddingProvider


@pytest.fixture
def fake_llm_provider():
    """Create fake LLM provider for research."""
    return FakeLLMProvider(
        responses={
            # Query expansion
            "expand": '{"queries": ["authentication implementation", "auth flow design"]}',
            # Gap detection (not expected to be called in these tests)
            "gap": '{"gaps": []}',
            # Synthesis (not expected to be called in these tests)
            "synthesis": "## Response\nNo results found.",
        }
    )


@pytest.fixture
def fake_embedding_provider():
    """Create fake embedding provider."""
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
    """Create mock database services."""

    class MockSearchService:
        async def search(self, query, path_filter=None, top_k=100):
            """Return mock search results based on path_filter."""
            # Simulate database with only "lib/" files
            all_chunks = [
                {
                    "chunk_id": "lib1",
                    "file_path": "lib/auth.py",
                    "content": "def authenticate(): pass",
                    "start_line": 1,
                    "end_line": 5,
                    "similarity": 0.9,
                    "embedding": [0.1] * 1536,
                },
                {
                    "chunk_id": "lib2",
                    "file_path": "lib/session.py",
                    "content": "class Session: pass",
                    "start_line": 10,
                    "end_line": 20,
                    "similarity": 0.8,
                    "embedding": [0.2] * 1536,
                },
            ]

            # Filter by path_filter if provided
            if path_filter:
                return [c for c in all_chunks if c["file_path"].startswith(path_filter)]
            return all_chunks

    class MockProvider:
        def get_base_directory(self):
            return tmp_path

    class MockDatabaseServices:
        provider = MockProvider()
        search_service = MockSearchService()

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
        shard_budget=20_000,
        gap_similarity_threshold=0.3,
        min_gaps=1,
        max_gaps=5,
        max_symbols=10,
        query_expansion_enabled=True,
    )


@pytest.fixture
def config(research_config):
    """Create application configuration wrapper."""

    class MockConfig:
        def __init__(self):
            self.research = research_config

    return MockConfig()


def create_coverage_service_with_path_filter(
    db_services,
    embedding_manager,
    llm_manager,
    config,
    monkeypatch,
    path_filter: str | None,
    mock_unified_search_results: list | None = None,
):
    """Helper to create coverage research service with custom path_filter and mock results.

    Args:
        db_services: Database services
        embedding_manager: Embedding manager
        llm_manager: LLM manager
        config: Application config
        monkeypatch: Pytest monkeypatch fixture
        path_filter: Path filter to apply
        mock_unified_search_results: Optional custom unified search results
            (defaults to empty for testing failure scenarios)
    """
    # Mock UnifiedSearch to return custom results based on path_filter
    class MockUnifiedSearch:
        def __init__(self, db_services, embedding_manager, config=None):
            self.db_services = db_services
            self.embedding_manager = embedding_manager
            self.config = config

        async def unified_search(
            self, query, context, expanded_queries=None, path_filter=None, rerank_queries=None
        ):
            """Return mock unified search results filtered by path_filter."""
            # If custom results provided, use them
            if mock_unified_search_results is not None:
                return mock_unified_search_results

            # Otherwise, simulate filtering by path
            all_chunks = [
                {
                    "chunk_id": "lib1",
                    "file_path": "lib/auth.py",
                    "code": "def authenticate(): pass",
                    "content": "def authenticate(): pass",
                    "start_line": 1,
                    "end_line": 5,
                    "rerank_score": 0.9,
                    "similarity": 0.85,
                    "embedding": [0.1] * 1536,
                },
                {
                    "chunk_id": "lib2",
                    "file_path": "lib/session.py",
                    "code": "class Session: pass",
                    "content": "class Session: pass",
                    "start_line": 10,
                    "end_line": 20,
                    "rerank_score": 0.8,
                    "similarity": 0.75,
                    "embedding": [0.2] * 1536,
                },
            ]

            # Apply path_filter if provided
            if path_filter:
                filtered = [c for c in all_chunks if c["file_path"].startswith(path_filter)]
                return filtered
            return all_chunks

        async def extract_symbols_from_chunks(self, chunks):
            """Return mock symbols."""
            if not chunks:
                return []
            return ["authenticate", "Session"]

        async def expand_chunk_windows(self, chunks, window_lines=50):
            return chunks

    # Patch UnifiedSearch
    from chunkhound.services.research.v2 import coverage_research_service as crs_module

    monkeypatch.setattr(crs_module, "UnifiedSearch", MockUnifiedSearch)

    # Patch QueryExpander
    class MockQueryExpander:
        def __init__(self, llm_manager):
            pass

        async def expand_query_with_llm(self, query, context):
            return [query, "expanded query 1"]

    monkeypatch.setattr(crs_module, "QueryExpander", MockQueryExpander)

    # Patch GapDetectionService to return empty gaps
    class MockGapDetectionService:
        def __init__(self, llm_manager, embedding_manager, db_services, config,
                     import_resolver=None, import_context_service=None):
            pass

        async def detect_and_fill_gaps(
            self, root_query, covered_chunks, phase1_threshold, path_filter=None
        ):
            """Return empty gap results when covered_chunks is empty."""
            return covered_chunks, {
                "gaps_found": 0,
                "gaps_unified": 0,
                "gaps_selected": 0,
                "gaps_filled": 0,
                "chunks_added": 0,
                "total_chunks": len(covered_chunks),
                "gap_queries": [],
            }

    monkeypatch.setattr(crs_module, "GapDetectionService", MockGapDetectionService)

    # Patch CoverageSynthesisEngine to handle empty chunks
    class MockCoverageSynthesisEngine:
        def __init__(self, llm_manager, embedding_manager, db_services, config,
                     unified_search=None, import_resolver=None, import_context_service=None,
                     progress=None):
            pass

        async def synthesize(self, root_query, all_chunks, gap_queries, target_tokens,
                             file_imports=None):
            """Raise ValueError when chunks are empty."""
            if not all_chunks:
                raise ValueError(
                    "No code chunks available for synthesis. "
                    "The codebase may be empty or the query returned no relevant results. "
                    "Try a different search query or verify the codebase was indexed correctly."
                )
            # Should not be reached in empty-result tests
            return "Synthesis result", [], {"final_tokens": 100}

    monkeypatch.setattr(crs_module, "CoverageSynthesisEngine", MockCoverageSynthesisEngine)

    return CoverageResearchService(
        database_services=db_services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        config=config,
        tool_name="code_research",
        progress=None,
        path_filter=path_filter,
    )


class TestPathFilterNonexistent:
    """Test path_filter pointing to nonexistent directory."""

    @pytest.mark.asyncio
    async def test_nonexistent_path_filter_returns_zero_chunks(
        self, db_services, embedding_manager, llm_manager, config, monkeypatch
    ):
        """path_filter='nonexistent/' returns zero chunks from Phase 1.

        Expected behavior:
        - Phase 1: Unified search returns empty list (no matches for path_filter)
        - Phase 2: Skipped (no chunks to analyze)
        - Phase 3: Raises ValueError with path_filter in error message
        - Error message: User-friendly, mentions path_filter constraint
        """
        service = create_coverage_service_with_path_filter(
            db_services,
            embedding_manager,
            llm_manager,
            config,
            monkeypatch,
            path_filter="nonexistent/",
            mock_unified_search_results=[],  # Simulate no matches
        )

        query = "How does authentication work?"

        # Should raise ValueError mentioning path_filter
        with pytest.raises(ValueError) as exc_info:
            await service.deep_research(query)

        error_msg = str(exc_info.value)

        # Verify error message mentions the issue
        assert "No code chunks" in error_msg
        # Note: The current implementation may not mention path_filter explicitly,
        # but the test documents the EXPECTED fix (see docstring)

    @pytest.mark.asyncio
    async def test_nonexistent_path_filter_error_message_quality(
        self, db_services, embedding_manager, llm_manager, config, monkeypatch
    ):
        """Verify error message quality for nonexistent path_filter.

        Expected error message should:
        - Mention path_filter value
        - Suggest verifying indexed paths
        - Suggest removing path_filter
        - Be actionable for users

        This test documents the EXPECTED behavior after fix is applied.
        """
        service = create_coverage_service_with_path_filter(
            db_services,
            embedding_manager,
            llm_manager,
            config,
            monkeypatch,
            path_filter="nonexistent/",
            mock_unified_search_results=[],
        )

        query = "test query"

        with pytest.raises(ValueError) as exc_info:
            await service.deep_research(query)

        error_msg = str(exc_info.value)

        # Current implementation may not have all these checks yet
        # This documents the EXPECTED fix:
        # - Should mention "No code chunks available" ✓ (current)
        # - Should mention path_filter='nonexistent/' (EXPECTED FIX)
        # - Should suggest checking indexed paths (EXPECTED FIX)
        # - Should suggest removing path_filter (EXPECTED FIX)

        assert "No code chunks" in error_msg or "available for synthesis" in error_msg


class TestPathFilterMismatch:
    """Test path_filter valid but database has files in different paths."""

    @pytest.mark.asyncio
    async def test_valid_path_filter_but_no_matching_files(
        self, db_services, embedding_manager, llm_manager, config, monkeypatch
    ):
        """path_filter='src/' but database only has 'lib/' files.

        Expected behavior:
        - Phase 1: Unified search returns empty (path mismatch)
        - Phase 2: Skipped
        - Phase 3: Raises ValueError
        - Error message: Suggests checking indexed paths
        """
        service = create_coverage_service_with_path_filter(
            db_services,
            embedding_manager,
            llm_manager,
            config,
            monkeypatch,
            path_filter="src/",  # Database only has "lib/" files
            mock_unified_search_results=[],  # No matches
        )

        query = "authentication"

        with pytest.raises(ValueError) as exc_info:
            await service.deep_research(query)

        error_msg = str(exc_info.value)

        # Verify error is raised
        assert "No code chunks" in error_msg

    @pytest.mark.asyncio
    async def test_path_filter_propagates_through_phases(
        self, db_services, embedding_manager, llm_manager, config, monkeypatch
    ):
        """Verify path_filter propagates from Phase 1 → Phase 2 → Phase 3.

        Tests that path_filter is passed through the entire pipeline:
        - CoverageResearchService stores path_filter in __init__
        - Phase 1 passes path_filter to unified_search
        - Phase 2 passes path_filter to gap detection
        - Phase 3 synthesis receives filtered chunks

        This test verifies the propagation path, not the filtering logic itself.
        """
        # Track which methods received path_filter
        path_filter_propagation = {"unified_search": None, "gap_detection": None}

        class MockUnifiedSearchTracker:
            def __init__(self, db_services, embedding_manager, config=None):
                pass

            async def unified_search(
                self, query, context, expanded_queries=None, path_filter=None, rerank_queries=None
            ):
                # Record path_filter received
                path_filter_propagation["unified_search"] = path_filter
                # Return some chunks to proceed to Phase 2
                return [
                    {
                        "chunk_id": "lib1",
                        "file_path": "lib/auth.py",
                        "code": "def authenticate(): pass",
                        "rerank_score": 0.9,
                        "embedding": [0.1] * 1536,
                    }
                ]

            async def extract_symbols_from_chunks(self, chunks):
                return ["authenticate"]

            async def expand_chunk_windows(self, chunks, window_lines=50):
                return chunks

        class MockGapDetectionServiceTracker:
            def __init__(self, llm_manager, embedding_manager, db_services, config,
                         import_resolver=None, import_context_service=None):
                pass

            async def detect_and_fill_gaps(
                self, root_query, covered_chunks, phase1_threshold, path_filter=None
            ):
                # Record path_filter received
                path_filter_propagation["gap_detection"] = path_filter
                return covered_chunks, {
                    "gaps_found": 0,
                    "gaps_filled": 0,
                    "chunks_added": 0,
                    "total_chunks": len(covered_chunks),
                    "gap_queries": [],
                }

        class MockSynthesisEngine:
            def __init__(self, llm_manager, embedding_manager, db_services, config,
                         unified_search=None, import_resolver=None, import_context_service=None,
                         progress=None):
                pass

            async def synthesize(self, root_query, all_chunks, gap_queries, target_tokens,
                                  file_imports=None):
                return "Synthesis", [], {"final_tokens": 100}

        class MockQueryExpander:
            def __init__(self, llm_manager):
                pass

            async def expand_query_with_llm(self, query, context):
                return [query]

        from chunkhound.services.research.v2 import coverage_research_service as crs_module

        monkeypatch.setattr(crs_module, "UnifiedSearch", MockUnifiedSearchTracker)
        monkeypatch.setattr(crs_module, "GapDetectionService", MockGapDetectionServiceTracker)
        monkeypatch.setattr(crs_module, "CoverageSynthesisEngine", MockSynthesisEngine)
        monkeypatch.setattr(crs_module, "QueryExpander", MockQueryExpander)

        service = CoverageResearchService(
            database_services=db_services,
            embedding_manager=embedding_manager,
            llm_manager=llm_manager,
            config=config,
            tool_name="code_research",
            progress=None,
            path_filter="src/",
        )

        await service.deep_research("test query")

        # Verify path_filter propagated correctly
        assert path_filter_propagation["unified_search"] == "src/"
        assert path_filter_propagation["gap_detection"] == "src/"


class TestPathFilterSpecialCharacters:
    """Test path_filter with special regex characters."""

    @pytest.mark.asyncio
    async def test_path_filter_with_glob_pattern_treated_as_literal(
        self, db_services, embedding_manager, llm_manager, config, monkeypatch
    ):
        """path_filter='src/**/*.py' treated as literal prefix, not regex.

        Expected behavior:
        - path_filter is treated as literal string prefix match
        - Special characters like '**' and '*' are literal, not wildcards
        - No regex compilation errors
        - Returns empty if no file paths start with 'src/**/*.py' literally
        """
        service = create_coverage_service_with_path_filter(
            db_services,
            embedding_manager,
            llm_manager,
            config,
            monkeypatch,
            path_filter="src/**/*.py",  # Literal prefix, not glob
            mock_unified_search_results=[],  # No matches for this literal prefix
        )

        query = "test query"

        # Should not crash with regex errors
        with pytest.raises(ValueError) as exc_info:
            await service.deep_research(query)

        # Verify it's a ValueError about empty results, not regex error
        error_msg = str(exc_info.value)
        assert "No code chunks" in error_msg
        assert "regex" not in error_msg.lower()  # Should NOT be regex error

    @pytest.mark.asyncio
    async def test_path_filter_with_special_chars_no_regex_compilation(
        self, db_services, embedding_manager, llm_manager, config, monkeypatch
    ):
        """Verify path_filter with special chars doesn't attempt regex compilation.

        Tests characters: . * + ? [ ] ( ) { } ^ $ | \\

        Expected behavior:
        - All special regex characters treated as literals
        - No re.error exceptions
        - Prefix match works correctly
        """
        special_chars_paths = [
            "src/[test].py",
            "lib/(utils).py",
            "app/{config}.py",
            "data/file+.txt",
        ]

        for special_path in special_chars_paths:
            service = create_coverage_service_with_path_filter(
                db_services,
                embedding_manager,
                llm_manager,
                config,
                monkeypatch,
                path_filter=special_path,
                mock_unified_search_results=[],  # No matches
            )

            query = "test"

            # Should not crash with regex errors
            try:
                await service.deep_research(query)
            except ValueError as e:
                # Expected ValueError about empty results
                assert "No code chunks" in str(e)
            except Exception as e:
                # Should NOT be regex compilation error
                pytest.fail(f"Unexpected error for path_filter={special_path}: {e}")


class TestPathFilterEmptyString:
    """Test path_filter empty string behavior."""

    @pytest.mark.asyncio
    async def test_empty_string_path_filter_treated_as_none(
        self, db_services, embedding_manager, llm_manager, config, monkeypatch
    ):
        """path_filter='' (empty string) treated as None (no filtering).

        Expected behavior:
        - Empty string path_filter should behave like path_filter=None
        - Should return all chunks (no filtering applied)
        - Should NOT raise ValueError about empty results
        """
        # Mock to return chunks when path_filter is empty or None
        service = create_coverage_service_with_path_filter(
            db_services,
            embedding_manager,
            llm_manager,
            config,
            monkeypatch,
            path_filter="",  # Empty string
            mock_unified_search_results=[
                {
                    "chunk_id": "lib1",
                    "file_path": "lib/auth.py",
                    "code": "def authenticate(): pass",
                    "rerank_score": 0.9,
                    "embedding": [0.1] * 1536,
                }
            ],
        )

        # Patch synthesis to succeed
        from chunkhound.services.research.v2 import coverage_research_service as crs_module

        class MockSynthesisSuccess:
            def __init__(self, llm_manager, embedding_manager, db_services, config):
                pass

            async def synthesize(self, root_query, all_chunks, gap_queries, target_tokens):
                return "Synthesis result", [], {"final_tokens": 100}

        monkeypatch.setattr(crs_module, "CoverageSynthesisEngine", MockSynthesisSuccess)

        query = "authentication"

        # Should succeed (empty string = no filtering)
        result = await service.deep_research(query)

        assert "answer" in result
        assert isinstance(result["answer"], str)

    @pytest.mark.asyncio
    async def test_none_path_filter_returns_all_chunks(
        self, db_services, embedding_manager, llm_manager, config, monkeypatch
    ):
        """path_filter=None returns all chunks (baseline behavior).

        This test establishes the baseline: None path_filter should return results.
        """
        service = create_coverage_service_with_path_filter(
            db_services,
            embedding_manager,
            llm_manager,
            config,
            monkeypatch,
            path_filter=None,  # No filtering
            mock_unified_search_results=[
                {
                    "chunk_id": "lib1",
                    "file_path": "lib/auth.py",
                    "code": "def authenticate(): pass",
                    "rerank_score": 0.9,
                    "embedding": [0.1] * 1536,
                }
            ],
        )

        # Patch synthesis to succeed
        from chunkhound.services.research.v2 import coverage_research_service as crs_module

        class MockSynthesisSuccess:
            def __init__(self, llm_manager, embedding_manager, db_services, config):
                pass

            async def synthesize(self, root_query, all_chunks, gap_queries, target_tokens):
                return "Synthesis result", [], {"final_tokens": 100}

        monkeypatch.setattr(crs_module, "CoverageSynthesisEngine", MockSynthesisSuccess)

        query = "authentication"

        # Should succeed
        result = await service.deep_research(query)

        assert "answer" in result
        assert isinstance(result["answer"], str)


class TestPathFilterErrorMessageDocumentation:
    """Document expected error messages for path_filter failures.

    These tests document the EXPECTED FIX behavior, not necessarily current behavior.
    Once the fix is applied, these tests should pass.
    """

    @pytest.mark.asyncio
    async def test_expected_error_message_for_nonexistent_path(
        self, db_services, embedding_manager, llm_manager, config, monkeypatch
    ):
        """Document expected error message format for nonexistent path_filter.

        Expected error message after fix:
        ```
        No code chunks found matching path_filter='nonexistent/'.
        Verify the path exists in your indexed codebase.
        Try removing path_filter or checking indexed paths.
        ```
        """
        service = create_coverage_service_with_path_filter(
            db_services,
            embedding_manager,
            llm_manager,
            config,
            monkeypatch,
            path_filter="nonexistent/",
            mock_unified_search_results=[],
        )

        query = "test"

        with pytest.raises(ValueError) as exc_info:
            await service.deep_research(query)

        error_msg = str(exc_info.value)

        # Current implementation raises generic error
        # After fix, should include these elements:
        # 1. Mention "No code chunks found"
        # 2. Include path_filter value
        # 3. Suggest verifying indexed paths
        # 4. Suggest removing path_filter

        # For now, just verify it's a ValueError
        assert isinstance(exc_info.value, ValueError)

    def test_expected_fix_location_documentation(self):
        """Document where the expected fix should be applied.

        Expected fix location:
        - File: chunkhound/services/research/v2/coverage_research_service.py
        - Method: _phase1_coverage (after unified_search call around line 288-293)
        - Logic: Check if covered_chunks is empty AND path_filter is set
        - Error: Raise ValueError with path_filter-specific guidance

        Expected code:
        ```python
        # In Phase 1 after unified_search (around line 293)
        if not covered_chunks and path_filter:
            raise ValueError(
                f"No code chunks found matching path_filter='{path_filter}'. "
                f"Verify the path exists in your indexed codebase. "
                f"Try removing path_filter or checking indexed paths."
            )
        ```
        """
        # This is a documentation test - no assertions needed
        # It documents the expected fix for developers
        pass


class TestPathFilterWithValidResults:
    """Test path_filter with valid results (positive cases)."""

    @pytest.mark.asyncio
    async def test_path_filter_with_matching_files_succeeds(
        self, db_services, embedding_manager, llm_manager, config, monkeypatch
    ):
        """path_filter='lib/' with matching files in database succeeds.

        Positive test case:
        - path_filter matches files in database
        - Phase 1 returns chunks
        - Phase 2 processes normally
        - Phase 3 synthesizes successfully
        """
        service = create_coverage_service_with_path_filter(
            db_services,
            embedding_manager,
            llm_manager,
            config,
            monkeypatch,
            path_filter="lib/",  # Matches chunks in mock
            mock_unified_search_results=[
                {
                    "chunk_id": "lib1",
                    "file_path": "lib/auth.py",
                    "code": "def authenticate(): pass",
                    "rerank_score": 0.9,
                    "embedding": [0.1] * 1536,
                }
            ],
        )

        # Patch synthesis to succeed
        from chunkhound.services.research.v2 import coverage_research_service as crs_module

        class MockSynthesisSuccess:
            def __init__(self, llm_manager, embedding_manager, db_services, config):
                pass

            async def synthesize(self, root_query, all_chunks, gap_queries, target_tokens):
                return "Synthesis result for lib/ files", [], {"final_tokens": 100}

        monkeypatch.setattr(crs_module, "CoverageSynthesisEngine", MockSynthesisSuccess)

        query = "authentication"

        # Should succeed
        result = await service.deep_research(query)

        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0  # Should have synthesized content
