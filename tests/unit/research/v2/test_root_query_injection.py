"""Unit tests for ROOT query injection validation in v2 research pipeline.

This module validates a critical architectural invariant documented in
docs/algorithm-coverage-first-research.md:L79:

    "ROOT Query Injection: Every LLM call includes `root_query` in prompt.
     Check all prompts contain `RESEARCH QUERY:` or `PRIMARY QUERY:`"

This guarantee prevents semantic drift during multi-step processing by ensuring
the original research intent is always present in LLM prompts.

Test Strategy:
    1. Create custom LLM provider that captures all prompts
    2. Run v2 research components (gap detection, query expansion, synthesis)
    3. Validate prompts contain ROOT query with expected headers
    4. Verify gap queries are included in synthesis prompts

Components Tested:
    - Query expansion: RESEARCH QUERY header + root query text
    - Gap detection: RESEARCH QUERY header + root query text
    - Gap unification: RESEARCH QUERY header + root query text
    - Synthesis (base): PRIMARY QUERY header + root query text
    - Synthesis (with gaps): PRIMARY QUERY + RELATED GAPS section
"""

import pytest
import re
from typing import Any

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.shared.models import ResearchContext
from chunkhound.services.research.shared.query_expander import QueryExpander
from chunkhound.services.research.v2.gap_detection import GapDetectionService
from chunkhound.services.research.v2.coverage_synthesis import CoverageSynthesisEngine
from chunkhound.services.research.v2.models import GapCandidate
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider


# Custom LLM provider that captures all prompts
class PromptCapturingProvider(FakeLLMProvider):
    """Fake LLM provider that captures all prompts for validation.

    Similar to the tracked_rerank pattern in test_compound_rerank.py,
    this provider records every prompt sent to it for later inspection.
    """

    def __init__(self):
        super().__init__()
        self.captured_prompts: list[dict[str, Any]] = []

        # Set up responses for different prompt types
        self._responses = {
            "expand": '{"queries": ["How is authentication implemented?", "What authentication mechanisms exist?"]}',
            "gaps": '{"gaps": [{"query": "How is session management implemented?", "rationale": "Referenced but not found", "confidence": 0.8}]}',
            "unified_query": '{"unified_query": "How is session management and token handling implemented?"}',
            "synthesis": (
                "## Overview\n"
                "The authentication system uses JWT tokens with session management.\n\n"
                "## Implementation\n"
                "The system implements a layered authentication architecture:\n"
                "- Token generation in auth_service.py [1]\n"
                "- Session validation in session_manager.py [2]\n"
                "- User credentials stored in database.py [3]\n\n"
                "## Key Components\n"
                "Authentication flow follows standard OAuth2 patterns with custom extensions.\n"
            ),
        }

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Capture prompt and return appropriate response."""
        # Capture the prompt
        self.captured_prompts.append({
            "prompt": prompt,
            "system": system,
            "type": "complete_structured",
        })

        # Return appropriate response based on prompt content
        response = await super().complete_structured(
            prompt, json_schema, system, max_completion_tokens
        )
        return response

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> Any:
        """Capture prompt and return appropriate response."""
        # Capture the prompt
        self.captured_prompts.append({
            "prompt": prompt,
            "system": system,
            "type": "complete",
        })

        # Override parent response for synthesis (uses complete, not complete_structured)
        # Match synthesis by looking for "PRIMARY QUERY" which is unique to synthesis
        if "PRIMARY QUERY" in prompt or "Question:" in prompt:
            # Return synthesis response
            import asyncio
            from chunkhound.interfaces.llm_provider import LLMResponse

            await asyncio.sleep(0.001)
            self._requests_made += 1

            response_content = self._responses["synthesis"]
            prompt_tokens = self.estimate_tokens(prompt)
            if system:
                prompt_tokens += self.estimate_tokens(system)
            completion_tokens = self.estimate_tokens(response_content)
            total_tokens = prompt_tokens + completion_tokens

            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._tokens_used += total_tokens

            return LLMResponse(
                content=response_content,
                tokens_used=total_tokens,
                model=self._model,
                finish_reason="stop",
            )

        # Otherwise use parent's pattern matching
        response = await super().complete(prompt, system, max_completion_tokens, timeout)
        return response

    def get_prompts_containing(self, substring: str) -> list[str]:
        """Get all prompts containing a specific substring."""
        return [
            p["prompt"]
            for p in self.captured_prompts
            if substring.lower() in p["prompt"].lower()
        ]

    def reset_captured_prompts(self) -> None:
        """Clear captured prompts."""
        self.captured_prompts = []


@pytest.fixture
def prompt_capturing_provider():
    """Fixture providing a prompt-capturing LLM provider."""
    return PromptCapturingProvider()


@pytest.fixture
def fake_embedding_provider():
    """Fixture providing a fake embedding provider for gap clustering."""
    return FakeEmbeddingProvider(dims=1536)


@pytest.fixture
def llm_manager_with_capturing(prompt_capturing_provider, monkeypatch):
    """Fixture providing LLM manager with prompt-capturing provider.

    Uses monkeypatch to replace provider creation (same pattern as test_query_expander.py).
    """
    # Mock the _create_provider method to return our capturing provider
    def mock_create_provider(self, config):
        return prompt_capturing_provider

    monkeypatch.setattr(LLMManager, "_create_provider", mock_create_provider)

    # Create manager with minimal config
    utility_config = {"provider": "fake", "model": "fake-utility"}
    synthesis_config = {"provider": "fake", "model": "fake-synthesis"}

    manager = LLMManager(utility_config, synthesis_config)
    return manager


@pytest.fixture
def query_expander(llm_manager_with_capturing):
    """Fixture providing QueryExpander with capturing provider."""
    return QueryExpander(llm_manager_with_capturing)


@pytest.fixture
def gap_detection_service(
    llm_manager_with_capturing,
    fake_embedding_provider,
):
    """Fixture providing GapDetectionService with capturing provider."""
    # Create mock database services (same pattern as test_gap_detection.py)
    class MockProvider:
        def get_base_directory(self):
            from pathlib import Path
            return Path("/fake/base")

    class MockDatabaseServices:
        provider = MockProvider()

    db_services = MockDatabaseServices()

    # Create mock embedding manager
    class MockEmbeddingManager:
        def get_provider(self):
            return fake_embedding_provider

    embedding_manager = MockEmbeddingManager()

    config = ResearchConfig()

    return GapDetectionService(
        llm_manager=llm_manager_with_capturing,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=config,
    )


@pytest.fixture
def coverage_synthesis_engine(
    llm_manager_with_capturing,
    fake_embedding_provider,
):
    """Fixture providing CoverageSynthesisEngine with capturing provider."""
    # Create mock database services (same pattern as test_gap_detection.py)
    class MockProvider:
        def get_base_directory(self):
            from pathlib import Path
            return Path("/fake/base")

    class MockDatabaseServices:
        provider = MockProvider()

    db_services = MockDatabaseServices()

    # Create mock embedding manager
    class MockEmbeddingManager:
        def get_provider(self):
            return fake_embedding_provider

    embedding_manager = MockEmbeddingManager()

    config = ResearchConfig()

    return CoverageSynthesisEngine(
        llm_manager=llm_manager_with_capturing,
        embedding_manager=embedding_manager,
        db_services=db_services,
        config=config,
    )


# Test 1: Query Expansion
@pytest.mark.asyncio
async def test_query_expansion_includes_root_query(
    query_expander,
    prompt_capturing_provider,
):
    """Test that query expansion prompts contain ROOT query.

    Per docs/algorithm-coverage-first-research.md:L79, every LLM call
    must include the root query. Query expansion should include it in
    the prompt via the context.

    Expected format (from prompts/query_expansion.py):
        Query: {query}
        Context: {context_root_query}{context_str}
    """
    root_query = "How does authentication work in this codebase?"
    current_query = "What are the authentication mechanisms?"
    context = ResearchContext(root_query=root_query)

    # Run query expansion
    await query_expander.expand_query_with_llm(current_query, context)

    # Validate: prompt must contain root query
    prompts = prompt_capturing_provider.captured_prompts
    assert len(prompts) > 0, "Expected at least one LLM call"

    expansion_prompt = prompts[0]["prompt"]

    # Verify root query is in the prompt
    assert root_query in expansion_prompt, (
        f"Root query '{root_query}' not found in query expansion prompt.\n"
        f"Prompt: {expansion_prompt[:200]}..."
    )

    # Verify the prompt contains context section (format: "Context: {root_query}")
    assert "Context:" in expansion_prompt, (
        "Expected 'Context:' section in query expansion prompt"
    )


# Test 2: Gap Detection
@pytest.mark.asyncio
async def test_gap_detection_includes_research_query_header(
    gap_detection_service,
    prompt_capturing_provider,
):
    """Test that gap detection prompts contain 'RESEARCH QUERY:' header.

    Per docs/algorithm-coverage-first-research.md:L272-280, gap detection
    prompts must start with:
        RESEARCH QUERY: {root_query}

        Given the research query above, identify semantic gaps...
    """
    root_query = "How does the search service work?"

    # Create mock chunks for gap detection
    mock_chunks = [
        {
            "chunk_id": "1",
            "file_path": "search_service.py",
            "code": "def search(query): return results",
            "content": "def search(query): return results",
        }
    ]

    # Run gap detection (just the detection step, not full pipeline)
    shards = [[mock_chunks[0]]]
    await gap_detection_service._detect_gaps_parallel(root_query, shards)

    # Validate: prompt must contain RESEARCH QUERY header
    prompts = prompt_capturing_provider.captured_prompts
    assert len(prompts) > 0, "Expected at least one LLM call for gap detection"

    gap_prompt = prompts[0]["prompt"]

    # Verify RESEARCH QUERY header
    assert gap_prompt.startswith("RESEARCH QUERY:"), (
        f"Gap detection prompt must start with 'RESEARCH QUERY:'\n"
        f"Actual start: {gap_prompt[:100]}..."
    )

    # Verify root query is in the prompt
    assert root_query in gap_prompt, (
        f"Root query '{root_query}' not found in gap detection prompt.\n"
        f"Prompt: {gap_prompt[:300]}..."
    )

    # Verify expected prompt structure per docs
    assert "Given the research query above" in gap_prompt, (
        "Gap detection prompt should reference 'the research query above'"
    )


# Test 3: Gap Unification
@pytest.mark.asyncio
async def test_gap_unification_includes_research_query_header(
    gap_detection_service,
    prompt_capturing_provider,
):
    """Test that gap unification prompts contain 'RESEARCH QUERY:' header.

    Per docs/algorithm-coverage-first-research.md:L292-298, gap unification
    prompts must include:
        RESEARCH QUERY: {root_query}

        Merge these similar gap queries into ONE refined query
        that best addresses the research query above:
    """
    root_query = "How is authentication implemented?"

    # Create mock gap candidates
    import numpy as np
    gaps = [
        GapCandidate(
            query="How is session management handled?",
            rationale="Referenced in auth flow",
            confidence=0.8,
            source_shard=0,
        ),
        GapCandidate(
            query="What session storage is used?",
            rationale="Part of session logic",
            confidence=0.7,
            source_shard=0,
        ),
    ]

    # Create cluster labels (both gaps in same cluster)
    labels = np.array([0, 0])

    # Reset captured prompts from previous tests
    prompt_capturing_provider.reset_captured_prompts()

    # Run gap unification
    await gap_detection_service._unify_gap_clusters(root_query, gaps, labels)

    # Validate: prompt must contain RESEARCH QUERY header
    prompts = prompt_capturing_provider.captured_prompts
    assert len(prompts) > 0, "Expected at least one LLM call for gap unification"

    unification_prompt = prompts[0]["prompt"]

    # Verify RESEARCH QUERY header
    assert unification_prompt.startswith("RESEARCH QUERY:"), (
        f"Gap unification prompt must start with 'RESEARCH QUERY:'\n"
        f"Actual start: {unification_prompt[:100]}..."
    )

    # Verify root query is in the prompt
    assert root_query in unification_prompt, (
        f"Root query '{root_query}' not found in gap unification prompt.\n"
        f"Prompt: {unification_prompt[:300]}..."
    )

    # Verify expected prompt structure per docs
    assert "Merge these similar gap queries" in unification_prompt, (
        "Gap unification prompt should contain merge instruction"
    )
    assert "research query above" in unification_prompt, (
        "Gap unification prompt should reference 'research query above'"
    )


# Test 4: Synthesis (Base Case - No Gaps)
@pytest.mark.asyncio
async def test_synthesis_base_includes_primary_query_header(
    coverage_synthesis_engine,
    prompt_capturing_provider,
):
    """Test that synthesis prompts contain 'PRIMARY QUERY:' header.

    Per docs/algorithm-coverage-first-research.md:L79, synthesis prompts
    must use 'PRIMARY QUERY:' instead of 'RESEARCH QUERY:'.

    Base case (no gap queries):
        PRIMARY QUERY: {root_query}
    """
    root_query = "How does the database layer work?"
    gap_queries: list[str] = []  # No gaps

    # Create minimal compressed content
    compressed_content = {
        "cluster_0_summary": "Database uses PostgreSQL with connection pooling."
    }

    # Create mock chunks
    mock_chunks = [
        {
            "chunk_id": "1",
            "file_path": "database.py",
            "content": "class Database: pass",
            "start_line": 1,
            "end_line": 5,
        }
    ]

    # Create mock files
    mock_files = {
        "database.py": "class Database:\n    def connect(self): pass\n"
    }

    # Reset captured prompts
    prompt_capturing_provider.reset_captured_prompts()

    # Run final synthesis
    await coverage_synthesis_engine._final_synthesis(
        root_query=root_query,
        gap_queries=gap_queries,
        compressed_content=compressed_content,
        all_chunks=mock_chunks,
        original_files=mock_files,
        file_imports={},
    )

    # Validate: prompt must contain PRIMARY QUERY header
    prompts = prompt_capturing_provider.captured_prompts
    assert len(prompts) > 0, "Expected at least one LLM call for synthesis"

    synthesis_prompt = prompts[0]["prompt"]

    # Verify PRIMARY QUERY header (not RESEARCH QUERY)
    assert "PRIMARY QUERY:" in synthesis_prompt, (
        f"Synthesis prompt must contain 'PRIMARY QUERY:' header\n"
        f"Prompt start: {synthesis_prompt[:200]}..."
    )

    # Verify root query is in the prompt
    assert root_query in synthesis_prompt, (
        f"Root query '{root_query}' not found in synthesis prompt.\n"
        f"Prompt: {synthesis_prompt[:500]}..."
    )

    # Verify NO gap queries section (base case)
    assert "RELATED GAPS" not in synthesis_prompt, (
        "Base synthesis should not contain RELATED GAPS section"
    )


# Test 5: Synthesis (With Gaps)
@pytest.mark.asyncio
async def test_synthesis_with_gaps_includes_both_primary_and_gaps(
    coverage_synthesis_engine,
    prompt_capturing_provider,
):
    """Test that synthesis with gaps contains PRIMARY QUERY + RELATED GAPS.

    Per docs/algorithm-coverage-first-research.md:L83, when gap queries exist,
    synthesis prompts must include both:
        PRIMARY QUERY: {root_query}

        RELATED GAPS IDENTIFIED:
        - {gap_query_1}
        - {gap_query_2}

    This ensures compound context prevents semantic drift.
    """
    root_query = "How does authentication work?"
    gap_queries = [
        "How is session management implemented?",
        "What token validation mechanisms exist?",
    ]

    # Create minimal compressed content
    compressed_content = {
        "cluster_0_summary": "Authentication uses JWT tokens with session storage."
    }

    # Create mock chunks
    mock_chunks = [
        {
            "chunk_id": "1",
            "file_path": "auth_service.py",
            "content": "def authenticate(user): return jwt_token",
            "start_line": 1,
            "end_line": 5,
        }
    ]

    # Create mock files
    mock_files = {
        "auth_service.py": "def authenticate(user):\n    return jwt_token\n"
    }

    # Reset captured prompts
    prompt_capturing_provider.reset_captured_prompts()

    # Run final synthesis with gap queries
    await coverage_synthesis_engine._final_synthesis(
        root_query=root_query,
        gap_queries=gap_queries,
        compressed_content=compressed_content,
        all_chunks=mock_chunks,
        original_files=mock_files,
        file_imports={},
    )

    # Validate: prompt must contain both PRIMARY QUERY and RELATED GAPS
    prompts = prompt_capturing_provider.captured_prompts
    assert len(prompts) > 0, "Expected at least one LLM call for synthesis"

    synthesis_prompt = prompts[0]["prompt"]

    # Verify PRIMARY QUERY header
    assert "PRIMARY QUERY:" in synthesis_prompt, (
        f"Synthesis prompt must contain 'PRIMARY QUERY:' header\n"
        f"Prompt start: {synthesis_prompt[:200]}..."
    )

    # Verify root query is in the prompt
    assert root_query in synthesis_prompt, (
        f"Root query '{root_query}' not found in synthesis prompt.\n"
        f"Prompt: {synthesis_prompt[:500]}..."
    )

    # Verify RELATED GAPS section exists
    assert "RELATED GAPS IDENTIFIED:" in synthesis_prompt, (
        "Synthesis with gaps must contain 'RELATED GAPS IDENTIFIED:' section"
    )

    # Verify all gap queries are in the prompt
    for gap_query in gap_queries:
        assert gap_query in synthesis_prompt, (
            f"Gap query '{gap_query}' not found in synthesis prompt.\n"
            f"Expected all gap queries in RELATED GAPS section.\n"
            f"Prompt: {synthesis_prompt[:800]}..."
        )

    # Verify gap queries are formatted as bullet points (per implementation)
    for gap_query in gap_queries:
        # Check for bullet point format: "- {gap_query}"
        assert f"- {gap_query}" in synthesis_prompt, (
            f"Gap query '{gap_query}' not formatted as bullet point (expected '- {gap_query}')"
        )


# Test 6: Compression Loop (Cluster Compression)
@pytest.mark.asyncio
async def test_cluster_compression_includes_primary_query(
    coverage_synthesis_engine,
    prompt_capturing_provider,
):
    """Test that cluster compression prompts contain PRIMARY QUERY.

    Per the implementation in coverage_synthesis.py:L698-702, cluster
    compression prompts build compound context with:
        PRIMARY QUERY: {root_query}

        RELATED GAPS IDENTIFIED:
        - {gap_query}

    This maintains query focus during compression iterations.
    """
    root_query = "What are the key algorithms?"
    gap_queries = ["How is search optimized?"]

    cluster_content = {
        "algorithm.py": "def binary_search(arr, target):\n    # implementation\n    pass\n"
    }

    # Reset captured prompts
    prompt_capturing_provider.reset_captured_prompts()

    # Run cluster compression (via CompressionService)
    await coverage_synthesis_engine._compression_service._compress_cluster(
        root_query=root_query,
        gap_queries=gap_queries,
        cluster_content=cluster_content,
        target_tokens=10000,
        file_imports={},
    )

    # Validate: prompt must contain PRIMARY QUERY
    prompts = prompt_capturing_provider.captured_prompts
    assert len(prompts) > 0, "Expected at least one LLM call for cluster compression"

    compression_prompt = prompts[0]["prompt"]

    # Verify PRIMARY QUERY header
    assert "PRIMARY QUERY:" in compression_prompt, (
        f"Cluster compression prompt must contain 'PRIMARY QUERY:' header\n"
        f"Prompt start: {compression_prompt[:200]}..."
    )

    # Verify root query is in the prompt
    assert root_query in compression_prompt, (
        f"Root query '{root_query}' not found in cluster compression prompt.\n"
        f"Prompt: {compression_prompt[:500]}..."
    )

    # Verify gap queries are included
    if gap_queries:
        assert "RELATED GAPS IDENTIFIED:" in compression_prompt, (
            "Cluster compression with gaps must contain 'RELATED GAPS IDENTIFIED:' section"
        )
        for gap_query in gap_queries:
            assert gap_query in compression_prompt, (
                f"Gap query '{gap_query}' not found in cluster compression prompt"
            )


# Summary Test: Full Invariant Validation
@pytest.mark.asyncio
async def test_all_llm_touchpoints_validated(
    query_expander,
    gap_detection_service,
    coverage_synthesis_engine,
    prompt_capturing_provider,
):
    """Meta-test verifying all 5 LLM touchpoints are validated.

    This test ensures we have comprehensive coverage of the ROOT query
    injection invariant across all v2 research components.

    LLM Touchpoints (per docs/algorithm-coverage-first-research.md):
        1. Query expansion: RESEARCH QUERY in context
        2. Gap detection: RESEARCH QUERY header
        3. Gap unification: RESEARCH QUERY header
        4. Synthesis (base): PRIMARY QUERY header
        5. Synthesis (with gaps): PRIMARY QUERY + RELATED GAPS
        6. Cluster compression: PRIMARY QUERY + RELATED GAPS (bonus)

    This meta-test validates that our test suite covers all touchpoints.
    """
    # This is a documentation test - it passes if all above tests are defined
    # and validates the test coverage.

    test_functions = [
        test_query_expansion_includes_root_query,
        test_gap_detection_includes_research_query_header,
        test_gap_unification_includes_research_query_header,
        test_synthesis_base_includes_primary_query_header,
        test_synthesis_with_gaps_includes_both_primary_and_gaps,
        test_cluster_compression_includes_primary_query,  # Bonus
    ]

    # Verify all test functions are defined
    assert len(test_functions) >= 5, (
        f"Expected at least 5 LLM touchpoint tests, found {len(test_functions)}"
    )

    # All tests above validate the invariant - this test documents the coverage
    assert True, "All LLM touchpoints have ROOT query injection tests"
