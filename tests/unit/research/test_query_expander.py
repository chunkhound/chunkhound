"""Unit tests for QueryExpander in deep research.

Tests query building strategies and LLM-based query expansion
without requiring real API calls.
"""

import pytest

from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.models import ResearchContext
from chunkhound.services.research.query_expander import QueryExpander
from tests.fixtures.fake_providers import FakeLLMProvider


@pytest.fixture
def fake_llm_provider():
    """Create fake LLM provider with scripted responses."""
    return FakeLLMProvider(
        responses={
            # Match any query expansion request (searches for words in the prompt)
            "query": '{"queries": ["authentication system implementation", "user auth flow design"]}',
            "search": '{"queries": ["code search variation 1", "code search variation 2"]}',
            "authentication": '{"queries": ["auth system architecture", "authentication workflow"]}',
            "fallback": "default response",
        }
    )


@pytest.fixture
def llm_manager(fake_llm_provider, monkeypatch):
    """Create LLM manager with fake provider.

    Monkeypatch the provider creation to return our fake provider
    instead of initializing real providers.
    """

    # Mock the _create_provider method to return our fake provider
    def mock_create_provider(self, config):
        return fake_llm_provider

    monkeypatch.setattr(LLMManager, "_create_provider", mock_create_provider)

    # Create manager with minimal config
    utility_config = {"provider": "fake", "model": "fake-gpt"}
    synthesis_config = {"provider": "fake", "model": "fake-gpt"}

    manager = LLMManager(utility_config, synthesis_config)
    return manager


@pytest.fixture
def query_expander(llm_manager):
    """Create query expander with fake LLM."""
    return QueryExpander(llm_manager)


class TestBuildSearchQuery:
    """Test query building strategies for semantic search."""

    def test_root_node_uses_query_directly(self, query_expander):
        """Root nodes should use the original query without modification."""
        query = "How does authentication work?"
        context = ResearchContext(root_query=query, ancestors=[])

        result = query_expander.build_search_query(query, context)

        assert result == query, "Root node should return query unchanged"

    def test_child_node_with_single_ancestor(self, query_expander):
        """Child nodes should combine query with minimal parent context."""
        query = "token validation logic"
        root = "How does authentication work?"
        context = ResearchContext(root_query=root, ancestors=["authentication flow"])

        result = query_expander.build_search_query(query, context)

        assert query in result, "Result should contain original query"
        assert "authentication flow" in result, "Result should contain ancestor context"
        assert result.startswith(query), "Query should come first (position bias)"
        assert " | Context: " in result, "Should have clear separator"

    def test_child_node_with_multiple_ancestors(self, query_expander):
        """Child nodes should use last 2 ancestors for context."""
        query = "session storage implementation"
        root = "How does authentication work?"
        context = ResearchContext(
            root_query=root,
            ancestors=[
                "authentication flow",
                "token validation logic",
                "session management",
            ],
        )

        result = query_expander.build_search_query(query, context)

        assert query in result, "Result should contain original query"
        assert result.startswith(query), "Query should come first (position bias)"
        # Should use last 2 ancestors only
        assert "token validation logic" in result
        assert "session management" in result
        # Should NOT include first ancestor (too far back)
        assert "authentication flow" not in result

    def test_query_first_position_for_embedding_bias(self, query_expander):
        """Query must come before context for embedding model position bias."""
        query = "database query"
        context = ResearchContext(root_query="root", ancestors=["parent1", "parent2"])

        result = query_expander.build_search_query(query, context)

        # Verify query is at the very beginning
        assert result.index(query) == 0, "Query must be first for position bias"
        assert result.startswith(query), "Query must start the search string"


class TestExpandQueryWithLLM:
    """Test LLM-based query expansion."""

    @pytest.mark.asyncio
    async def test_successful_query_expansion(self, query_expander):
        """Should expand query into multiple variations using LLM."""
        query = "authentication implementation"
        context = ResearchContext(root_query=query, ancestors=[])

        # FakeLLMProvider will return scripted response for "expand" pattern
        result = await query_expander.expand_query_with_llm(query, context)

        assert len(result) == 3, "Should return original + 2 expanded queries"
        assert result[0] == query, "Original query should be first"
        assert len(result[1]) > 0, "Expanded queries should not be empty"
        assert len(result[2]) > 0, "Expanded queries should not be empty"

    @pytest.mark.asyncio
    async def test_expansion_with_context(self, query_expander):
        """Should include context in expansion prompt."""
        query = "token storage"
        context = ResearchContext(
            root_query="authentication system",
            ancestors=["auth flow", "token validation"],
        )

        result = await query_expander.expand_query_with_llm(query, context)

        # Should still work with context
        assert len(result) >= 1, "Should at least return original query"
        assert result[0] == query, "Original query should be first"

    @pytest.mark.asyncio
    async def test_expansion_failure_returns_original(self, monkeypatch):
        """Should gracefully fallback to original query on LLM failure."""
        # Create query expander with failing LLM
        failing_provider = FakeLLMProvider(responses={})

        def mock_create_provider(self, config):
            return failing_provider

        monkeypatch.setattr(LLMManager, "_create_provider", mock_create_provider)

        utility_config = {"provider": "fake", "model": "fake-gpt"}
        synthesis_config = {"provider": "fake", "model": "fake-gpt"}
        failing_manager = LLMManager(utility_config, synthesis_config)
        failing_expander = QueryExpander(failing_manager)

        query = "test query"
        context = ResearchContext(root_query=query, ancestors=[])

        result = await failing_expander.expand_query_with_llm(query, context)

        assert result == [query], "Should return original query on failure"

    @pytest.mark.asyncio
    async def test_empty_expanded_queries_returns_original(self, monkeypatch):
        """Should handle empty expansion results gracefully."""
        # Create provider that returns empty queries
        empty_provider = FakeLLMProvider(responses={"expand": '{"queries": []}'})

        def mock_create_provider(self, config):
            return empty_provider

        monkeypatch.setattr(LLMManager, "_create_provider", mock_create_provider)

        utility_config = {"provider": "fake", "model": "fake-gpt"}
        synthesis_config = {"provider": "fake", "model": "fake-gpt"}
        empty_manager = LLMManager(utility_config, synthesis_config)
        empty_expander = QueryExpander(empty_manager)

        query = "test query"
        context = ResearchContext(root_query=query, ancestors=[])

        result = await empty_expander.expand_query_with_llm(query, context)

        assert result == [query], "Should return original query when expansion is empty"

    @pytest.mark.asyncio
    async def test_original_query_always_first(self, query_expander):
        """Original query must always be first for position bias."""
        query = "original query"
        context = ResearchContext(root_query=query, ancestors=[])

        result = await query_expander.expand_query_with_llm(query, context)

        assert result[0] == query, "Original query MUST be first for position bias"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_ancestors_list(self, query_expander):
        """Empty ancestors should work like root node."""
        query = "test query"
        context = ResearchContext(root_query=query, ancestors=[])

        result = query_expander.build_search_query(query, context)

        assert result == query, "Empty ancestors should return query unchanged"

    def test_single_ancestor_uses_it(self, query_expander):
        """Single ancestor should be included in context."""
        query = "child query"
        context = ResearchContext(root_query="root", ancestors=["parent"])

        result = query_expander.build_search_query(query, context)

        assert "parent" in result, "Single ancestor should be in context"

    def test_whitespace_handling(self, query_expander):
        """Should handle queries with extra whitespace."""
        query = "  query with spaces  "
        context = ResearchContext(root_query=query.strip(), ancestors=[])

        result = query_expander.build_search_query(query, context)

        assert query in result, "Should preserve query including whitespace"

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, query_expander):
        """Should handle special characters in queries."""
        query = "auth && validation || tokens"
        context = ResearchContext(root_query=query, ancestors=[])

        result = await query_expander.expand_query_with_llm(query, context)

        assert len(result) >= 1, "Should handle special characters"
        assert result[0] == query, "Original query should be preserved"
