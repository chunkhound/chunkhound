"""Test that semantic search correctly detects configured embedding provider.

This test verifies the bug where provider/model parameters with hardcoded
defaults prevent automatic detection of the configured embedding provider.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from chunkhound.mcp_server.tools import search_impl
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager


@pytest.mark.asyncio
async def test_semantic_search_uses_configured_provider():
    """Verify semantic search uses configured provider when not explicitly specified.

    This test catches the regression where provider/model defaults changed from
    None to hardcoded strings, breaking automatic provider detection.
    """
    # Setup: Create a mock embedding provider with non-default configuration
    mock_provider = Mock()
    mock_provider.name = "custom-provider"  # NOT "openai"
    mock_provider.model = "nomic-embed-text"  # NOT "text-embedding-3-small"

    # Create mock embedding manager
    mock_embedding_manager = Mock(spec=EmbeddingManager)
    mock_embedding_manager.list_providers.return_value = ["custom-provider"]
    mock_embedding_manager.get_provider.return_value = mock_provider

    # Create mock database services
    mock_services = Mock(spec=DatabaseServices)
    mock_provider_db = Mock()
    mock_provider_db.is_connected = True
    mock_services.provider = mock_provider_db

    # Create mock search service with async search method
    mock_search_service = Mock()
    mock_search_service.search_semantic = AsyncMock(return_value=([], {"total": 0, "page": 1, "page_size": 10}))
    mock_services.search_service = mock_search_service

    # Execute: Call search_impl with type="semantic" WITHOUT specifying provider/model
    # This should use the configured provider (ollama/nomic-embed-text)
    await search_impl(
        services=mock_services,
        embedding_manager=mock_embedding_manager,
        type="semantic",
        query="test query",
    )

    # Verify: Check that search was called with the CONFIGURED provider, not hardcoded defaults
    mock_search_service.search_semantic.assert_called_once()
    call_kwargs = mock_search_service.search_semantic.call_args.kwargs

    # CRITICAL ASSERTION: Should use configured provider, not hardcoded "openai"
    assert call_kwargs["provider"] == "custom-provider", \
        f"Expected provider='custom-provider' but got provider='{call_kwargs['provider']}'. " \
        "Semantic search should use configured provider when not explicitly specified."

    # CRITICAL ASSERTION: Should use configured model, not hardcoded "text-embedding-3-small"
    assert call_kwargs["model"] == "nomic-embed-text", \
        f"Expected model='nomic-embed-text' but got model='{call_kwargs['model']}'. " \
        "Semantic search should use configured model when not explicitly specified."


def _strategy_fixture():
    """Services and embedding manager for a reranking (Voyage) provider."""
    provider = Mock()
    provider.name, provider.model = "voyageai", "voyage-code-4"
    manager = Mock(spec=EmbeddingManager)
    manager.list_providers.return_value = ["voyageai"]
    manager.get_provider.return_value = provider
    services = Mock(spec=DatabaseServices)
    services.search_service = Mock()
    page = {"total": 0, "page": 1, "page_size": 10}
    services.search_service.search_semantic = AsyncMock(return_value=([], page))
    services.search_service.search_regex_async = AsyncMock(return_value=([], page))
    return services, manager


def test_search_schema_exposes_two_strategies():
    from chunkhound.mcp_server.tools import TOOL_REGISTRY

    params = TOOL_REGISTRY["search"].parameters
    assert params["properties"]["strategy"]["enum"] == ["single_hop", "multi_hop"]
    assert "strategy" not in params["required"]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["single_hop", "multi_hop"])
async def test_semantic_search_forwards_strategy(strategy):
    services, manager = _strategy_fixture()

    await search_impl(
        services=services,
        embedding_manager=manager,
        type="semantic",
        query="q",
        strategy=strategy,
    )

    call = services.search_service.search_semantic.await_args
    assert call.kwargs["force_strategy"] == strategy


@pytest.mark.asyncio
async def test_semantic_search_without_strategy_lets_the_service_choose():
    services, manager = _strategy_fixture()

    await search_impl(
        services=services, embedding_manager=manager, type="semantic", query="q"
    )

    call = services.search_service.search_semantic.await_args
    assert call.kwargs["force_strategy"] is None


@pytest.mark.asyncio
async def test_regex_search_ignores_strategy():
    services, manager = _strategy_fixture()

    await search_impl(
        services=services,
        embedding_manager=manager,
        type="regex",
        query="def retry",
        strategy="multi_hop",
    )

    services.search_service.search_semantic.assert_not_called()
    call = services.search_service.search_regex_async.await_args
    assert "force_strategy" not in call.kwargs


@pytest.mark.asyncio
async def test_unknown_strategy_is_rejected():
    services, manager = _strategy_fixture()

    with pytest.raises(ValueError, match="'single_hop' or 'multi_hop'"):
        await search_impl(
            services=services,
            embedding_manager=manager,
            type="semantic",
            query="q",
            strategy="foo",  # type: ignore[arg-type]
        )

    services.search_service.search_semantic.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "commit_arg",
    [
        {"commit_range": "HEAD~5..HEAD"},
        {"commit_hash": "abc123"},
        {"last_n_commits": 3},
    ],
)
@pytest.mark.parametrize("vector_source", ["diff", "both"])
async def test_strategy_with_a_commit_diff_search_is_rejected(
    commit_arg, vector_source
):
    # The diff path has no strategies, so the request could not be honoured.
    services, manager = _strategy_fixture()

    with patch(
        "chunkhound.mcp_server.tools._inject_diff_service", new=AsyncMock()
    ) as inject:
        with pytest.raises(ValueError) as excinfo:
            await search_impl(
                services=services,
                embedding_manager=manager,
                type="semantic",
                query="q",
                strategy="multi_hop",
                vector_source=vector_source,
                **commit_arg,
            )

    message = str(excinfo.value)
    assert "strategy" in message
    assert "commit_range/commit_hash/last_n_commits" in message
    inject.assert_not_called()
    services.search_service.search_semantic.assert_not_called()


@pytest.mark.asyncio
async def test_commit_diff_search_without_strategy_still_runs():
    services, manager = _strategy_fixture()

    with patch(
        "chunkhound.mcp_server.tools._inject_diff_service",
        new=AsyncMock(return_value=(services, None)),
    ) as inject:
        await search_impl(
            services=services,
            embedding_manager=manager,
            type="semantic",
            query="q",
            commit_range="HEAD~5..HEAD",
        )

    inject.assert_awaited_once()
    call = services.search_service.search_semantic.await_args
    assert call.kwargs["force_strategy"] is None


@pytest.mark.asyncio
async def test_strategy_with_commit_args_and_db_source_runs():
    # vector_source="db" ignores the commit input, so the strategy applies.
    services, manager = _strategy_fixture()

    with patch(
        "chunkhound.mcp_server.tools._inject_diff_service", new=AsyncMock()
    ) as inject:
        await search_impl(
            services=services,
            embedding_manager=manager,
            type="semantic",
            query="q",
            commit_range="HEAD~5..HEAD",
            vector_source="db",
            strategy="multi_hop",
        )

    inject.assert_not_called()
    call = services.search_service.search_semantic.await_args
    assert call.kwargs["force_strategy"] == "multi_hop"


@pytest.mark.asyncio
async def test_non_reranking_provider_searches_without_a_warning():
    # Omitting strategy must not trip the "multi-hop requested" fallback warning.
    from loguru import logger

    from chunkhound.services.search_service import SearchService

    provider = Mock()
    provider.name, provider.model = "openai", "text-embedding-3-small"
    provider.supports_reranking = Mock(return_value=False)
    provider.embed = AsyncMock(return_value=[[0.1, 0.2]])
    db = Mock()
    db.search_semantic = Mock(return_value=([], {"total": 0}))
    manager = Mock(spec=EmbeddingManager)
    manager.list_providers.return_value = ["openai"]
    manager.get_provider.return_value = provider
    services = Mock(spec=DatabaseServices)
    services.search_service = SearchService(db, provider)

    warnings: list[str] = []
    sink = logger.add(lambda m: warnings.append(str(m)), level="WARNING")
    try:
        await search_impl(
            services=services, embedding_manager=manager, type="semantic", query="q"
        )
    finally:
        logger.remove(sink)

    db.search_semantic.assert_called_once()
    assert warnings == []
