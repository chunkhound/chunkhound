"""Regression tests for uncapped transient diff research responses."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.mcp_server.tools import deep_research_impl


def _make_embedding_manager() -> MagicMock:
    provider = MagicMock()
    provider.supports_reranking.return_value = True
    manager = MagicMock()
    manager.list_providers.return_value = ["test"]
    manager.get_provider.return_value = provider
    return manager


@pytest.mark.asyncio
async def test_large_diff_path_has_no_cap_warning_and_preserves_dict():
    research_result = {"answer": "original answer text"}

    mock_research_service = MagicMock()
    mock_research_service.deep_research = AsyncMock(return_value=research_result)

    mock_services = MagicMock()
    mock_llm = MagicMock()
    embedding_manager = _make_embedding_manager()

    with (
        patch(
            "chunkhound.mcp_server.tools._resolve_commit_range",
            return_value="HEAD~1..HEAD",
        ),
        patch(
            "chunkhound.mcp_server.tools._inject_diff_service",
            new=AsyncMock(return_value=(mock_services, None)),
        ),
        patch(
            "chunkhound.mcp_server.tools.ResearchServiceFactory.create",
            return_value=mock_research_service,
        ),
        patch("chunkhound.mcp_server.tools.Config.from_environment", return_value=MagicMock()),
    ):
        result = await deep_research_impl(
            services=mock_services,
            embedding_manager=embedding_manager,
            llm_manager=mock_llm,
            query="how does auth work?",
            commit_range="HEAD~1..HEAD",
        )

    assert isinstance(result, dict)
    assert result["answer"] == "original answer text"
    assert "> **Note:**" not in result["answer"]


@pytest.mark.asyncio
async def test_no_truncation_warning_returns_dict_unchanged():
    """Without truncation_warning, result dict is returned as-is."""
    research_result = {"answer": "clean answer", "sources": []}

    mock_research_service = MagicMock()
    mock_research_service.deep_research = AsyncMock(return_value=research_result)

    mock_services = MagicMock()
    mock_llm = MagicMock()
    embedding_manager = _make_embedding_manager()

    with (
        patch("chunkhound.mcp_server.tools._resolve_commit_range", return_value=None),
        patch(
            "chunkhound.mcp_server.tools.ResearchServiceFactory.create",
            return_value=mock_research_service,
        ),
        patch("chunkhound.mcp_server.tools.Config.from_environment", return_value=MagicMock()),
    ):
        result = await deep_research_impl(
            services=mock_services,
            embedding_manager=embedding_manager,
            llm_manager=mock_llm,
            query="how does auth work?",
        )

    assert isinstance(result, dict)
    assert result["answer"] == "clean answer"
    assert "> **Note:**" not in result["answer"]
