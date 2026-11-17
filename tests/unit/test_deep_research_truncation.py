"""Contract tests: deep_research_impl returns the research dict unchanged.

There is no diff-chunk cap and no truncation notice to inject, so the
answer must reach the caller exactly as the research service produced it.
"""

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
async def test_diff_injection_path_preserves_research_dict():
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
            new=AsyncMock(return_value=mock_services),
        ),
        patch(
            "chunkhound.mcp_server.tools.ResearchServiceFactory.create",
            return_value=mock_research_service,
        ),
        patch(
            "chunkhound.mcp_server.tools.Config.from_environment",
            return_value=MagicMock(),
        ),
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


@pytest.mark.asyncio
async def test_plain_path_returns_dict_unchanged():
    """Without a commit range, the result dict is returned as-is."""
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
        patch(
            "chunkhound.mcp_server.tools.Config.from_environment",
            return_value=MagicMock(),
        ),
    ):
        result = await deep_research_impl(
            services=mock_services,
            embedding_manager=embedding_manager,
            llm_manager=mock_llm,
            query="how does auth work?",
        )

    assert isinstance(result, dict)
    assert result["answer"] == "clean answer"
