"""Runtime dispatch + end-to-end propagation tests for ``code_research``.

Schema-surface tests live in ``test_previous_query_surface.py``; these
cover forwarding, service propagation, and empty-string coercion.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.mcp_server.tools import (
    TOOL_REGISTRY,
    deep_research_impl,
    execute_tool,
)


def _make_embedding_manager() -> MagicMock:
    provider = MagicMock()
    provider.supports_reranking.return_value = True
    manager = MagicMock()
    manager.list_providers.return_value = ["test"]
    manager.get_provider.return_value = provider
    return manager


@pytest.mark.asyncio
async def test_execute_tool_forwards_previous_query_to_code_research(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def spy(
        services,
        embedding_manager,
        llm_manager,
        query,
        *,
        previous_query=None,
        **kwargs,
    ):
        captured["query"] = query
        captured["previous_query"] = previous_query
        return {"answer": "spied"}

    monkeypatch.setattr(TOOL_REGISTRY["code_research"], "implementation", spy)

    await execute_tool(
        tool_name="code_research",
        arguments={"query": "how does auth work?", "previous_query": "prior topic"},
        services=MagicMock(),
        embedding_manager=_make_embedding_manager(),
        llm_manager=MagicMock(),
        config=MagicMock(),
        scan_progress=None,
    )

    assert captured.get("query") == "how does auth work?"
    assert captured.get("previous_query") == "prior topic"


@pytest.mark.asyncio
async def test_deep_research_impl_threads_previous_query_to_service() -> None:
    mock_research_service = MagicMock()
    mock_research_service.deep_research = AsyncMock(return_value={"answer": "ok"})

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
        await deep_research_impl(
            services=MagicMock(),
            embedding_manager=_make_embedding_manager(),
            llm_manager=MagicMock(),
            query="how does auth work?",
            previous_query="prior topic",
        )

    mock_research_service.deep_research.assert_awaited_once()
    call_kwargs = mock_research_service.deep_research.await_args.kwargs
    assert call_kwargs.get("previous_query") == "prior topic"


@pytest.mark.asyncio
async def test_deep_research_impl_coerces_empty_previous_query_to_none() -> None:
    mock_research_service = MagicMock()
    mock_research_service.deep_research = AsyncMock(return_value={"answer": "ok"})

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
        await deep_research_impl(
            services=MagicMock(),
            embedding_manager=_make_embedding_manager(),
            llm_manager=MagicMock(),
            query="q",
            previous_query="",
        )

    call_kwargs = mock_research_service.deep_research.await_args.kwargs
    assert call_kwargs.get("previous_query") is None


@pytest.mark.asyncio
async def test_research_command_coerces_empty_previous_query_to_none() -> None:
    import argparse

    from chunkhound.api.cli.commands import research as research_mod

    captured: dict[str, object] = {}

    async def spy_run_research(*args, **kwargs):
        captured["previous_query"] = kwargs.get("previous_query")

    args = argparse.Namespace(
        query="q",
        path_filter=None,
        verbose=False,
        commit_range=None,
        commit_hash=None,
        last_n_commits=None,
        vector_source="diff",
        previous_query="",
    )

    with (
        patch.object(research_mod, "run_research", spy_run_research),
        patch.object(research_mod, "verify_database_exists", return_value="/tmp/db"),
        patch.object(research_mod, "setup_embedding_manager", return_value=MagicMock()),
        patch.object(research_mod, "setup_llm_manager", return_value=MagicMock()),
        patch.object(research_mod, "create_services", return_value=MagicMock()),
    ):
        await research_mod.research_command(args, config=MagicMock())

    assert captured["previous_query"] is None
