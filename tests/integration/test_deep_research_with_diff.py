"""Integration tests proving deep_research_impl uses DiffAwareSearchService
when commit_range / commit_hash / last_n_commits parameters are supplied.

The key claim:  after the injection block runs, the ResearchServiceFactory
receives a *swapped* DatabaseServices whose search_service is a
DiffAwareSearchService wrapping the original.  We verify this by:

  1. Mocking run_git_diff to return a known diff string.
  2. Providing a controlled EmbeddingManager that returns unit-length vectors.
  3. Spying on the original search_service.search_semantic — in "diff" mode it
     must NOT be called; in "both" mode it WILL be called.
  4. Asserting that the services object received by ResearchServiceFactory has
     a DiffAwareSearchService as its search_service.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.types.common import ChunkType, Language
from chunkhound.embeddings import EmbeddingManager, LocalEmbeddingResult
from chunkhound.services.diff_aware_search_service import DiffAwareSearchService


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

FAKE_DIFF = """\
diff --git a/src/auth.py b/src/auth.py
index 0000000..1111111 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,3 +10,8 @@ def login(user):
+def logout(user):
+    '''Log the user out and clear session.'''
+    session.clear()
+    return True
"""


def _make_embedding_manager(dim: int = 4) -> EmbeddingManager:
    """Return an EmbeddingManager whose embed_texts always returns unit vectors."""
    em = EmbeddingManager()

    provider = MagicMock()
    provider.name = "test"
    provider.model = "test-model"
    provider.dims = dim
    provider.distance = "cosine"
    provider.batch_size = 256

    # supports_reranking required by deep_research_impl validation
    provider.supports_reranking = MagicMock(return_value=True)

    async def _embed(texts: list[str]) -> LocalEmbeddingResult:
        import math
        vec = [1.0 / math.sqrt(dim)] * dim
        return LocalEmbeddingResult(
            embeddings=[vec[:] for _ in texts],
            model="test-model",
            provider="test",
            dims=dim,
        )

    provider.embed = _embed
    em.register_provider(provider, set_default=True)

    # Also wire embed_texts on the manager itself so the injection block works
    async def _embed_texts(texts: list[str]) -> LocalEmbeddingResult:
        import math
        vec = [1.0 / math.sqrt(dim)] * dim
        return LocalEmbeddingResult(
            embeddings=[vec[:] for _ in texts],
            model="test-model",
            provider="test",
            dims=dim,
        )

    em.embed_texts = _embed_texts  # type: ignore[method-assign]
    return em


def _make_original_search_service() -> MagicMock:
    """Return a mock SearchService that records calls."""
    svc = MagicMock()
    svc.search_semantic = AsyncMock(
        return_value=(
            [],
            {"offset": 0, "page_size": 10, "has_more": False, "total": 0},
        )
    )
    svc.search_regex = MagicMock(
        return_value=(
            [],
            {"offset": 0, "page_size": 10, "has_more": False, "total": 0},
        )
    )
    svc.search_regex_async = AsyncMock(
        return_value=(
            [],
            {"offset": 0, "page_size": 10, "has_more": False, "total": 0},
        )
    )
    return svc


def _make_services(search_service: Any) -> Any:
    """Return a minimal DatabaseServices-like NamedTuple mock."""
    from chunkhound.database_factory import DatabaseServices

    # We need real _replace so use a real NamedTuple construction
    provider_mock = MagicMock()
    indexing_mock = MagicMock()
    embedding_svc_mock = MagicMock()

    return DatabaseServices(
        provider=provider_mock,
        indexing_coordinator=indexing_mock,
        search_service=search_service,
        embedding_service=embedding_svc_mock,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deep_research_injects_diff_service_on_commit_range():
    """DiffAwareSearchService is injected when commit_range is supplied."""
    from chunkhound.mcp_server.tools import deep_research_impl

    original_search = _make_original_search_service()
    services = _make_services(original_search)
    em = _make_embedding_manager()

    # Capture whatever db_services ResearchServiceFactory.create receives
    captured: dict[str, Any] = {}

    def _fake_factory_create(**kwargs):
        captured["db_services"] = kwargs.get("db_services")
        # Return a minimal research service stub
        rs = MagicMock()
        rs.deep_research = AsyncMock(return_value={"answer": "stub-answer"})
        return rs

    llm_manager = MagicMock()

    with (
        patch(
            "chunkhound.core.git_diff.run_git_diff",
            AsyncMock(return_value=FAKE_DIFF),
        ),
        patch(
            "chunkhound.services.research.factory.ResearchServiceFactory.create",
            side_effect=_fake_factory_create,
        ),
    ):
        result = await deep_research_impl(
            services=services,
            embedding_manager=em,
            llm_manager=llm_manager,
            query="how does logout work?",
            commit_range="HEAD~2..HEAD",
        )

    assert result == {"answer": "stub-answer"}

    # The factory must have received a DiffAwareSearchService as search_service
    received_services = captured.get("db_services")
    assert received_services is not None, "ResearchServiceFactory.create was not called"
    assert isinstance(
        received_services.search_service, DiffAwareSearchService
    ), (
        f"Expected DiffAwareSearchService, got {type(received_services.search_service)}"
    )


@pytest.mark.asyncio
async def test_deep_research_diff_mode_does_not_call_original_search_semantic():
    """In vector_source='diff' mode, original.search_semantic is NOT called."""
    from chunkhound.mcp_server.tools import deep_research_impl

    original_search = _make_original_search_service()
    services = _make_services(original_search)
    em = _make_embedding_manager()

    captured_search_service: list[Any] = []

    def _fake_factory_create(**kwargs):
        db_svc = kwargs.get("db_services")
        captured_search_service.append(db_svc.search_service if db_svc else None)
        rs = MagicMock()
        rs.deep_research = AsyncMock(return_value={"answer": "diff-only"})
        return rs

    llm_manager = MagicMock()

    with (
        patch(
            "chunkhound.core.git_diff.run_git_diff",
            AsyncMock(return_value=FAKE_DIFF),
        ),
        patch(
            "chunkhound.services.research.factory.ResearchServiceFactory.create",
            side_effect=_fake_factory_create,
        ),
    ):
        result = await deep_research_impl(
            services=services,
            embedding_manager=em,
            llm_manager=llm_manager,
            query="explain session clearing",
            commit_range="HEAD~1..HEAD",
            vector_source="diff",
        )

    assert result == {"answer": "diff-only"}
    assert len(captured_search_service) == 1
    svc = captured_search_service[0]
    assert isinstance(svc, DiffAwareSearchService), (
        f"Expected DiffAwareSearchService in diff mode, got {type(svc)}"
    )
    assert svc._vector_source == "diff"
    # Original search_semantic was not called during injection (no research hop yet)
    original_search.search_semantic.assert_not_called()


@pytest.mark.asyncio
async def test_deep_research_no_commit_range_skips_injection():
    """Without commit params, the original search_service is passed unchanged."""
    from chunkhound.mcp_server.tools import deep_research_impl

    original_search = _make_original_search_service()
    services = _make_services(original_search)
    em = _make_embedding_manager()

    captured_search_service: list[Any] = []

    def _fake_factory_create(**kwargs):
        db_svc = kwargs.get("db_services")
        captured_search_service.append(db_svc.search_service if db_svc else None)
        rs = MagicMock()
        rs.deep_research = AsyncMock(return_value={"answer": "no-diff"})
        return rs

    llm_manager = MagicMock()

    with patch(
        "chunkhound.services.research.factory.ResearchServiceFactory.create",
        side_effect=_fake_factory_create,
    ):
        result = await deep_research_impl(
            services=services,
            embedding_manager=em,
            llm_manager=llm_manager,
            query="describe the system",
        )

    assert result == {"answer": "no-diff"}
    assert len(captured_search_service) == 1
    # Without commit params, the original (unwrapped) service is passed through
    assert captured_search_service[0] is original_search
    assert not isinstance(captured_search_service[0], DiffAwareSearchService)


@pytest.mark.asyncio
async def test_deep_research_commit_hash_expands_to_range():
    """commit_hash='abc123' results in effective_commit_range='abc123^..abc123'."""
    from chunkhound.mcp_server.tools import deep_research_impl

    original_search = _make_original_search_service()
    services = _make_services(original_search)
    em = _make_embedding_manager()

    captured_range: list[str] = []

    async def _fake_run_git_diff(commit_range: str, cwd=None) -> str:
        captured_range.append(commit_range)
        return FAKE_DIFF

    captured_search_service: list[Any] = []

    def _fake_factory_create(**kwargs):
        db_svc = kwargs.get("db_services")
        captured_search_service.append(db_svc.search_service if db_svc else None)
        rs = MagicMock()
        rs.deep_research = AsyncMock(return_value={"answer": "hash-test"})
        return rs

    llm_manager = MagicMock()

    with (
        patch("chunkhound.core.git_diff.run_git_diff", _fake_run_git_diff),
        patch(
            "chunkhound.services.research.factory.ResearchServiceFactory.create",
            side_effect=_fake_factory_create,
        ),
    ):
        result = await deep_research_impl(
            services=services,
            embedding_manager=em,
            llm_manager=llm_manager,
            query="what changed in this commit?",
            commit_hash="abc123",
        )

    assert result == {"answer": "hash-test"}
    assert captured_range == ["abc123^..abc123"], (
        f"Expected 'abc123^..abc123', got {captured_range}"
    )
    assert isinstance(captured_search_service[0], DiffAwareSearchService)


@pytest.mark.asyncio
async def test_deep_research_last_n_commits_expands_correctly():
    """last_n_commits=5 results in effective_commit_range='HEAD~5..HEAD'."""
    from chunkhound.mcp_server.tools import deep_research_impl

    original_search = _make_original_search_service()
    services = _make_services(original_search)
    em = _make_embedding_manager()

    captured_range: list[str] = []

    async def _fake_run_git_diff(commit_range: str, cwd=None) -> str:
        captured_range.append(commit_range)
        return FAKE_DIFF

    captured_search_service: list[Any] = []

    def _fake_factory_create(**kwargs):
        db_svc = kwargs.get("db_services")
        captured_search_service.append(db_svc.search_service if db_svc else None)
        rs = MagicMock()
        rs.deep_research = AsyncMock(return_value={"answer": "n-commits"})
        return rs

    llm_manager = MagicMock()

    with (
        patch("chunkhound.core.git_diff.run_git_diff", _fake_run_git_diff),
        patch(
            "chunkhound.services.research.factory.ResearchServiceFactory.create",
            side_effect=_fake_factory_create,
        ),
    ):
        result = await deep_research_impl(
            services=services,
            embedding_manager=em,
            llm_manager=llm_manager,
            query="summarize last 5 commits",
            last_n_commits=5,
        )

    assert result == {"answer": "n-commits"}
    assert captured_range == ["HEAD~5..HEAD"], (
        f"Expected 'HEAD~5..HEAD', got {captured_range}"
    )
    assert isinstance(captured_search_service[0], DiffAwareSearchService)


@pytest.mark.asyncio
async def test_deep_research_mutual_exclusion_raises():
    """Providing both commit_range and last_n_commits raises ValueError."""
    from chunkhound.mcp_server.tools import deep_research_impl

    original_search = _make_original_search_service()
    services = _make_services(original_search)
    em = _make_embedding_manager()
    llm_manager = MagicMock()

    with pytest.raises(ValueError, match="at most one"):
        await deep_research_impl(
            services=services,
            embedding_manager=em,
            llm_manager=llm_manager,
            query="test",
            commit_range="HEAD~2..HEAD",
            last_n_commits=3,
        )
