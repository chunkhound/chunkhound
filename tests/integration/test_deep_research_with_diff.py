"""Integration tests proving deep_research_impl uses TransientSearchService
when commit_range / commit_hash / last_n_commits parameters are supplied.

The key claim:  after the injection block runs, the ResearchServiceFactory
receives a *swapped* DatabaseServices whose search_service is a
TransientSearchService wrapping the original.  We verify this by:

  1. Mocking stream_git_diff_fragments to yield a known diff block.
  2. Providing a controlled EmbeddingManager that returns unit-length vectors.
  3. Spying on the original search_service.search_semantic — in "diff" mode it
     must NOT be called; in "both" mode it WILL be called.
  4. Asserting that the services object received by ResearchServiceFactory has
     a TransientSearchService as its search_service.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.embeddings import EmbeddingManager, LocalEmbeddingResult
from chunkhound.services.research.shared.file_reader import FileReader
from chunkhound.services.transient_search_service import TransientSearchService

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


async def _stream_fake_diff(commit_range: str, cwd=None):
    yield FAKE_DIFF


def _make_embedding_manager(dim: int = 4) -> EmbeddingManager:
    """Return an EmbeddingManager whose embed_texts always returns unit vectors."""
    em = EmbeddingManager()

    provider = MagicMock()
    provider.name = "test"
    provider.model = "test-model"
    provider.dims = dim
    provider.distance = "cosine"
    provider.batch_size = 256
    provider.embedding_space_identity = "test-space"

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
            embedding_space_identity="test-space",
        )

    provider.embed = _embed
    em.register_provider(provider, set_default=True)

    # Also wire embed_texts on the manager itself so the injection block works
    async def _embed_texts(
        texts: list[str], provider_name: str | None = None
    ) -> LocalEmbeddingResult:
        import math

        vec = [1.0 / math.sqrt(dim)] * dim
        return LocalEmbeddingResult(
            embeddings=[vec[:] for _ in texts],
            model="test-model",
            provider="test",
            dims=dim,
            embedding_space_identity="test-space",
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
    """TransientSearchService is injected when commit_range is supplied."""
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
            "chunkhound.core.git_diff.stream_git_diff_fragments",
            _stream_fake_diff,
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

    # The factory must receive the streaming search service.
    received_services = captured.get("db_services")
    assert received_services is not None, "ResearchServiceFactory.create was not called"
    assert isinstance(received_services.search_service, TransientSearchService), (
        f"Expected TransientSearchService, got {type(received_services.search_service)}"
    )


@pytest.mark.asyncio
async def test_deep_research_synthesizes_from_diff_when_file_is_not_on_disk():
    """Historical diff hunks remain available when the working-tree file is absent."""
    from chunkhound.mcp_server.tools import deep_research_impl

    services = _make_services(_make_original_search_service())
    services.provider.get_base_directory.return_value = "/path/that/does/not/exist"
    em = _make_embedding_manager()
    llm_manager = MagicMock()
    llm_manager.get_utility_provider.return_value.estimate_tokens.side_effect = len

    def _fake_factory_create(**kwargs):
        db_services = kwargs["db_services"]
        research = MagicMock()

        async def deep_research(query, previous_query=None):
            chunks, _ = await db_services.search_service.search_semantic(query)
            evidence = await FileReader(db_services).read_files_with_budget(
                chunks, llm_manager
            )
            assert list(evidence) == ["src/auth.py"]
            assert "def logout(user):" in evidence["src/auth.py"]
            return {"answer": evidence["src/auth.py"]}

        research.deep_research = deep_research
        return research

    with (
        patch(
            "chunkhound.core.git_diff.stream_git_diff_fragments",
            _stream_fake_diff,
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

    assert "def logout(user):" in result["answer"]
    services.provider.get_base_directory.assert_not_called()


@pytest.mark.asyncio
async def test_deep_research_both_reads_database_results_from_current_source(tmp_path):
    """Mixed research keeps diff evidence while expanding DB source normally."""
    from chunkhound.mcp_server.tools import deep_research_impl

    current_file = tmp_path / "src" / "current.py"
    current_file.parent.mkdir()
    current_file.write_text(
        "from current_dependency import helper\n\n"
        "def current_function():\n"
        "    return 'CURRENT_SOURCE_MARKER'\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "auth.py").write_text(
        "def unchanged_helper():\n    return 'SAME_PATH_CURRENT_MARKER'\n",
        encoding="utf-8",
    )
    original_search = _make_original_search_service()
    original_search.search_semantic.return_value = (
        [
            {
                "chunk_id": 42,
                "file_path": "src/current.py",
                "content": "STALE_DB_SNIPPET",
                "start_line": 3,
                "end_line": 4,
                "similarity": 0.9,
                "score": 0.9,
            },
            {
                "chunk_id": 43,
                "file_path": "src/auth.py",
                "content": "STALE_SAME_PATH_SNIPPET",
                "start_line": 1,
                "end_line": 2,
                "similarity": 0.8,
                "score": 0.8,
            },
        ],
        {"offset": 0, "page_size": 50, "has_more": False, "total": 1},
    )
    services = _make_services(original_search)
    services.provider.get_transient_file_content_async = None
    services.provider.get_transient_file_content = None
    services.provider.get_base_directory.return_value = tmp_path
    em = _make_embedding_manager()
    llm_manager = MagicMock()
    llm_manager.get_utility_provider.return_value.estimate_tokens.side_effect = len

    def _fake_factory_create(**kwargs):
        db_services = kwargs["db_services"]
        research = MagicMock()

        async def deep_research(query, previous_query=None):
            chunks, _ = await db_services.search_service.search_semantic(
                query, force_strategy="single_hop"
            )
            evidence = await FileReader(db_services).read_files_with_budget(
                chunks, llm_manager
            )
            return {"answer": dict(evidence)}

        research.deep_research = deep_research
        return research

    with (
        patch(
            "chunkhound.core.git_diff.stream_git_diff_fragments",
            _stream_fake_diff,
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
            query="compare logout and the current helper",
            commit_range="HEAD~2..HEAD",
            vector_source="both",
        )

    evidence = result["answer"]
    assert "def logout(user):" in evidence["src/auth.py"]
    assert "SAME_PATH_CURRENT_MARKER" in evidence["src/auth.py"]
    assert "STALE_SAME_PATH_SNIPPET" not in evidence["src/auth.py"]
    assert "CURRENT_SOURCE_MARKER" in evidence["src/current.py"]
    assert "current_dependency" in evidence["src/current.py"]
    assert "STALE_DB_SNIPPET" not in evidence["src/current.py"]


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
            "chunkhound.core.git_diff.stream_git_diff_fragments",
            _stream_fake_diff,
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
    assert isinstance(svc, TransientSearchService), (
        f"Expected TransientSearchService in diff mode, got {type(svc)}"
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
    assert not isinstance(captured_search_service[0], TransientSearchService)


@pytest.mark.parametrize(
    ("commit_inputs", "expected_range"),
    [
        ({"commit_hash": "abc123"}, "abc123^..abc123"),
        ({"last_n_commits": 5}, "HEAD~5..HEAD"),
        ({"commit_range": "v1..v2"}, "v1..v2"),
    ],
)
async def test_semantic_search_returns_changed_code_for_commit_inputs(
    commit_inputs, expected_range
):
    """Each public shorthand selects the expected diff and returns its code."""
    from chunkhound.mcp_server.tools import search_impl

    async def stream_diff(*args, **kwargs):
        yield FAKE_DIFF

    git_diff = MagicMock(side_effect=stream_diff)
    with patch("chunkhound.core.git_diff.stream_git_diff_fragments", git_diff):
        result = await search_impl(
            services=_make_services(_make_original_search_service()),
            embedding_manager=_make_embedding_manager(),
            type="semantic",
            query="log the user out",
            **commit_inputs,
        )

    assert git_diff.call_args.args[0] == expected_range
    assert result["results"], "changed code must be searchable via the public tool"


@pytest.mark.parametrize(
    ("commit_inputs", "message"),
    [
        ({"commit_hash": ""}, "commit_hash must not be empty"),
        ({"commit_hash": "   "}, "commit_hash must not be empty"),
        ({"last_n_commits": 0}, "last_n_commits must be a positive integer"),
        ({"last_n_commits": -1}, "last_n_commits must be a positive integer"),
        ({"commit_range": "v1..v2", "commit_hash": "abc123"}, "at most one"),
        ({"commit_range": "v1..v2", "last_n_commits": 2}, "at most one"),
        ({"commit_hash": "abc123", "last_n_commits": 2}, "at most one"),
    ],
)
async def test_semantic_search_rejects_invalid_commit_inputs_before_git(
    commit_inputs, message
):
    from chunkhound.mcp_server.tools import search_impl

    git_diff = MagicMock()
    with (
        patch("chunkhound.core.git_diff.stream_git_diff_fragments", git_diff),
        pytest.raises(ValueError, match=message),
    ):
        await search_impl(
            services=_make_services(_make_original_search_service()),
            embedding_manager=_make_embedding_manager(),
            type="semantic",
            query="changed code",
            **commit_inputs,
        )
    git_diff.assert_not_called()


@pytest.mark.asyncio
async def test_deep_research_commit_hash_expands_to_range():
    """commit_hash='abc123' results in effective_commit_range='abc123^..abc123'."""
    from chunkhound.mcp_server.tools import deep_research_impl

    original_search = _make_original_search_service()
    services = _make_services(original_search)
    em = _make_embedding_manager()

    captured_range: list[str] = []

    async def _fake_stream_git_diff(commit_range: str, cwd=None):
        captured_range.append(commit_range)
        yield FAKE_DIFF

    captured_search_service: list[Any] = []

    def _fake_factory_create(**kwargs):
        db_svc = kwargs.get("db_services")
        captured_search_service.append(db_svc.search_service if db_svc else None)
        rs = MagicMock()
        rs.deep_research = AsyncMock(return_value={"answer": "hash-test"})
        return rs

    llm_manager = MagicMock()

    with (
        patch(
            "chunkhound.core.git_diff.stream_git_diff_fragments",
            _fake_stream_git_diff,
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
            query="what changed in this commit?",
            commit_hash="abc123",
        )

    assert result == {"answer": "hash-test"}
    await captured_search_service[0].search_semantic("query")
    assert captured_range == ["abc123^..abc123"], (
        f"Expected 'abc123^..abc123', got {captured_range}"
    )
    assert isinstance(captured_search_service[0], TransientSearchService)


@pytest.mark.asyncio
async def test_deep_research_last_n_commits_expands_correctly():
    """last_n_commits=5 results in effective_commit_range='HEAD~5..HEAD'."""
    from chunkhound.mcp_server.tools import deep_research_impl

    original_search = _make_original_search_service()
    services = _make_services(original_search)
    em = _make_embedding_manager()

    captured_range: list[str] = []

    async def _fake_stream_git_diff(commit_range: str, cwd=None):
        captured_range.append(commit_range)
        yield FAKE_DIFF

    captured_search_service: list[Any] = []

    def _fake_factory_create(**kwargs):
        db_svc = kwargs.get("db_services")
        captured_search_service.append(db_svc.search_service if db_svc else None)
        rs = MagicMock()
        rs.deep_research = AsyncMock(return_value={"answer": "n-commits"})
        return rs

    llm_manager = MagicMock()

    with (
        patch(
            "chunkhound.core.git_diff.stream_git_diff_fragments",
            _fake_stream_git_diff,
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
            query="summarize last 5 commits",
            last_n_commits=5,
        )

    assert result == {"answer": "n-commits"}
    await captured_search_service[0].search_semantic("query")
    assert captured_range == ["HEAD~5..HEAD"], (
        f"Expected 'HEAD~5..HEAD', got {captured_range}"
    )
    assert isinstance(captured_search_service[0], TransientSearchService)


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
