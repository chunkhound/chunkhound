from pathlib import Path
from typing import Any

import pytest

from chunkhound.code_mapper import service as code_mapper_service
from chunkhound.code_mapper.coverage import compute_db_scope_stats
from chunkhound.code_mapper.metadata import build_generation_stats_with_coverage
from chunkhound.core.types.common import Language
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.mcp_server.tools import deep_research_impl
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.embedding_service import EmbeddingService
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.services.search_service import SearchService
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider
from tests.integration.code_mapper_scope_helpers import write_scope_repo_layout


@pytest.mark.asyncio
async def test_run_code_mapper_pipeline_raises_when_no_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_overview(**_: Any) -> tuple[str, list[str]]:
        return "overview", []

    monkeypatch.setattr(
        code_mapper_service, "_run_code_mapper_overview_hyde", fake_overview
    )

    with pytest.raises(code_mapper_service.CodeMapperNoPointsError):
        await code_mapper_service.run_code_mapper_pipeline(
            services=object(),
            embedding_manager=object(),
            llm_manager=object(),
            target_dir=object(),
            scope_path=object(),
            scope_label="scope",
            path_filter=None,
            comprehensiveness="low",
            max_points=5,
            out_dir=None,
            assembly_provider=None,
            indexing_cfg=None,
            progress=None,
        )


@pytest.mark.asyncio
async def test_run_code_mapper_pipeline_skips_empty_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_overview(**_: Any) -> tuple[str, list[str]]:
        return "overview", ["Core Flow", "Error Handling"]

    async def fake_deep_research_impl(*, query: str, **__: Any) -> dict[str, Any]:
        if "Core Flow" in query:
            return {"answer": "", "metadata": {"sources": {"files": [], "chunks": []}}}
        return {
            "answer": "content",
            "metadata": {
                "sources": {
                    "files": ["scope/a.py"],
                    "chunks": [
                        {"file_path": "scope/a.py", "start_line": 1, "end_line": 2}
                    ],
                },
                "aggregation_stats": {"files_total": 3, "chunks_total": 4},
            },
        }

    monkeypatch.setattr(
        code_mapper_service, "_run_code_mapper_overview_hyde", fake_overview
    )
    monkeypatch.setattr(
        code_mapper_service, "deep_research_impl", fake_deep_research_impl
    )
    monkeypatch.setattr(
        code_mapper_service,
        "compute_db_scope_stats",
        lambda *_: (3, 4, set()),
    )

    result = await code_mapper_service.run_code_mapper_pipeline(
        services=object(),
        embedding_manager=object(),
        llm_manager=object(),
        target_dir=object(),
        scope_path=object(),
        scope_label="scope",
        path_filter=None,
        comprehensiveness="low",
        max_points=5,
        out_dir=None,
        assembly_provider=None,
        indexing_cfg=None,
        progress=None,
    )

    assert result.overview_result["answer"] == "overview"
    assert len(result.poi_sections) == 1
    assert "scope/a.py" in result.unified_source_files
    assert result.total_files_global == 3
    assert result.total_chunks_global == 4


@pytest.mark.asyncio
async def test_code_mapper_coverage_uses_deep_research_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    write_scope_repo_layout(repo_root)

    embedding_manager = EmbeddingManager()
    embedding_manager.register_provider(FakeEmbeddingProvider(), set_default=True)

    def _fake_create_provider(self, provider_config):
        return FakeLLMProvider()

    original_create_provider = LLMManager._create_provider
    LLMManager._create_provider = _fake_create_provider  # type: ignore[assignment]
    try:
        llm_manager = LLMManager(
            {"provider": "fake", "model": "fake-gpt"},
            {"provider": "fake", "model": "fake-gpt"},
        )
    finally:
        LLMManager._create_provider = original_create_provider  # type: ignore[assignment]

    db = DuckDBProvider(":memory:", base_directory=repo_root)
    db.connect()
    try:
        parser = create_parser_for_language(Language.PYTHON)
        coordinator = IndexingCoordinator(
            db,
            repo_root,
            embedding_manager.get_default_provider(),
            {Language.PYTHON: parser},
        )
        await coordinator.process_directory(
            repo_root, patterns=["**/*.py"], exclude_patterns=[]
        )
        await coordinator.generate_missing_embeddings()

        search_service = SearchService(db, embedding_manager.get_default_provider())
        embedding_service = EmbeddingService(
            db, embedding_manager.get_default_provider()
        )
        services = DatabaseServices(
            provider=db,
            indexing_coordinator=coordinator,
            search_service=search_service,
            embedding_service=embedding_service,
        )

        import chunkhound.services.deep_research_service as dr_mod

        async def _no_followups(*_: Any, **__: Any) -> list[str]:
            return []

        monkeypatch.setattr(
            dr_mod.DeepResearchService,
            "_generate_follow_up_questions",
            _no_followups,
            raising=True,
        )

        original_threshold = dr_mod.RELEVANCE_THRESHOLD
        dr_mod.RELEVANCE_THRESHOLD = 0.0
        try:
            result = await deep_research_impl(
                services=services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                query="alpha function",
                progress=None,
                path="scope",
            )
        finally:
            dr_mod.RELEVANCE_THRESHOLD = original_threshold

        metadata = result.get("metadata") or {}
        sources = metadata.get("sources") or {}
        files = sources.get("files") or []
        chunks = sources.get("chunks") or []

        assert files, "Expected deep research to report source files"
        assert chunks, "Expected deep research to report source chunks"

        unified_source_files = {path: "" for path in files}
        unified_chunks_dedup = list(chunks)

        scope_total_files, scope_total_chunks, _scoped = compute_db_scope_stats(
            services, "scope"
        )

        stats, _coverage = build_generation_stats_with_coverage(
            generator_mode="code_research",
            total_research_calls=1,
            unified_source_files=unified_source_files,
            unified_chunks_dedup=unified_chunks_dedup,
            scope_label="scope",
            scope_total_files=scope_total_files,
            scope_total_chunks=scope_total_chunks,
        )

        assert stats["files"]["referenced"] > 0
        assert stats["files"]["coverage"] not in ("0.00%", None)
    finally:
        db.disconnect()
