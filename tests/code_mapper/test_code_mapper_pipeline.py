import asyncio
from pathlib import Path
from typing import Any

import pytest

from chunkhound.code_mapper import service as code_mapper_service
from chunkhound.code_mapper.coverage import compute_db_scope_stats
from chunkhound.code_mapper.metadata import build_generation_stats_with_coverage
from chunkhound.code_mapper.models import CodeMapperPOI
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
    async def fake_overview(**_: Any) -> tuple[str, list[CodeMapperPOI]]:
        return "overview", []

    monkeypatch.setattr(
        code_mapper_service, "run_code_mapper_overview_hyde", fake_overview
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
            map_hyde_provider=None,
            indexing_cfg=None,
            poi_jobs=None,
            progress=None,
        )


@pytest.mark.asyncio
async def test_run_code_mapper_pipeline_skips_empty_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_overview(**_: Any) -> tuple[str, list[CodeMapperPOI]]:
        return "overview", [
            CodeMapperPOI(mode="architectural", text="Core Flow"),
            CodeMapperPOI(mode="architectural", text="Error Handling"),
        ]

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
        code_mapper_service, "run_code_mapper_overview_hyde", fake_overview
    )
    monkeypatch.setattr(
        code_mapper_service, "run_deep_research", fake_deep_research_impl
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
        map_hyde_provider=None,
        indexing_cfg=None,
        poi_jobs=None,
        progress=None,
    )

    assert result.overview_result["answer"] == "overview"
    assert len(result.poi_sections) == 1
    assert "scope/a.py" in result.unified_source_files
    assert result.total_files_global == 3
    assert result.total_chunks_global == 4


@pytest.mark.asyncio
async def test_run_code_mapper_pipeline_uses_mode_aware_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_overview(**_: Any) -> tuple[str, list[CodeMapperPOI]]:
        return "overview", [
            CodeMapperPOI(mode="architectural", text="Core Flow"),
            CodeMapperPOI(mode="operational", text="Quickstart / Local run"),
        ]

    seen_queries: list[str] = []

    async def fake_deep_research_impl(*, query: str, **__: Any) -> dict[str, Any]:
        seen_queries.append(query)
        return {
            "answer": "content",
            "metadata": {"sources": {"files": [], "chunks": []}},
        }

    monkeypatch.setattr(
        code_mapper_service, "run_code_mapper_overview_hyde", fake_overview
    )
    monkeypatch.setattr(
        code_mapper_service, "run_deep_research", fake_deep_research_impl
    )
    monkeypatch.setattr(
        code_mapper_service,
        "compute_db_scope_stats",
        lambda *_: (0, 0, set()),
    )

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
        map_hyde_provider=None,
        indexing_cfg=None,
        progress=None,
    )

    assert any("ARCHITECTURAL point of interest" in q for q in seen_queries)
    assert any("OPERATIONAL point of interest" in q for q in seen_queries)


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


@pytest.mark.asyncio
async def test_run_code_mapper_pipeline_runs_poi_research_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_overview(**_: Any) -> tuple[str, list[CodeMapperPOI]]:
        return "overview", [
            CodeMapperPOI(mode="architectural", text="Core Flow"),
            CodeMapperPOI(mode="architectural", text="Error Handling"),
            CodeMapperPOI(mode="architectural", text="Observability"),
        ]

    active_calls = 0
    max_active_calls = 0
    started_calls = 0
    saw_parallelism = asyncio.Event()
    release_calls = asyncio.Event()

    async def fake_deep_research_impl(*, query: str, **__: Any) -> dict[str, Any]:
        nonlocal active_calls, max_active_calls, started_calls
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        started_calls += 1
        if max_active_calls >= 2:
            saw_parallelism.set()
        try:
            await release_calls.wait()
        finally:
            active_calls -= 1
        return {
            "answer": f"content for {query}",
            "metadata": {"sources": {"files": [], "chunks": []}},
        }

    monkeypatch.setattr(code_mapper_service, "run_code_mapper_overview_hyde", fake_overview)
    monkeypatch.setattr(
        code_mapper_service, "run_deep_research", fake_deep_research_impl
    )
    monkeypatch.setattr(
        code_mapper_service,
        "compute_db_scope_stats",
        lambda *_: (0, 0, set()),
    )
    monkeypatch.setenv("CH_CODE_MAPPER_POI_CONCURRENCY", "2")

    pipeline_task = asyncio.create_task(
        code_mapper_service.run_code_mapper_pipeline(
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
            map_hyde_provider=None,
            indexing_cfg=None,
            poi_jobs=None,
            progress=None,
        )
    )

    try:
        await asyncio.wait_for(saw_parallelism.wait(), timeout=2.0)
    finally:
        release_calls.set()
    result = await pipeline_task

    assert started_calls == 3
    assert max_active_calls >= 2
    assert len(result.poi_sections) == 3


@pytest.mark.asyncio
async def test_run_code_mapper_pipeline_backs_off_to_serial_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_overview(**_: Any) -> tuple[str, list[CodeMapperPOI]]:
        return "overview", [
            CodeMapperPOI(mode="architectural", text="Alpha"),
            CodeMapperPOI(mode="architectural", text="Boom"),
            CodeMapperPOI(mode="architectural", text="Charlie"),
        ]

    events: list[str] = []
    alpha_started = asyncio.Event()
    alpha_can_finish = asyncio.Event()
    alpha_finished = asyncio.Event()
    boom_called = asyncio.Event()
    charlie_started = asyncio.Event()

    real_sleep = asyncio.sleep

    async def fake_sleep(_: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(code_mapper_service.random, "uniform", lambda *_: 0.0)
    monkeypatch.setattr(code_mapper_service.asyncio, "sleep", fake_sleep)

    async def fake_deep_research_impl(*, query: str, **__: Any) -> dict[str, Any]:
        if "Alpha" in query:
            events.append("alpha_start")
            alpha_started.set()
            await alpha_can_finish.wait()
            events.append("alpha_done")
            alpha_finished.set()
            return {
                "answer": "alpha",
                "metadata": {"sources": {"files": [], "chunks": []}},
            }
        if "Boom" in query:
            events.append("boom_start")
            boom_called.set()
            raise RuntimeError("boom")
        if "Charlie" in query:
            events.append("charlie_start")
            charlie_started.set()
            return {
                "answer": "charlie",
                "metadata": {"sources": {"files": [], "chunks": []}},
            }
        raise AssertionError("unexpected query")

    monkeypatch.setattr(code_mapper_service, "run_code_mapper_overview_hyde", fake_overview)
    monkeypatch.setattr(
        code_mapper_service, "run_deep_research", fake_deep_research_impl
    )
    monkeypatch.setattr(
        code_mapper_service,
        "compute_db_scope_stats",
        lambda *_: (0, 0, set()),
    )

    pipeline_task = asyncio.create_task(
        code_mapper_service.run_code_mapper_pipeline(
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
            map_hyde_provider=None,
            indexing_cfg=None,
            poi_jobs=2,
            progress=None,
        )
    )

    await asyncio.wait_for(alpha_started.wait(), timeout=2.0)
    await asyncio.wait_for(boom_called.wait(), timeout=2.0)

    assert not charlie_started.is_set(), "Expected serial backoff to block new PoIs"

    alpha_can_finish.set()
    await asyncio.wait_for(alpha_finished.wait(), timeout=2.0)
    result = await asyncio.wait_for(pipeline_task, timeout=2.0)

    assert events.index("alpha_done") < events.index("charlie_start")
    assert len(result.poi_sections) == 2
    assert len(result.failed_poi_sections) == 1


@pytest.mark.asyncio
async def test_run_code_mapper_pipeline_retries_empty_result_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_overview(**_: Any) -> tuple[str, list[CodeMapperPOI]]:
        return "overview", [CodeMapperPOI(mode="architectural", text="Alpha")]

    attempts = 0
    sleep_calls: list[float] = []
    uniform_calls: list[tuple[float, float]] = []

    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        await real_sleep(0)

    def fake_uniform(low: float, high: float) -> float:
        uniform_calls.append((low, high))
        return 0.0

    async def fake_deep_research_impl(*, query: str, **__: Any) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return {
                "answer": "",
                "metadata": {"sources": {"files": [], "chunks": []}},
            }
        return {
            "answer": "ok",
            "metadata": {"sources": {"files": [], "chunks": []}},
        }

    monkeypatch.setattr(code_mapper_service, "run_code_mapper_overview_hyde", fake_overview)
    monkeypatch.setattr(
        code_mapper_service, "run_deep_research", fake_deep_research_impl
    )
    monkeypatch.setattr(
        code_mapper_service,
        "compute_db_scope_stats",
        lambda *_: (0, 0, set()),
    )
    monkeypatch.setattr(code_mapper_service.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(code_mapper_service.random, "uniform", fake_uniform)

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
        map_hyde_provider=None,
        indexing_cfg=None,
        poi_jobs=2,
        progress=None,
    )

    assert attempts == 2
    assert uniform_calls == [(0.0, 1.0)]
    assert sleep_calls == [0.0]
    assert len(result.poi_sections) == 1
    assert result.failed_poi_sections == []


@pytest.mark.asyncio
async def test_run_code_mapper_pipeline_emits_poi_progress_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_overview(**_: Any) -> tuple[str, list[CodeMapperPOI]]:
        return "overview", [CodeMapperPOI(mode="architectural", text="Observability")]

    class FakeProgress:
        def __init__(self) -> None:
            self.events: list[tuple[str, str, int | None, int | None]] = []

        async def emit_event(
            self,
            event_type: str,
            message: str,
            node_id: int | None = None,
            depth: int | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> None:
            self.events.append((event_type, message, node_id, depth))

    async def fake_deep_research_impl(*, progress: Any, **__: Any) -> dict[str, Any]:
        await progress.emit_event("node_start", "inner", node_id=1, depth=0)
        return {"answer": "ok", "metadata": {"sources": {"files": [], "chunks": []}}}

    monkeypatch.setattr(code_mapper_service, "run_code_mapper_overview_hyde", fake_overview)
    monkeypatch.setattr(
        code_mapper_service, "run_deep_research", fake_deep_research_impl
    )
    monkeypatch.setattr(
        code_mapper_service,
        "compute_db_scope_stats",
        lambda *_: (0, 0, set()),
    )

    progress = FakeProgress()
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
        map_hyde_provider=None,
        indexing_cfg=None,
        poi_jobs=1,
        progress=progress,
    )

    assert result.poi_sections
    assert progress.events[0][0] == "poi_start"
    assert progress.events[0][2] == 1_000_000
    assert progress.events[0][3] == 0

    assert any(
        event_type == "node_start" and node_id == 1_000_002 and depth == 1
        for event_type, _message, node_id, depth in progress.events
    )
    assert any(event_type == "poi_complete" for event_type, *_ in progress.events)


@pytest.mark.asyncio
async def test_run_code_mapper_pipeline_ignores_progress_emit_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_overview(**_: Any) -> tuple[str, list[CodeMapperPOI]]:
        return "overview", [
            CodeMapperPOI(mode="architectural", text="Alpha"),
            CodeMapperPOI(mode="architectural", text="Bravo"),
        ]

    class FlakyProgress:
        async def emit_event(
            self,
            event_type: str,
            message: str,
            node_id: int | None = None,
            depth: int | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> None:
            raise RuntimeError(f"progress failed for {event_type}")

    async def fake_deep_research_impl(*, query: str, **__: Any) -> dict[str, Any]:
        return {"answer": f"ok: {query}", "metadata": {"sources": {"files": [], "chunks": []}}}

    monkeypatch.setattr(code_mapper_service, "run_code_mapper_overview_hyde", fake_overview)
    monkeypatch.setattr(
        code_mapper_service, "run_deep_research", fake_deep_research_impl
    )
    monkeypatch.setattr(
        code_mapper_service,
        "compute_db_scope_stats",
        lambda *_: (0, 0, set()),
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
        map_hyde_provider=None,
        indexing_cfg=None,
        poi_jobs=2,
        progress=FlakyProgress(),
    )

    assert len(result.poi_sections) == 2
    assert result.failed_poi_sections == []


@pytest.mark.asyncio
async def test_run_code_mapper_pipeline_ignores_inner_progress_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_overview(**_: Any) -> tuple[str, list[CodeMapperPOI]]:
        return "overview", [CodeMapperPOI(mode="architectural", text="Observability")]

    class FlakyProgress:
        async def emit_event(
            self,
            event_type: str,
            message: str,
            node_id: int | None = None,
            depth: int | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> None:
            raise RuntimeError(f"progress failed for {event_type}")

    async def fake_deep_research_impl(*, progress: Any, **__: Any) -> dict[str, Any]:
        await progress.emit_event("node_start", "inner", node_id=1, depth=0)
        return {"answer": "ok", "metadata": {"sources": {"files": [], "chunks": []}}}

    monkeypatch.setattr(code_mapper_service, "run_code_mapper_overview_hyde", fake_overview)
    monkeypatch.setattr(
        code_mapper_service, "run_deep_research", fake_deep_research_impl
    )
    monkeypatch.setattr(
        code_mapper_service,
        "compute_db_scope_stats",
        lambda *_: (0, 0, set()),
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
        map_hyde_provider=None,
        indexing_cfg=None,
        poi_jobs=1,
        progress=FlakyProgress(),
    )

    assert len(result.poi_sections) == 1
    assert result.failed_poi_sections == []

@pytest.mark.asyncio
async def test_run_code_mapper_pipeline_emits_poi_failed_only_after_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_overview(**_: Any) -> tuple[str, list[CodeMapperPOI]]:
        return "overview", [CodeMapperPOI(mode="architectural", text="Observability")]

    class FakeProgress:
        def __init__(self) -> None:
            self.events: list[tuple[str, str, int | None, int | None]] = []

        async def emit_event(
            self,
            event_type: str,
            message: str,
            node_id: int | None = None,
            depth: int | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> None:
            self.events.append((event_type, message, node_id, depth))

    real_sleep = asyncio.sleep

    async def fake_sleep(_: float) -> None:
        await real_sleep(0)

    async def fake_deep_research_impl(**__: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(code_mapper_service.random, "uniform", lambda *_: 0.0)
    monkeypatch.setattr(code_mapper_service.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(code_mapper_service, "run_code_mapper_overview_hyde", fake_overview)
    monkeypatch.setattr(
        code_mapper_service, "run_deep_research", fake_deep_research_impl
    )
    monkeypatch.setattr(
        code_mapper_service,
        "compute_db_scope_stats",
        lambda *_: (0, 0, set()),
    )

    progress = FakeProgress()
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
        map_hyde_provider=None,
        indexing_cfg=None,
        poi_jobs=1,
        progress=progress,
    )

    assert result.poi_sections == []
    assert len(result.failed_poi_sections) == 1

    event_types = [event_type for event_type, *_ in progress.events]
    assert event_types.count("poi_start") == 1
    assert event_types.count("poi_complete") == 0
    assert event_types.count("poi_failed") == 1
    assert event_types.index("poi_failed") > event_types.index("poi_start")

    assert any(
        event_type == "main_info"
        and depth == 1
        and message == "Retrying after error"
        for event_type, message, _node_id, depth in progress.events
    )
