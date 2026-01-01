from typing import Any

import pytest

from chunkhound.code_mapper import service as code_mapper_service


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
