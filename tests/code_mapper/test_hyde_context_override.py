from pathlib import Path
from typing import Any

import pytest

from chunkhound.code_mapper import pipeline as code_mapper_pipeline


@pytest.mark.asyncio
async def test_overview_hyde_skips_scope_collection_when_context_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode_collect_scope_files(*_: Any, **__: Any) -> list[str]:
        raise AssertionError("collect_scope_files should not run when context is set")

    captured_contexts: list[str | None] = []

    def fake_build_hyde_scope_prompt(*, context: str | None = None, **__: Any) -> str:
        captured_contexts.append(context)
        return "scope prompt"

    async def fake_run_hyde_only_query(*_: Any, **__: Any) -> tuple[str, bool]:
        return "1. Example\n", True

    monkeypatch.setattr(
        code_mapper_pipeline, "collect_scope_files", explode_collect_scope_files
    )
    monkeypatch.setattr(
        code_mapper_pipeline, "build_hyde_scope_prompt", fake_build_hyde_scope_prompt
    )
    monkeypatch.setattr(
        code_mapper_pipeline, "run_hyde_only_query", fake_run_hyde_only_query
    )

    overview, pois = await code_mapper_pipeline.run_code_mapper_overview_hyde(
        llm_manager=None,
        target_dir=tmp_path,
        scope_path=tmp_path,
        scope_label="scope",
        context="STEERING CONTEXT",
        max_points=1,
        comprehensiveness="minimal",
        out_dir=None,
        map_hyde_provider=None,
        indexing_cfg=None,
    )

    assert "## Architectural Map (HyDE)" in overview
    assert "## Operational Map (HyDE)" in overview
    assert [p.mode for p in pois] == ["architectural", "operational"]
    assert captured_contexts == ["STEERING CONTEXT", "STEERING CONTEXT"]
