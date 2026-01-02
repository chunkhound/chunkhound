from pathlib import Path
from typing import Any

import pytest

from chunkhound.code_mapper import pipeline as code_mapper_pipeline


@pytest.mark.asyncio
async def test_dual_hyde_overview_returns_both_maps_and_injects_quickstart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_collect_scope_files(*_: Any, **__: Any) -> list[str]:
        return []

    def fake_build_hyde_scope_prompt(*, mode: str, **__: Any) -> str:
        return f"{mode} scope prompt"

    async def fake_run_hyde_only_query(*, prompt: str, **__: Any) -> tuple[str, bool]:
        if "operational scope prompt" in prompt:
            return "1. Troubleshooting: common failures\n", True
        return "1. Core Flow: overview\n", True

    monkeypatch.setattr(
        code_mapper_pipeline, "collect_scope_files", fake_collect_scope_files
    )
    monkeypatch.setattr(
        code_mapper_pipeline, "build_hyde_scope_prompt", fake_build_hyde_scope_prompt
    )
    monkeypatch.setattr(
        code_mapper_pipeline, "run_hyde_only_query", fake_run_hyde_only_query
    )

    overview, pois = await code_mapper_pipeline._run_code_mapper_overview_hyde(
        llm_manager=None,
        target_dir=tmp_path,
        scope_path=tmp_path,
        scope_label="/",
        max_points=1,
        comprehensiveness="low",
        out_dir=None,
        map_hyde_provider=None,
        indexing_cfg=None,
    )

    assert "## Architectural Map (HyDE)" in overview
    assert "## Operational Map (HyDE)" in overview

    assert [p.mode for p in pois] == ["architectural", "operational", "operational"]
    assert "Core Flow" in pois[0].text
    assert "Quickstart" in pois[1].text
    assert "Troubleshooting" in pois[2].text
