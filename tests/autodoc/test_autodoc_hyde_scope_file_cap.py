from pathlib import Path
from typing import Any

import pytest

from chunkhound.autodoc import pipeline as autodoc_pipeline


@pytest.mark.asyncio
async def test_autodoc_hyde_scope_file_cap_scales_with_comprehensiveness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_caps: list[int] = []

    def fake_collect_scope_files(*, hyde_cfg: Any, **__: Any) -> list[str]:
        seen_caps.append(int(getattr(hyde_cfg, "max_scope_files")))
        return []

    def fake_build_hyde_scope_prompt(*_: Any, **__: Any) -> str:
        return "scope prompt"

    async def fake_run_hyde_only_query(*_: Any, **__: Any) -> str:
        return "1. Example\n"

    monkeypatch.setattr(
        autodoc_pipeline, "collect_scope_files", fake_collect_scope_files
    )
    monkeypatch.setattr(
        autodoc_pipeline, "build_hyde_scope_prompt", fake_build_hyde_scope_prompt
    )
    monkeypatch.setattr(
        autodoc_pipeline, "run_hyde_only_query", fake_run_hyde_only_query
    )

    await autodoc_pipeline._run_autodoc_overview_hyde(
        llm_manager=None,
        target_dir=tmp_path,
        scope_path=tmp_path,
        scope_label="/",
        max_points=1,
        comprehensiveness="minimal",
        out_dir=None,
        assembly_provider=None,
        indexing_cfg=None,
    )
    await autodoc_pipeline._run_autodoc_overview_hyde(
        llm_manager=None,
        target_dir=tmp_path,
        scope_path=tmp_path,
        scope_label="/",
        max_points=20,
        comprehensiveness="ultra",
        out_dir=None,
        assembly_provider=None,
        indexing_cfg=None,
    )

    assert seen_caps == [200, 5000]
