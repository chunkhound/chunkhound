from pathlib import Path
from typing import Any

import pytest

from chunkhound.autodoc import pipeline as autodoc_pipeline


@pytest.mark.asyncio
async def test_hyde_prompt_persistence_is_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_collect_scope_files(*_: Any, **__: Any) -> list[str]:
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

    scope_label = "scope"
    prompt_path = tmp_path / f"hyde_scope_prompt_{scope_label}.md"

    await autodoc_pipeline._run_autodoc_overview_hyde(
        llm_manager=None,
        target_dir=tmp_path,
        scope_path=tmp_path,
        scope_label=scope_label,
        max_points=1,
        comprehensiveness="minimal",
        out_dir=tmp_path,
        assembly_provider=None,
        indexing_cfg=None,
    )

    assert not prompt_path.exists()

    monkeypatch.setenv("CH_AUTODOC_WRITE_HYDE_PROMPT", "1")

    await autodoc_pipeline._run_autodoc_overview_hyde(
        llm_manager=None,
        target_dir=tmp_path,
        scope_path=tmp_path,
        scope_label=scope_label,
        max_points=1,
        comprehensiveness="minimal",
        out_dir=tmp_path,
        assembly_provider=None,
        indexing_cfg=None,
    )

    assert prompt_path.exists()
    assert "scope prompt" in prompt_path.read_text(encoding="utf-8")
