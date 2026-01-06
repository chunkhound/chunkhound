from pathlib import Path
from typing import Any

import pytest

from chunkhound.code_mapper import pipeline as code_mapper_pipeline


@pytest.mark.asyncio
async def test_hyde_prompt_persistence_is_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_collect_scope_files(*_: Any, **__: Any) -> list[str]:
        return []

    def fake_build_hyde_scope_prompt(*_: Any, **__: Any) -> str:
        return "scope prompt"

    async def fake_run_hyde_only_query(*_: Any, **__: Any) -> tuple[str, bool]:
        return "1. Example\n", True

    monkeypatch.setattr(
        code_mapper_pipeline, "collect_scope_files", fake_collect_scope_files
    )
    monkeypatch.setattr(
        code_mapper_pipeline, "build_hyde_scope_prompt", fake_build_hyde_scope_prompt
    )
    monkeypatch.setattr(
        code_mapper_pipeline, "run_hyde_only_query", fake_run_hyde_only_query
    )

    scope_label = "scope"
    arch_prompt_path = tmp_path / f"hyde_scope_prompt_arch_{scope_label}.md"
    ops_prompt_path = tmp_path / f"hyde_scope_prompt_ops_{scope_label}.md"

    await code_mapper_pipeline.run_code_mapper_overview_hyde(
        llm_manager=None,
        target_dir=tmp_path,
        scope_path=tmp_path,
        scope_label=scope_label,
        max_points=1,
        comprehensiveness="minimal",
        out_dir=tmp_path,
        map_hyde_provider=None,
        indexing_cfg=None,
    )

    assert not arch_prompt_path.exists()
    assert not ops_prompt_path.exists()

    monkeypatch.setenv("CH_CODE_MAPPER_WRITE_HYDE_PROMPT", "1")

    await code_mapper_pipeline.run_code_mapper_overview_hyde(
        llm_manager=None,
        target_dir=tmp_path,
        scope_path=tmp_path,
        scope_label=scope_label,
        max_points=1,
        comprehensiveness="minimal",
        out_dir=tmp_path,
        map_hyde_provider=None,
        indexing_cfg=None,
    )

    assert arch_prompt_path.exists()
    assert ops_prompt_path.exists()
    assert "scope prompt" in arch_prompt_path.read_text(encoding="utf-8")
    assert "scope prompt" in ops_prompt_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_hyde_prompt_persistence_when_persist_prompt_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_collect_scope_files(*_: Any, **__: Any) -> list[str]:
        return []

    def fake_build_hyde_scope_prompt(*_: Any, **__: Any) -> str:
        return "scope prompt"

    async def fake_run_hyde_only_query(*_: Any, **__: Any) -> tuple[str, bool]:
        return "1. Example\n", True

    monkeypatch.setattr(
        code_mapper_pipeline, "collect_scope_files", fake_collect_scope_files
    )
    monkeypatch.setattr(
        code_mapper_pipeline, "build_hyde_scope_prompt", fake_build_hyde_scope_prompt
    )
    monkeypatch.setattr(
        code_mapper_pipeline, "run_hyde_only_query", fake_run_hyde_only_query
    )

    scope_label = "scope"
    arch_prompt_path = tmp_path / f"hyde_scope_prompt_arch_{scope_label}.md"
    ops_prompt_path = tmp_path / f"hyde_scope_prompt_ops_{scope_label}.md"

    await code_mapper_pipeline.run_code_mapper_overview_hyde(
        llm_manager=None,
        target_dir=tmp_path,
        scope_path=tmp_path,
        scope_label=scope_label,
        max_points=1,
        comprehensiveness="minimal",
        out_dir=tmp_path,
        persist_prompt=True,
        map_hyde_provider=None,
        indexing_cfg=None,
    )

    assert arch_prompt_path.exists()
    assert ops_prompt_path.exists()
