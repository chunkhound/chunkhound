from pathlib import Path
from typing import Any

import pytest

import chunkhound.api.cli.commands.code_mapper as code_mapper_mod
from chunkhound.code_mapper.models import CodeMapperPOI
from chunkhound.core.config.config import Config


@pytest.mark.asyncio
async def test_code_mapper_comprehensiveness_minimal_maps_to_one_poi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_max_points: list[int] = []

    async def fake_overview(
        *,
        llm_manager: Any,
        target_dir: Path,
        scope_path: Path,
        scope_label: str,
        meta: Any | None = None,
        context: str | None = None,
        max_points: int = 10,
        comprehensiveness: str = "medium",
        out_dir: Path | None = None,
        persist_prompt: bool = False,
        map_hyde_provider: Any | None = None,
        indexing_cfg: Any | None = None,
    ) -> tuple[str, list[CodeMapperPOI]]:
        _ = (llm_manager, target_dir, scope_path, scope_label)
        _ = (meta, context, comprehensiveness, out_dir, persist_prompt, map_hyde_provider, indexing_cfg)
        seen_max_points.append(max_points)
        return "1. Example\n", [CodeMapperPOI(mode="architectural", text="Example")]

    monkeypatch.setattr(code_mapper_mod, "_run_code_mapper_overview_hyde", fake_overview)
    monkeypatch.setattr(
        code_mapper_mod,
        "build_llm_metadata_and_map_hyde",
        lambda **_: ({}, None),
    )

    config = Config(
        target_dir=tmp_path,
        database={"path": tmp_path / ".chunkhound" / "db", "provider": "duckdb"},
        embedding={
            "provider": "openai",
            "api_key": "test",
            "model": "text-embedding-3-small",
        },
        llm={"provider": "openai", "api_key": "test"},
    )

    class Args:
        def __init__(self) -> None:
            self.path = Path(".")
            self.verbose = False
            self.overview_only = True
            self.out = tmp_path / "out"
            self.comprehensiveness = "minimal"
            self.config = None
            self.db = None
            self.database_path = None

    await code_mapper_mod.code_mapper_command(Args(), config)

    assert seen_max_points == [1]
