from typing import Any

import pytest

from chunkhound.code_mapper import service as code_mapper_service
from chunkhound.code_mapper.models import CodeMapperPOI


@pytest.mark.asyncio
async def test_run_code_mapper_overview_only_raises_when_no_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_overview(**_: Any) -> tuple[str, list[CodeMapperPOI]]:
        return "overview", []

    monkeypatch.setattr(
        code_mapper_service, "run_code_mapper_overview_hyde", fake_overview
    )

    with pytest.raises(code_mapper_service.CodeMapperNoPointsError):
        await code_mapper_service.run_code_mapper_overview_only(
            llm_manager=None,
            target_dir=object(),
            scope_path=object(),
            scope_label="scope",
            max_points=5,
            comprehensiveness="low",
            out_dir=None,
            map_hyde_provider=None,
            indexing_cfg=None,
        )


@pytest.mark.asyncio
async def test_run_code_mapper_overview_only_returns_answer_and_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_overview(**_: Any) -> tuple[str, list[CodeMapperPOI]]:
        return "overview", [CodeMapperPOI(mode="architectural", text="Point")]

    monkeypatch.setattr(
        code_mapper_service, "run_code_mapper_overview_hyde", fake_overview
    )

    result = await code_mapper_service.run_code_mapper_overview_only(
        llm_manager=None,
        target_dir=object(),
        scope_path=object(),
        scope_label="scope",
        max_points=5,
        comprehensiveness="low",
        out_dir=None,
        map_hyde_provider=None,
        indexing_cfg=None,
    )

    assert result == ("overview", [CodeMapperPOI(mode="architectural", text="Point")])
