from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.api.cli.commands import autodoc as autodoc_cmd


@pytest.mark.asyncio
async def test_autorun_code_mapper_for_autodoc_returns_plan_out_dir(
    tmp_path: Path, monkeypatch
) -> None:
    called: dict[str, object] = {}

    monkeypatch.setattr(
        autodoc_cmd,
        "_confirm_autorun_and_validate_prereqs",
        lambda **_kwargs: None,
    )

    options = autodoc_cmd._AutoMapOptions(
        map_out_dir=(tmp_path / "maps").resolve(),
        comprehensiveness="high",
        taint="technical",
        map_context=tmp_path / "ctx.md",
    )
    monkeypatch.setattr(
        autodoc_cmd,
        "_resolve_auto_map_options",
        lambda *, args, output_dir: options,
    )

    async def fake_run(**kwargs):  # type: ignore[no-untyped-def]
        called.update(kwargs)
        return autodoc_cmd._AutoMapPlan(
            map_out_dir=options.map_out_dir,
            map_scope=tmp_path,
            comprehensiveness=options.comprehensiveness,
            taint=options.taint,
        )

    monkeypatch.setattr(autodoc_cmd, "_run_code_mapper_for_autodoc", fake_run)

    formatter = SimpleNamespace(info=lambda _m: None)
    args = SimpleNamespace(verbose=True, config=None)

    out_dir = await autodoc_cmd._autorun_code_mapper_for_autodoc(
        args=args,
        config=SimpleNamespace(),  # type: ignore[arg-type]
        formatter=formatter,  # type: ignore[arg-type]
        output_dir=tmp_path / "site",
        question="Proceed?",
        decline_error="nope",
        decline_exit_code=2,
    )

    assert out_dir == options.map_out_dir
    assert called["map_out_dir"] == options.map_out_dir
    assert called["map_context"] == options.map_context
    assert called["comprehensiveness"] == options.comprehensiveness
    assert called["taint"] == options.taint

