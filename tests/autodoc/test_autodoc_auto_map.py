from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.api.cli.commands import autodoc as autodoc_command
from chunkhound.core.config.config import Config


@pytest.mark.asyncio
async def test_autodoc_offers_auto_map_when_map_dir_missing_index(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    provided_map_dir = tmp_path / "provided_maps"
    provided_map_dir.mkdir(parents=True)

    calls: list[tuple[Path, Path]] = []

    async def fake_generate_docsite(  # type: ignore[no-untyped-def]
        *, input_dir: Path, output_dir: Path, **_kwargs
    ):
        calls.append((input_dir, output_dir))
        if len(calls) == 1:
            raise FileNotFoundError("No AutoDoc index file found")
        return SimpleNamespace(output_dir=output_dir, pages=[], missing_topics=[])

    ran_map: list[Path] = []

    async def fake_run_code_mapper_for_autodoc(  # type: ignore[no-untyped-def]
        *,
        output_dir: Path,
        map_out_dir: Path | None,
        map_context: Path | None,
        comprehensiveness: str | None,
        taint: str | None,
        **_kwargs,
    ):
        plan = autodoc_command._build_auto_map_plan(
            output_dir=output_dir,
            map_out_dir=map_out_dir,
            comprehensiveness=comprehensiveness,
            taint=taint,
        )
        ran_map.append(plan.map_out_dir)
        assert map_context is None
        plan.map_out_dir.mkdir(parents=True, exist_ok=True)
        (plan.map_out_dir / "scope_code_mapper_index.md").write_text(
            "# AutoDoc Topics (/repo)\n\n1. [Topic One](topic_one.md)\n",
            encoding="utf-8",
        )
        (plan.map_out_dir / "topic_one.md").write_text(
            "# Topic One\n", encoding="utf-8"
        )
        assert plan.comprehensiveness == "medium"
        assert plan.taint == "balanced"
        return plan

    monkeypatch.setattr(autodoc_command, "generate_docsite", fake_generate_docsite)
    monkeypatch.setattr(
        autodoc_command,
        "_run_code_mapper_for_autodoc",
        fake_run_code_mapper_for_autodoc,
    )
    monkeypatch.setattr(autodoc_command, "_is_interactive", lambda: True)
    inputs = iter(["y", "", "", "", ""])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(inputs))

    args = SimpleNamespace(
        map_in=provided_map_dir,
        out_dir=tmp_path / "autodoc",
        map_out_dir=None,
        map_comprehensiveness=None,
        map_context=None,
        assets_only=False,
        site_title=None,
        site_tagline=None,
        cleanup_mode="minimal",
        cleanup_batch_size=1,
        cleanup_max_tokens=512,
        taint="balanced",
        map_taint=None,
        index_patterns=None,
        verbose=False,
        config=None,
    )

    await autodoc_command.autodoc_command(args, Config(target_dir=tmp_path))

    assert ran_map == [tmp_path / "map_autodoc"]
    assert calls == [
        (provided_map_dir, tmp_path / "autodoc"),
        (tmp_path / "map_autodoc", tmp_path / "autodoc"),
    ]


@pytest.mark.asyncio
async def test_autodoc_does_not_prompt_in_non_interactive_mode(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    provided_map_dir = tmp_path / "provided_maps"
    provided_map_dir.mkdir(parents=True)

    async def fake_generate_docsite(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("No AutoDoc index file found")

    monkeypatch.setattr(autodoc_command, "generate_docsite", fake_generate_docsite)
    monkeypatch.setattr(autodoc_command, "_is_interactive", lambda: False)

    args = SimpleNamespace(
        map_in=provided_map_dir,
        out_dir=tmp_path / "autodoc",
        map_out_dir=None,
        map_comprehensiveness=None,
        assets_only=False,
        site_title=None,
        site_tagline=None,
        cleanup_mode="minimal",
        cleanup_batch_size=1,
        cleanup_max_tokens=512,
        taint="balanced",
        map_taint=None,
        map_context=None,
        index_patterns=None,
        verbose=False,
        config=None,
    )

    with pytest.raises(SystemExit) as excinfo:
        await autodoc_command.autodoc_command(args, Config(target_dir=tmp_path))

    assert excinfo.value.code == 1


@pytest.mark.asyncio
async def test_autodoc_generates_map_when_map_in_omitted(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    calls: list[tuple[Path, Path]] = []

    async def fake_generate_docsite(  # type: ignore[no-untyped-def]
        *, input_dir: Path, output_dir: Path, **_kwargs
    ):
        calls.append((input_dir, output_dir))
        return SimpleNamespace(output_dir=output_dir, pages=[], missing_topics=[])

    ran_map: list[Path] = []

    async def fake_run_code_mapper_for_autodoc(  # type: ignore[no-untyped-def]
        *,
        output_dir: Path,
        map_out_dir: Path | None,
        map_context: Path | None,
        comprehensiveness: str | None,
        taint: str | None,
        **_kwargs,
    ):
        plan = autodoc_command._build_auto_map_plan(
            output_dir=output_dir,
            map_out_dir=map_out_dir,
            comprehensiveness=comprehensiveness,
            taint=taint,
        )
        ran_map.append(plan.map_out_dir)
        assert map_context is None
        return plan

    monkeypatch.setattr(autodoc_command, "generate_docsite", fake_generate_docsite)
    monkeypatch.setattr(
        autodoc_command,
        "_run_code_mapper_for_autodoc",
        fake_run_code_mapper_for_autodoc,
    )
    monkeypatch.setattr(autodoc_command, "_is_interactive", lambda: True)
    inputs = iter(["y", "", "", "", ""])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(inputs))

    args = SimpleNamespace(
        map_in=None,
        out_dir=tmp_path / "autodoc",
        map_out_dir=None,
        map_comprehensiveness=None,
        map_context=None,
        assets_only=False,
        site_title=None,
        site_tagline=None,
        cleanup_mode="minimal",
        cleanup_batch_size=1,
        cleanup_max_tokens=512,
        taint="balanced",
        map_taint=None,
        index_patterns=None,
        verbose=False,
        config=None,
    )

    await autodoc_command.autodoc_command(args, Config(target_dir=tmp_path))

    assert ran_map == [tmp_path / "map_autodoc"]
    assert calls == [(tmp_path / "map_autodoc", tmp_path / "autodoc")]
