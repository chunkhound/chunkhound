from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.api.cli.commands import autodoc as autodoc_command
from chunkhound.api.cli.commands import autodoc_cleanup
from chunkhound.api.cli.commands import autodoc_generate
from chunkhound.api.cli.commands import autodoc_prompts
from chunkhound.core.config.config import Config


@pytest.mark.asyncio
async def test_autodoc_prompts_before_deleting_existing_topics_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    map_dir = tmp_path / "maps"
    map_dir.mkdir(parents=True)

    output_dir = tmp_path / "out"
    topics_dir = output_dir / "src" / "pages" / "topics"
    topics_dir.mkdir(parents=True, exist_ok=True)
    (topics_dir / "sentinel.md").write_text("sentinel", encoding="utf-8")

    monkeypatch.setattr(autodoc_prompts, "is_interactive", lambda: True)
    monkeypatch.setattr(autodoc_cleanup, "resolve_llm_manager", lambda **_kw: object())
    inputs = iter(["n"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(inputs))

    calls: list[dict[str, object]] = []

    async def fake_generate_docsite(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return SimpleNamespace(output_dir=output_dir, pages=[], missing_topics=[])

    monkeypatch.setattr(autodoc_generate, "generate_docsite", fake_generate_docsite)

    args = SimpleNamespace(
        map_in=map_dir,
        out_dir=output_dir,
        force=False,
        assets_only=False,
        site_title=None,
        site_tagline=None,
        cleanup_mode="llm",
        cleanup_batch_size=1,
        cleanup_max_tokens=512,
        audience="balanced",
        map_out_dir=None,
        map_comprehensiveness=None,
        map_context=None,
        map_audience=None,
        index_patterns=None,
        verbose=False,
        config=None,
    )

    with pytest.raises(SystemExit) as excinfo:
        await autodoc_command.autodoc_command(args, Config(target_dir=tmp_path))

    assert excinfo.value.code == 2
    assert calls == []


@pytest.mark.asyncio
async def test_autodoc_force_allows_deleting_existing_topics_dir_non_interactive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    map_dir = tmp_path / "maps"
    map_dir.mkdir(parents=True)

    output_dir = tmp_path / "out"
    topics_dir = output_dir / "src" / "pages" / "topics"
    topics_dir.mkdir(parents=True, exist_ok=True)
    (topics_dir / "sentinel.md").write_text("sentinel", encoding="utf-8")

    monkeypatch.setattr(autodoc_prompts, "is_interactive", lambda: False)
    monkeypatch.setattr(autodoc_cleanup, "resolve_llm_manager", lambda **_kw: object())

    calls: list[dict[str, object]] = []

    async def fake_generate_docsite(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return SimpleNamespace(output_dir=output_dir, pages=[], missing_topics=[])

    monkeypatch.setattr(autodoc_generate, "generate_docsite", fake_generate_docsite)

    args = SimpleNamespace(
        map_in=map_dir,
        out_dir=output_dir,
        force=True,
        assets_only=False,
        site_title=None,
        site_tagline=None,
        cleanup_mode="llm",
        cleanup_batch_size=1,
        cleanup_max_tokens=512,
        audience="balanced",
        map_out_dir=None,
        map_comprehensiveness=None,
        map_context=None,
        map_audience=None,
        index_patterns=None,
        verbose=False,
        config=None,
    )

    await autodoc_command.autodoc_command(args, Config(target_dir=tmp_path))

    assert calls
    assert calls[0].get("allow_delete_topics_dir") is True
