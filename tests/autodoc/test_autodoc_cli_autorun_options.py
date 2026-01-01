from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from chunkhound.api.cli.commands.autodoc import _resolve_auto_map_options


def test_resolve_auto_map_options_defaults_noninteractive(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = (tmp_path / "site").resolve()

    args = SimpleNamespace(
        map_out_dir=None,
        map_comprehensiveness=None,
        map_context=None,
        map_audience=None,
        audience="end-user",
    )

    options = _resolve_auto_map_options(args=args, output_dir=output_dir)

    assert options.map_out_dir == output_dir.with_name(f"map_{output_dir.name}")
    assert options.comprehensiveness == "medium"
    assert options.audience == "end-user"
    assert options.map_context is None


def test_resolve_auto_map_options_honors_explicit_args(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = (tmp_path / "site").resolve()

    args = SimpleNamespace(
        map_out_dir="maps",
        map_comprehensiveness="high",
        map_context="ctx.md",
        map_audience="technical",
        audience="balanced",
    )

    options = _resolve_auto_map_options(args=args, output_dir=output_dir)

    assert options.map_out_dir == (tmp_path / "maps").resolve()
    assert options.comprehensiveness == "high"
    assert options.audience == "technical"
    assert options.map_context == Path("ctx.md")
