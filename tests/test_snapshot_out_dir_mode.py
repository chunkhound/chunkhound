#!/usr/bin/env python3
"""Tests for `chunkhound snapshot --out-dir-mode {prompt,reuse,force}` behavior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

import chunkhound.api.cli.commands.snapshot as snapshot_mod


def _write_minimal_valid_tui_assets(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "snapshot.chunk_systems.json").write_text(
        json.dumps({"schema_version": "snapshot.chunk_systems.v1"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "snapshot.chunk_systems.system_groups.json").write_text(
        json.dumps(
            {
                "schema_version": "snapshot.chunk_systems.system_groups.v1",
                "systems": [{"cluster_id": 1}, {"cluster_id": 2}],
                "partitions": [{"resolution": 1.0, "group_count": 1, "membership": [1, 1]}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "snapshot.chunk_systems.system_adjacency.json").write_text(
        json.dumps(
            {
                "schema_version": "snapshot.chunk_systems.system_adjacency_directed.v1",
                "systems": [{"cluster_id": 1}, {"cluster_id": 2}],
                "links": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_snapshot_out_dir_mode_prompt_non_tty_non_empty_fails_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "note.txt").write_text("non-empty\n", encoding="utf-8")

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    messages: list[str] = []

    def _capture_error(msg: object, *_args: object, **_kwargs: object) -> None:
        messages.append(str(msg))

    monkeypatch.setattr(snapshot_mod.logger, "error", _capture_error)

    args = argparse.Namespace(
        out_dir=out_dir,
        chunk_systems=True,
        verbose=False,
        tui=False,
        out_dir_mode="prompt",
    )
    with pytest.raises(SystemExit) as exc:
        await snapshot_mod.snapshot_command(args, config=object())

    assert int(exc.value.code) == 2
    joined = "\n".join(messages)
    assert "Non-interactive run with non-empty --out-dir" in joined
    assert "Use --out-dir-mode force|reuse." in joined


@pytest.mark.asyncio
async def test_snapshot_out_dir_mode_reuse_valid_assets_skips_compute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out_valid"
    _write_minimal_valid_tui_assets(out_dir)

    validate_calls: list[Path] = []
    original = snapshot_mod.validate_chunk_systems_tui_assets

    def _validate(out_dir_arg: Path) -> list[str]:
        validate_calls.append(Path(out_dir_arg))
        return original(out_dir_arg)

    monkeypatch.setattr(snapshot_mod, "validate_chunk_systems_tui_assets", _validate)

    args = argparse.Namespace(
        out_dir=out_dir,
        chunk_systems=True,
        verbose=False,
        tui=False,
        out_dir_mode="reuse",
    )
    # Pass a config object that would fail loudly if compute were attempted.
    await snapshot_mod.snapshot_command(args, config=object())

    assert validate_calls == [out_dir.resolve()]

