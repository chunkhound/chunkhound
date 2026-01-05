from __future__ import annotations

import os
from pathlib import Path

import pytest

from chunkhound.services.realtime_indexing_service import normalize_file_path

IS_WINDOWS = os.name == "nt"


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows-only path normalization test")
def test_normalize_file_path_converts_backslashes(tmp_path: Path) -> None:
    raw = str(tmp_path / "a" / "b" / "c.txt").replace("/", "\\")
    norm = normalize_file_path(raw)
    # Should produce an absolute path with canonical separators that Path.as_posix can represent
    assert Path(norm).is_absolute()
    assert Path(norm).as_posix().endswith("a/b/c.txt")


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows-only path matching test")
def test_ignore_engine_matches_with_windows_separators(tmp_path: Path) -> None:
    # Create a pseudo repo root with .gitignore ignoring build/
    (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")
    (tmp_path / "build").mkdir(parents=True, exist_ok=True)
    f = tmp_path / "build" / "x.txt"
    f.write_text("x\n", encoding="utf-8")

    from chunkhound.utils.ignore_engine import build_ignore_engine

    eng = build_ignore_engine(
        root=tmp_path,
        sources=["gitignore"],
        chignore_file=".chignore",
        config_exclude=[],
    )

    # Simulate backslash path string as would arrive on Windows events
    win_style = str(f).replace("/", "\\")
    assert eng.matches(Path(win_style), is_dir=False) is not None
