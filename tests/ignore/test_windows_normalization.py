from __future__ import annotations

import os
from pathlib import Path

import pytest

from chunkhound.services.realtime.events import normalize_file_path

IS_WINDOWS = os.name == "nt"


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows-only path normalization test")
def test_normalize_file_path_converts_backslashes(tmp_path: Path) -> None:
    raw = str(tmp_path / "a" / "b" / "c.txt").replace("/", "\\")
    norm = normalize_file_path(raw)
    # Absolute path with canonical separators that Path.as_posix can represent
    assert Path(norm).is_absolute()
    assert Path(norm).as_posix().endswith("a/b/c.txt")
