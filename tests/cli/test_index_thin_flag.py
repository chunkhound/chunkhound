from __future__ import annotations

import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["uv", "run", *cmd], cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout)


def test_index_help_contains_thin_flag() -> None:
    proc = _run(["chunkhound", "index", "--help"], timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "--thin-missing-embeddings" in proc.stdout


def test_index_accepts_thin_flag_on_noop(tmp_path: Path) -> None:
    # minimal project with no embeddings to avoid provider validation
    (tmp_path / "a.py").write_text("print('ok')\n")
    proc = _run(["chunkhound", "index", str(tmp_path), "--thin-missing-embeddings", "--no-embeddings"], timeout=120)
    assert proc.returncode == 0, proc.stderr

