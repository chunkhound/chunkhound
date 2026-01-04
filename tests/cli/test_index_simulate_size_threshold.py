"""Ensure simulate respects config_file_size_threshold_kb like real indexing.

Two cases:
1) Default threshold (20 KB) should hide large structured config files (JSON/YAML/TOML/HCL).
2) Threshold disabled (0) should show those files in simulate output.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _run(
    cmd: list[str], cwd: Path | None = None, timeout: int = 30
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("CHUNKHOUND_NO_RICH", "1")
    return subprocess.run(
        ["uv", "run", *cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
    )


def _write_large_json(path: Path, size_kb: int = 30) -> None:
    # Write a JSON file slightly larger than size_kb
    payload = {"k": "x" * 1024}  # ~1KB per entry
    items = max(1, size_kb)
    data = {f"item_{i:04d}": payload for i in range(items)}
    path.write_text(json.dumps(data))


def test_simulate_skips_large_structured_config_files(tmp_path: Path) -> None:
    # Fresh workspace without DB, create a large JSON > 20KB (default threshold)
    big = tmp_path / "data.json"
    _write_large_json(big, size_kb=30)
    # Add a small code file to ensure simulate lists something
    (tmp_path / "a.py").write_text("print('ok')\n")

    proc = _run(
        ["chunkhound", "index", "--simulate", str(tmp_path), "--sort", "path"],
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]

    assert any(ln.endswith("a.py") for ln in lines), (
        f"Expected a.py in simulate output: {lines!r}"
    )
    assert not any(ln.endswith("data.json") for ln in lines), (
        f"Large JSON should be skipped by default threshold: {lines!r}"
    )


def test_simulate_includes_when_threshold_disabled(tmp_path: Path) -> None:
    # Disable threshold via local config
    cfg = {"indexing": {"config_file_size_threshold_kb": 0}}
    (tmp_path / ".chunkhound.json").write_text(json.dumps(cfg))
    big = tmp_path / "data.json"
    _write_large_json(big, size_kb=30)

    proc = _run(
        ["chunkhound", "index", "--simulate", str(tmp_path), "--sort", "path"],
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]

    assert any(ln.endswith("data.json") for ln in lines), (
        f"Threshold disabled: expected data.json, got: {lines!r}"
    )
