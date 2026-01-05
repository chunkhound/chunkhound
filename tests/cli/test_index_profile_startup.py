from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _run(
    cmd: list[str], cwd: Path | None = None, timeout: int = 30
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", *cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def test_index_profile_startup_emits_json(tmp_path: Path) -> None:
    root = tmp_path
    (root / "a.py").write_text("print('x')\n")

    proc = _run(
        ["chunkhound", "index", str(root), "--no-embeddings", "--profile-startup"],
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr

    # Profile JSON is printed to stderr as the last line(s)
    # Extract the last JSON object safely by scanning lines
    err = proc.stderr.strip().splitlines()
    joined = "\n".join(err[-20:])  # tail window
    # Find first '{' from this tail and parse
    first = joined.find("{")
    assert first != -1, f"stderr tail did not contain JSON: {joined!r}"
    data = json.loads(joined[first:])
    assert "startup_profile" in data
    sp = data["startup_profile"]
    assert set(["discovery_ms", "cleanup_ms", "change_scan_ms"]).issubset(sp.keys())
