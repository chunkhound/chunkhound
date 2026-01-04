from __future__ import annotations

import subprocess
from pathlib import Path


def _run(
    cmd: list[str], cwd: Path | None = None, timeout: int = 20
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", *cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def test_simulate_shows_sizes_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a" * 10)
    (tmp_path / "b.txt").write_text("b" * 3)

    # Text output with sizes, ascending by size
    proc = _run(
        [
            "chunkhound",
            "index",
            "--simulate",
            str(tmp_path),
            "--show-sizes",
            "--sort",
            "size",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    # b.txt (3 bytes) should come before a.txt (10 bytes)
    assert lines[0].endswith("b.txt") and lines[1].endswith("a.txt"), lines

    # JSON output includes sizes
    proc_json = _run(
        [
            "chunkhound",
            "index",
            "--simulate",
            str(tmp_path),
            "--json",
            "--sort",
            "size_desc",
        ]
    )
    assert proc_json.returncode == 0, proc_json.stderr
    import json

    data = json.loads(proc_json.stdout)
    files = data.get("files", [])
    assert isinstance(files, list) and all("size_bytes" in f for f in files)
    # First should be the largest (a.txt)
    assert (
        files[0]["path"].endswith("a.txt")
        and files[0]["size_bytes"] >= files[1]["size_bytes"]
    )
