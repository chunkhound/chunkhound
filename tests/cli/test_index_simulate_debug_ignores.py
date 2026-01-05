from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _run(
    cmd: list[str], cwd: Path | None = None, timeout: int = 25
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", *cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def test_simulate_debug_ignores_prints_context(tmp_path: Path) -> None:
    # Create a trivial file so simulate has something to traverse
    (tmp_path / "README.md").write_text("hi")

    proc = _run(["chunkhound", "index", "--simulate", str(tmp_path), "--debug-ignores"])
    assert proc.returncode == 0, proc.stderr

    # Debug info should be printed to stderr to avoid corrupting stdout
    err = proc.stderr
    assert "[debug-ignores] CH root:" in err
    assert str(tmp_path.resolve()) in err
    assert "[debug-ignores] Active sources:" in err
    # Default behavior includes gitignore
    assert "gitignore" in err

    # First 10 default excludes should include ChunkHound entries
    assert ".chunkhound.json" in err
    assert "/.chunkhound/" in err or "**/.chunkhound/**" in err


def test_simulate_debug_ignores_with_json_is_clean(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('x')\n")

    proc = _run(
        [
            "chunkhound",
            "index",
            "--simulate",
            str(tmp_path),
            "--json",
            "--debug-ignores",
        ]
    )
    assert proc.returncode == 0, proc.stderr

    # Stdout should be valid JSON and not contain debug text
    data = json.loads(proc.stdout)
    assert isinstance(data, dict) and "files" in data
    assert "debug-ignores" not in proc.stdout

    # Debug text goes to stderr
    assert "[debug-ignores] Active sources:" in proc.stderr
