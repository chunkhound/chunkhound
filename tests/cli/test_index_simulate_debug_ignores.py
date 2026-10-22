from __future__ import annotations

import json
from pathlib import Path

from tests.helpers.cli import run_cli


def test_simulate_debug_ignores_prints_context(tmp_path: Path) -> None:
    # Create a trivial file so simulate has something to traverse
    (tmp_path / "README.md").write_text("hi")

    proc = run_cli(
        ["index", "--simulate", str(tmp_path), "--debug-ignores"], cwd=tmp_path
    )
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

    proc = run_cli(
        ["index", "--simulate", str(tmp_path), "--json", "--debug-ignores"],
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr

    # Stdout should be valid JSON and not contain debug text
    data = json.loads(proc.stdout)
    assert isinstance(data, dict) and "files" in data
    assert "debug-ignores" not in proc.stdout

    # Debug text goes to stderr
    assert "[debug-ignores] Active sources:" in proc.stderr
