from __future__ import annotations

from pathlib import Path

from tests.helpers.cli import run_cli


def test_default_excludes_exclude_chunkhound_files(tmp_path: Path) -> None:
    root = tmp_path
    # Create CH config and working dir
    (root / ".chunkhound.json").write_text('{\n  "indexing": { "exclude": [ ] }\n}\n')
    (root / ".chunkhound").mkdir(parents=True, exist_ok=True)
    # Avoid colliding with CH's DB path (which expects a file at .chunkhound/db).
    # Create a harmless directory under .chunkhound that should be excluded.
    (root / ".chunkhound" / "tmpdir").mkdir(parents=True, exist_ok=True)
    (root / ".chunkhound" / "tmpdir" / "ignored.txt").write_text("x")
    # Normal file that should appear
    (root / "ok.py").write_text("print('hi')\n")

    proc = run_cli(["index", "--simulate", str(root), "--sort", "path"], cwd=root)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.splitlines()
    # Ensure .chunkhound.json and .chunkhound/** are not listed
    assert all(not ln.strip().endswith(".chunkhound.json") for ln in out)
    assert all(".chunkhound/" not in ln for ln in out)
    # ok.py should be present
    assert any(ln.strip().endswith("ok.py") for ln in out)
