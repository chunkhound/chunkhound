from __future__ import annotations

from pathlib import Path

from tests.helpers.cli import run_cli


def test_simulate_hides_chunkhound_files_in_parallel_mode(tmp_path: Path) -> None:
    root = tmp_path

    # Force parallel discovery with a low threshold via local config
    (root / ".chunkhound.json").write_text(
        "{\n"
        '  "indexing": {\n'
        '    "exclude": ".gitignore",\n'
        '    "min_dirs_for_parallel": 1,\n'
        '    "max_discovery_workers": 2\n'
        "  }\n"
        "}\n"
    )

    # Create multiple top-level dirs so parallel path is taken
    for i in range(5):
        d = root / f"d{i}"
        d.mkdir()
        (d / f"f{i}.py").write_text("print('x')\n")

    # Also create hidden CH working directory; it must be ignored
    (root / ".chunkhound").mkdir(exist_ok=True)
    (root / ".chunkhound" / "cache.txt").write_text("x")

    # Simulate and ensure .chunkhound.json and .chunkhound/** are not listed
    proc = run_cli(["index", "--simulate", str(root), "--sort", "path"], cwd=root)
    assert proc.returncode == 0, proc.stderr

    lines = proc.stdout.strip().splitlines()
    assert all(not ln.strip().endswith(".chunkhound.json") for ln in lines)
    assert all(".chunkhound/" not in ln for ln in lines)
