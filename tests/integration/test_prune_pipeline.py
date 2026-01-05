from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _run_simulate(root: Path, include: list[str]) -> list[str]:
    env = os.environ.copy()
    env["CHUNKHOUND_NO_RICH"] = "1"
    env["CHUNKHOUND_INDEXING__DISCOVERY_BACKEND"] = (
        "python"  # force python walker (non-repo)
    )
    env["CHUNKHOUND_INDEXING__INCLUDE"] = ",".join(include)
    p = subprocess.run(
        ["uv", "run", "chunkhound", "index", "--simulate", str(root), "--sort", "path"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=90,
    )
    assert p.returncode == 0, p.stderr
    return [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]


def test_include_prefix_prune_skips_unrelated_top_dirs(tmp_path: Path) -> None:
    # Non-repo workspace
    ws = tmp_path / "ws"
    # Create several top-level dirs
    (ws / "src" / "pkg").mkdir(parents=True, exist_ok=True)
    (ws / "docs").mkdir(parents=True, exist_ok=True)
    (ws / "node_modules" / "lib").mkdir(parents=True, exist_ok=True)

    # Files
    (ws / "src" / "pkg" / "a.ts").write_text("export const a=1;\n", encoding="utf-8")
    (ws / "docs" / "a.md").write_text("# doc\n", encoding="utf-8")
    (ws / "node_modules" / "lib" / "nm.ts").write_text("export{};\n", encoding="utf-8")

    # Narrow include anchored to "src/" subtree
    lines = _run_simulate(ws, include=["src/**/*.ts"])
    s = set(lines)
    assert "src/pkg/a.ts" in s
    # Unrelated top-level dirs should be pruned quickly
    assert "docs/a.md" not in s
    assert "node_modules/lib/nm.ts" not in s


def test_heavy_dirs_prune_skips_node_modules_but_respects_explicit_include(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "ws2"
    (ws / "node_modules" / "foo").mkdir(parents=True, exist_ok=True)
    (ws / "src").mkdir(parents=True, exist_ok=True)

    (ws / "node_modules" / "foo" / "skip.py").write_text("x=1\n", encoding="utf-8")
    (ws / "node_modules" / "foo" / "keep.py").write_text("x=1\n", encoding="utf-8")
    (ws / "src" / "a.py").write_text("print('ok')\n", encoding="utf-8")

    # Wide include (**/*.py) should still prune node_modules, so only src/a.py remains
    lines_wide = _run_simulate(ws, include=["**/*.py"])
    s_wide = set(lines_wide)
    assert "src/a.py" in s_wide
    assert "node_modules/foo/skip.py" not in s_wide
    assert "node_modules/foo/keep.py" not in s_wide

    # Even with an explicit include, default excludes drop node_modules content
    lines_explicit = _run_simulate(ws, include=["**/*.py", "node_modules/**/keep.py"])
    s_explicit = set(lines_explicit)
    assert "src/a.py" in s_explicit
    assert "node_modules/foo/keep.py" not in s_explicit
    assert "node_modules/foo/skip.py" not in s_explicit
