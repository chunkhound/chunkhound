import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _w(p: Path, s: str = "x\n"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def _simulate(dir_path: Path, env: dict[str, str]) -> list[str]:
    e = os.environ.copy()
    e.update(env)
    e["CHUNKHOUND_NO_RICH"] = "1"
    p = subprocess.run([
        "uv","run","chunkhound","index","--simulate", str(dir_path), "--sort","path"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=e, timeout=60)
    assert p.returncode == 0, p.stderr
    return [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]


@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
def test_nonrepo_root_gitignore_affects_only_nonrepo(tmp_path: Path):
    ws = tmp_path / "ws"
    # CH root .gitignore excludes datasets/
    _w(ws / ".gitignore", "datasets/\n")
    # Non-repo datasets
    _w(ws / "datasets" / "data.json", "{}\n")
    # Create a Git repo subtree
    repo = ws / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git","init"], cwd=str(repo), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _w(repo / "src" / "a.py", "print('ok')\n")
    subprocess.run(["git","-C", str(repo), "add","-A"], check=True)
    subprocess.run(["git","-C", str(repo), "config","user.email","ci@example.com"], check=True)
    subprocess.run(["git","-C", str(repo), "config","user.name","CI"], check=True)
    subprocess.run(["git","-C", str(repo), "commit","-m","init"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Enable overlay explicitly via CLI flag: datasets/ should be pruned; repo file should remain
    e = os.environ.copy()
    e.update({"CHUNKHOUND_INDEXING__DISCOVERY_BACKEND":"git", "CHUNKHOUND_NO_RICH":"1"})
    p = subprocess.run(["uv","run","chunkhound","index","--simulate", str(ws), "--sort","path", "--nonrepo-gitignore"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=e, timeout=60)
    assert p.returncode == 0, p.stderr
    s_def = set([ln.strip() for ln in p.stdout.splitlines() if ln.strip()])
    assert "datasets/data.json" not in s_def
    assert "repo/src/a.py" in s_def

    # Disable overlay via local config and verify datasets shows up
    cfg = {
        "indexing": {
            "workspace_gitignore_nonrepo": False,
        }
    }
    (ws / ".chunkhound.json").write_text(__import__("json").dumps(cfg), encoding="utf-8")
    files_disabled = _simulate(ws, env={"CHUNKHOUND_INDEXING__DISCOVERY_BACKEND":"git"})
    s_dis = set(files_disabled)
    assert "datasets/data.json" in s_dis
    assert "repo/src/a.py" in s_dis
