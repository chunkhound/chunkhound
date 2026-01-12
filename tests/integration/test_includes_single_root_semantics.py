import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="git required",
)


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=check,
    )


def _git_init_and_commit(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git","init"], cwd=str(repo), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _git(repo, "config", "user.email", "ci@example.com")
    _git(repo, "config", "user.name", "CI")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")


def test_anchored_include_is_evaluated_from_ch_root(tmp_path: Path) -> None:
    ws = tmp_path
    repo = ws / "monorepo"
    # Files
    (repo / "src" / "app").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "app" / "main.ts").write_text("export const x=1;\n")
    (repo / "src" / "app" / "ignore.txt").write_text("nope\n")
    (repo / "other").mkdir(parents=True, exist_ok=True)
    (repo / "other" / "foo.ts").write_text("export const y=2;\n")
    _git_init_and_commit(repo)

    # Config anchored from CH root
    cfg = {
        "indexing": {
            "include": [
                "monorepo/src/**/*.ts"
            ]
        }
    }
    (ws / ".chunkhound.json").write_text(json.dumps(cfg))

    env = os.environ.copy()
    env["CHUNKHOUND_NO_RICH"] = "1"
    # Force git backend to exercise repo enumeration path
    env["CHUNKHOUND_INDEXING__DISCOVERY_BACKEND"] = "git"
    p = subprocess.run(
        ["uv","run","chunkhound","index","--simulate", str(ws), "--sort","path"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=90,
    )
    assert p.returncode == 0, p.stderr
    out = set(ln.strip() for ln in p.stdout.splitlines() if ln.strip())
    assert "monorepo/src/app/main.ts" in out
    assert "monorepo/other/foo.ts" not in out
    assert "monorepo/src/app/ignore.txt" not in out
