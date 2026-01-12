import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git","-C",str(repo),*args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)


def _git_init_and_commit(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git","init"], cwd=str(repo), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _git(repo, "config", "user.email", "ci@example.com")
    _git(repo, "config", "user.name", "CI")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")


def test_config_exclude_evaluated_from_ch_root_in_git_backend(tmp_path: Path) -> None:
    ws = tmp_path
    repo = ws / "monorepo"
    target = repo / "camunda-modeler" / "custom" / "plugins" / "camunda-script-editor-plugin" / "client" / "client-bundle.js"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("bundle\n")
    _git_init_and_commit(repo)

    # Write CH config at workspace root with exclude including 'monorepo/...' prefix
    cfg = {
        "indexing": {
            "exclude": [
                "**/monorepo/camunda-modeler/custom/plugins/camunda-script-editor-plugin/client/client-bundle.js"
            ],
            "exclude_mode": "combined",
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
    rel = target.resolve().relative_to(ws.resolve()).as_posix()
    assert rel not in out, f"Excluded path {rel} should not appear in simulate output"

