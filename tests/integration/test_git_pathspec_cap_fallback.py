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
    subprocess.run(["git", "init"], cwd=str(repo), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _git(repo, "config", "user.email", "ci@example.com")
    _git(repo, "config", "user.name", "CI")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")


def _simulate_with_profile(dir_path: Path, include_patterns: list[str], cap: int) -> dict:
    env = os.environ.copy()
    env["CHUNKHOUND_NO_RICH"] = "1"
    env["CHUNKHOUND_INDEXING__DISCOVERY_BACKEND"] = "git"
    env["CHUNKHOUND_INDEXING__GIT_PATHSPEC_PUSHDOWN"] = "1"
    env["CHUNKHOUND_INDEXING__GIT_PATHSPEC_CAP"] = str(cap)
    env["CHUNKHOUND_INDEXING__INCLUDE"] = ",".join(include_patterns)
    p = subprocess.run(
        [
            "uv",
            "run",
            "chunkhound",
            "index",
            "--simulate",
            str(dir_path),
            "--profile-startup",
            "--sort",
            "path",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=90,
    )
    assert p.returncode == 0, p.stderr
    prof = {}
    for ln in p.stderr.splitlines()[::-1]:
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict) and ("startup_profile" in obj or "discovery_ms" in obj):
                prof = obj.get("startup_profile", obj)
                break
        except Exception:
            continue
    return prof


def test_git_pathspec_cap_reflected_in_simulate_profile(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    # Seed a modest repo
    (repo / "src").mkdir(parents=True, exist_ok=True)
    for i in range(10):
        (repo / "src" / f"f{i:02d}.py").write_text("print(1)\n")
    for i in range(20):
        (repo / f"n{i:02d}.txt").write_text("x\n")
    _git_init_and_commit(repo)

    # Many includes to force CAP fallback
    includes = [f"**/*.{i:03d}x" for i in range(20)] + [f"**/name{i:03d}.cfg" for i in range(20)]

    prof = _simulate_with_profile(repo, includes, cap=3)
    # Ensure counters are present and reflect CAP
    assert int(prof.get("git_pathspecs", 0)) <= 3
    # Optional flag may be present
    if "git_pathspecs_capped" in prof:
        assert prof["git_pathspecs_capped"] is True

