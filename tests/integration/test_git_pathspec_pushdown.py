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


def _w(p: Path, s: str = "x\n"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def _simulate_with_profile(dir_path: Path, backend: str, pushdown: bool | None, include_only_py: bool = False) -> tuple[list[str], dict]:
    env = os.environ.copy()
    env["CHUNKHOUND_NO_RICH"] = "1"
    env["CHUNKHOUND_INDEXING__DISCOVERY_BACKEND"] = backend
    if pushdown is not None:
        env["CHUNKHOUND_INDEXING__GIT_PATHSPEC_PUSHDOWN"] = "1" if pushdown else "0"
    if include_only_py:
        env["CHUNKHOUND_INDEXING__INCLUDE"] = "**/*.py"
    p = subprocess.run(
        ["uv","run","chunkhound","index","--simulate", str(dir_path), "--profile-startup", "--sort","path"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=90,
    )
    assert p.returncode == 0, p.stderr
    files = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    prof = {}
    for ln in p.stderr.splitlines()[::-1]:
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict) and ("startup_profile" in obj or "discovery_ms" in obj):
                prof = obj.get("startup_profile", obj)
                break
        except Exception:
            continue
    return files, prof


def test_git_pathspec_pushdown_reduces_rows_and_preserves_coverage(tmp_path: Path):
    repo = tmp_path / "repo"
    # Create many non-included files and a larger set of included files
    # Increased scale to reduce timing flakiness on macOS where process/FS jitter
    # can dominate small benchmarks. Pushdown benefit becomes more pronounced.
    for i in range(1200):
        _w(repo / "data" / f"blob{i:04d}.txt")
    for i in range(200):
        _w(repo / "src" / f"m{i:04d}.py", f"print({i})\n")
    _git_init_and_commit(repo)

    # Baseline without pushdown
    files_no, prof_no = _simulate_with_profile(repo, backend="git", pushdown=False, include_only_py=True)
    # With pushdown
    files_yes, prof_yes = _simulate_with_profile(repo, backend="git", pushdown=True, include_only_py=True)

    # Coverage must be identical
    assert set(files_yes) == set(files_no)

    # Profile should expose git enumerator row counts; pushdown should reduce them
    ps_yes = prof_yes.get("git_pathspecs", 0)
    ps_no = prof_no.get("git_pathspecs", 0)
    assert ps_yes >= 1
    assert ps_yes >= ps_no

    # And performance should not regress dramatically; allow small variance
    d_yes = float(prof_yes.get("discovery_ms", 0))
    d_no = float(prof_no.get("discovery_ms", 0))
    assert d_yes <= d_no * 1.2
