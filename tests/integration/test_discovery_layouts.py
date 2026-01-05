import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="git required for discovery backend integration tests",
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _git_init_and_commit(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init"],
        cwd=str(repo),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Configure minimal identity for CI environments
    _git(repo, "config", "user.email", "ci@example.com")
    _git(repo, "config", "user.name", "CI")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")


def _write(path: Path, content: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _simulate(dir_path: Path, discovery_backend: str | None = None) -> list[str]:
    env = os.environ.copy()
    if discovery_backend:
        env["CHUNKHOUND_INDEXING__DISCOVERY_BACKEND"] = discovery_backend
    # Keep output lean and deterministic
    env["CHUNKHOUND_NO_RICH"] = "1"
    res = subprocess.run(
        [
            "uv",
            "run",
            "chunkhound",
            "index",
            "--simulate",
            str(dir_path),
            "--sort",
            "path",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=90,
    )
    assert res.returncode == 0, f"simulate failed: {res.stderr}"
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    return lines


def _simulate_with_profile(
    dir_path: Path, discovery_backend: str | None = None
) -> tuple[list[str], dict]:
    env = os.environ.copy()
    if discovery_backend:
        env["CHUNKHOUND_INDEXING__DISCOVERY_BACKEND"] = discovery_backend
    env["CHUNKHOUND_NO_RICH"] = "1"
    res = subprocess.run(
        [
            "uv",
            "run",
            "chunkhound",
            "index",
            "--simulate",
            str(dir_path),
            "--sort",
            "path",
            "--profile-startup",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=90,
    )
    assert res.returncode == 0, f"simulate failed: {res.stderr}"
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    # Parse last JSON object from stderr (supports both simulate and full run formats)
    prof = {}
    for ln in res.stderr.splitlines()[::-1]:
        try:
            obj = __import__("json").loads(ln)
            if isinstance(obj, dict):
                if "startup_profile" in obj and isinstance(
                    obj["startup_profile"], dict
                ):
                    prof = obj["startup_profile"]
                    break
                if "discovery_ms" in obj:  # simulate format
                    prof = obj
                    break
        except Exception:
            continue
    return lines, prof


def test_layout_A_single_repo_tracked_under_ignored(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    # Files
    _write(repo / "src" / "a.py")
    # Heavier subtree we want ignored in general
    for i in range(25):
        _write(repo / "runs" / f"job{i:02d}" / f"f{i:02d}.txt")
    # Track one file under ignored prefix to expose backend differences
    tracked = repo / "runs" / "job00" / "kept.md"
    _write(tracked, "# keep\n")
    # .gitignore hides runs/
    _write(repo / ".gitignore", "**/runs/\n")
    # Initialize and commit tracked files (only src/ and kept file)
    _git_init_and_commit(repo)
    _git(repo, "add", "src/a.py")
    _git(repo, "add", "-f", "runs/job00/kept.md")
    _git(repo, "commit", "-m", "track kept under ignored")

    py_list = set(_simulate(repo, discovery_backend="python"))
    git_list = set(_simulate(repo, discovery_backend="git"))

    # Python excludes ignored subtree entirely; git includes the tracked file
    assert "runs/job00/kept.md" not in py_list
    assert "runs/job00/kept.md" in git_list


def test_layout_C_workspace_multi_repos_and_nonrepo(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    repo1 = ws / "repo1"
    repo2 = ws / "repo2"
    misc = ws / "misc"  # non-repo area

    # repo1
    _write(repo1 / "src" / "a.py")
    _write(repo1 / ".gitignore", "node_modules/\n")
    _git_init_and_commit(repo1)
    # repo2
    _write(repo2 / "lib" / "b.ts")
    _git_init_and_commit(repo2)
    # non-repo content
    _write(misc / "tool.py")

    py_set = set(_simulate(ws, discovery_backend="python"))
    git_set = set(_simulate(ws, discovery_backend="git"))
    git_only_set = set(_simulate(ws, discovery_backend="git_only"))

    # python covers everything
    assert "misc/tool.py" in py_set
    # git (hybrid) also covers non-repo via fallback
    assert "misc/tool.py" in git_set
    # git_only skips non-repo
    assert "misc/tool.py" not in git_only_set


def test_layout_D_nested_subrepo_boundary_respected(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sub = root / "subrepo"

    _write(root / "app" / "main.py")
    _write(root / ".gitignore", "subrepo/\n")  # parent ignores the folder name
    _git_init_and_commit(root)

    # Create subrepo with a file that would otherwise be ignored by parent .gitignore
    _write(sub / "pkg" / "mod.py")
    _git_init_and_commit(sub)

    # Both backends should see subrepo file (repo-aware boundary in python; Git sees subrepo too)
    py_set = set(_simulate(root, discovery_backend="python"))
    git_set = set(_simulate(root, discovery_backend="git"))
    assert "subrepo/pkg/mod.py" in py_set
    assert "subrepo/pkg/mod.py" in git_set


def test_layout_E_mixed_workspace_python_git_cover_nonrepo_git_only_skips(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "ws2"
    repo = ws / "repo"
    _write(repo / "src" / "x.py")
    _git_init_and_commit(repo)
    _write(ws / "datasets" / "sample.json")

    py_set = set(_simulate(ws, discovery_backend="python"))
    git_set = set(_simulate(ws, discovery_backend="git"))
    git_only_set = set(_simulate(ws, discovery_backend="git_only"))

    assert "datasets/sample.json" in py_set
    assert "datasets/sample.json" in git_set
    assert "datasets/sample.json" not in git_only_set


def test_auto_selects_python_git_gitonly(tmp_path: Path) -> None:
    # F layout: non-repo only → python
    ws = tmp_path / "auto_ws"
    _write(ws / "src" / "a.py")
    _, prof = _simulate_with_profile(ws, discovery_backend="auto")
    assert prof.get("resolved_backend") == "python"

    # I layout: all repos → git_only
    ws2 = tmp_path / "auto_ws2"
    r1 = ws2 / "r1"
    r2 = ws2 / "r2"
    _write(r1 / "a.py")
    _git_init_and_commit(r1)
    _write(r2 / "b.ts")
    _git_init_and_commit(r2)
    _, prof2 = _simulate_with_profile(ws2, discovery_backend="auto")
    assert prof2.get("resolved_backend") in {"git_only", "git"}

    # E layout: mixed workspace → git
    ws3 = tmp_path / "auto_ws3"
    r3 = ws3 / "r3"
    _write(r3 / "a.py")
    _git_init_and_commit(r3)
    _write(ws3 / "misc" / "note.md")
    _, prof3 = _simulate_with_profile(ws3, discovery_backend="auto")
    assert prof3.get("resolved_backend") == "git"


def test_layout_F_non_repo_only_favors_python(tmp_path: Path) -> None:
    ws = tmp_path / "scratch"
    _write(ws / "src" / "a.py")
    # No git here
    py_set = set(_simulate(ws, discovery_backend="python"))
    git_set = set(_simulate(ws, discovery_backend="git"))
    git_only_set = set(_simulate(ws, discovery_backend="git_only"))

    assert "src/a.py" in py_set
    # git backend falls back to python when no repos detected
    assert "src/a.py" in git_set
    # git_only has nothing to enumerate
    assert "src/a.py" not in git_only_set


def test_layout_H_monorepo_heavy_ignored_with_tracked_artifacts(tmp_path: Path) -> None:
    # Monorepo with a large ignored tree (runs/) but several tracked artifacts inside it
    repo = tmp_path / "heavy"
    _write(repo / "src" / "main.py", "print('ok')\n")
    # Create many untracked files under ignored prefix
    for j in range(30):
        for k in range(10):
            _write(repo / "runs" / f"job{j:02d}" / f"junk{k:02d}.txt")
    # Mark runs/ ignored
    _write(repo / ".gitignore", "**/runs/\n")
    _git_init_and_commit(repo)

    # Track several artifacts inside ignored prefix
    tracked_paths = []
    for t in range(20):
        p = repo / "runs" / f"job{t:02d}" / f"kept{t:02d}.md"
        _write(p, f"# kept {t}\n")
        tracked_paths.append(p)
    # Force-add tracked under ignored
    _git(repo, "add", "-f", *[str(p.relative_to(repo)) for p in tracked_paths])
    _git(repo, "commit", "-m", "track artifacts under ignored runs/")

    py_set = set(_simulate(repo, discovery_backend="python"))
    git_set = set(_simulate(repo, discovery_backend="git"))

    # All tracked artifacts should appear in git backend but not in python backend
    kept_rel = [str(p.relative_to(repo).as_posix()) for p in tracked_paths]
    assert all(k in git_set for k in kept_rel)
    assert all(k not in py_set for k in kept_rel)


def test_layout_I_workspace_all_repos_git_only_equals(tmp_path: Path) -> None:
    # Entire workspace is repos; git_only should match python and git results
    ws = tmp_path / "ws_all_repos"
    repos = [ws / f"r{i}" for i in range(3)]

    for idx, rr in enumerate(repos):
        # Create several files per repo
        for j in range(20):
            if j % 2 == 0:
                _write(rr / "pkg" / f"m{j:02d}.py", "print(1)\n")
            else:
                _write(rr / "lib" / f"u{j:02d}.ts", "export const x=1;\n")
        _git_init_and_commit(rr)

    py_set = set(_simulate(ws, discovery_backend="python"))
    git_set = set(_simulate(ws, discovery_backend="git"))
    git_only_set = set(_simulate(ws, discovery_backend="git_only"))

    # With no non-repo content and no tracked-under-ignored tricks, all should match
    assert py_set == git_set == git_only_set
