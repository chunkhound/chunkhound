from __future__ import annotations

import subprocess
from pathlib import Path


def _run(
    cmd: list[str], cwd: Path | None = None, timeout: int = 30
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", *cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _git(
    args: list[str], cwd: Path, timeout: int = 20
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def test_git_discovery_skips_linked_worktree_under_parent_ignored_dir(
    tmp_path: Path,
) -> None:
    ws = tmp_path

    parent = ws / "parent"
    parent.mkdir()
    assert _git(["init"], parent).returncode == 0
    assert _git(["config", "user.email", "test@example.com"], parent).returncode == 0
    assert _git(["config", "user.name", "Test User"], parent).returncode == 0

    # Parent ignores the subtree containing a nested worktree.
    (parent / ".gitignore").write_text("ignored/\n")
    (parent / "main.py").write_text("print('ok')\n")

    ignored_dir = parent / "ignored"
    wt = ignored_dir / "wt"
    assert _git(["add", ".gitignore", "main.py"], parent).returncode == 0
    assert _git(["commit", "-m", "init"], parent).returncode == 0
    assert _git(["worktree", "add", "-b", "wtbranch", str(wt)], parent).returncode == 0
    assert (wt / ".git").is_file()

    db_path = ws / "chunks.duckdb"
    proc = _run(
        [
            "chunkhound",
            "index",
            "--simulate",
            "--discovery-backend",
            "git",
            "--db",
            str(db_path),
            str(ws),
        ],
        cwd=ws,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    out = set(proc.stdout.strip().splitlines())

    assert "parent/main.py" in out
    assert "parent/ignored/wt/main.py" not in out

    # Also verify the Python walker path (repo-aware ignore engine) does not treat
    # the linked worktree as a boundary when its root directory is ignored by parent.
    db_path2 = ws / "chunks2.duckdb"
    proc2 = _run(
        [
            "chunkhound",
            "index",
            "--simulate",
            "--discovery-backend",
            "python",
            "--db",
            str(db_path2),
            str(ws),
        ],
        cwd=ws,
        timeout=60,
    )
    assert proc2.returncode == 0, proc2.stderr
    out2 = set(proc2.stdout.strip().splitlines())
    assert "parent/main.py" in out2
    assert "parent/ignored/wt/main.py" not in out2
