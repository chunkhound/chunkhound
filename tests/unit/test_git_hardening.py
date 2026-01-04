from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def test_git_wrapper_sets_sanitized_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from chunkhound.utils import git_safe

    captured = {}

    def fake_run(
        args,
        cwd=None,
        stdout=None,
        stderr=None,
        check=None,
        env=None,
        timeout=None,
        text=None,
    ):
        captured["args"] = args
        captured["cwd"] = cwd
        captured["env"] = env

        class P:
            returncode = 0
            stdout = ""
            stderr = ""

        return P()

    monkeypatch.setattr(subprocess, "run", fake_run)
    git_safe.run_git(["--version"], cwd=tmp_path, timeout_s=1.0)

    env = captured["env"]
    assert env is not None
    # Must not be empty; PATH must be present
    assert "PATH" in env
    # Git configs disabled to avoid reading user/system files
    assert env.get("GIT_CONFIG_NOSYSTEM") == "1"
    assert env.get("GIT_CONFIG_GLOBAL") is not None
    assert env.get("GIT_CONFIG_SYSTEM") is not None


def test_git_wrapper_handles_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from chunkhound.utils import git_safe

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=kwargs.get("args") or kwargs.get("cmd") or ["git"], timeout=0.001
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(git_safe.GitCommandError) as ei:
        git_safe.run_git(["status"], cwd=tmp_path, timeout_s=0.001)
    assert "timeout" in str(ei.value).lower()


def test_git_discovery_handles_timeout_gracefully(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Monkeypatch subprocess.run used by git_discovery to simulate timeout
    import chunkhound.utils.git_discovery as gd

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=kwargs.get("args") or kwargs.get("cmd") or ["git"], timeout=0.001
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    files, stats = gd.list_repo_files_via_git(
        tmp_path, tmp_path, include_patterns=["**/*.py"], config_excludes=[]
    )
    assert files == []
    assert stats.get("git_rows_total", 0) == 0


@pytest.mark.skipif(
    not (subprocess.run(["which", "git"], stdout=subprocess.DEVNULL).returncode == 0),
    reason="git required",
)
def test_git_discovery_handles_spaces_in_filenames(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir(parents=True, exist_ok=True)
    # init repo
    subprocess.run(
        ["git", "init"],
        cwd=str(repo),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (repo / "a b.py").write_text("print('ok')\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "-A"],
        cwd=str(repo),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=ci@example.com",
            "-c",
            "user.name=CI",
            "commit",
            "-m",
            "init",
        ],
        cwd=str(repo),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    from chunkhound.utils.git_discovery import list_repo_files_via_git

    files, _ = list_repo_files_via_git(
        repo, repo, include_patterns=["**/*.py"], config_excludes=[]
    )
    rels = sorted([p.resolve().relative_to(repo).as_posix() for p in files])
    assert "a b.py" in rels
