import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    subprocess.run(["which", "git"], stdout=subprocess.DEVNULL).returncode != 0,
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
    subprocess.run(
        ["git", "init"],
        cwd=str(repo),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _git(repo, "config", "user.email", "ci@example.com")
    _git(repo, "config", "user.name", "CI")
    (repo / "a.py").write_text("print(1)\n")
    (repo / "b.txt").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")


def test_git_discovery_import_path_exposes_expected_api(tmp_path: Path) -> None:
    from chunkhound.utils import git_discovery as gd

    repo = tmp_path / "r"
    _git_init_and_commit(repo)

    includes = ["**/*.py", "**/*.txt"]
    files, stats = gd.list_repo_files_via_git(
        repo, repo, includes, config_excludes=[], filter_root=repo
    )

    assert any(p.name == "a.py" for p in files)
    assert any(p.name == "b.txt" for p in files)
    assert isinstance(stats.get("git_rows_tracked"), int)
