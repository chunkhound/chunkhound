"""``last_n_commits`` means the last N commits, not N first-parent hops (#512)."""

import os
import subprocess
from pathlib import Path

import pytest
from loguru import logger

from chunkhound.core.git_diff import (
    resolve_last_n_commits_range,
    stream_git_diff_fragments,
)

_DAY = 86400
_BASE_TIME = 1_700_000_000


def _git(repo: Path, *args: str, when: int | None = None) -> str:
    env = os.environ.copy()
    env.update(
        GIT_AUTHOR_NAME="Test",
        GIT_AUTHOR_EMAIL="test@example.com",
        GIT_COMMITTER_NAME="Test",
        GIT_COMMITTER_EMAIL="test@example.com",
    )
    if when is not None:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = f"{when} +0000"
    result = subprocess.run(
        ["git", *args], cwd=repo, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _init(repo: Path, object_format: str = "sha1") -> None:
    result = subprocess.run(
        ["git", "init", "-b", "main", f"--object-format={object_format}"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and object_format == "sha256":
        pytest.skip("installed Git does not support SHA-256 repositories")
    assert result.returncode == 0, result.stderr


def _commit(repo: Path, name: str, step: int) -> None:
    (repo / name).write_text(f"{name} = True\n", encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-m", name, when=_BASE_TIME + step * _DAY)


async def _diff_text(commit_range: str, repo: Path) -> str:
    return "".join(
        [fragment async for fragment in stream_git_diff_fragments(commit_range, repo)]
    )


@pytest.fixture
def linear_repo(tmp_path: Path) -> Path:
    _init(tmp_path)
    for step in range(8):
        _commit(tmp_path, f"c{step}.py", step)
    return tmp_path


@pytest.fixture
def merge_repo(tmp_path: Path) -> Path:
    """c1 - c2 - m3 - M (merge of f1 - f2): 6 commits, first-parent depth 3."""
    _init(tmp_path)
    _commit(tmp_path, "c1.py", 1)
    _commit(tmp_path, "c2.py", 2)
    _git(tmp_path, "checkout", "-b", "feature")
    _commit(tmp_path, "f1.py", 3)
    _commit(tmp_path, "f2.py", 4)
    _git(tmp_path, "checkout", "main")
    _commit(tmp_path, "m3.py", 5)
    _git(
        tmp_path,
        "merge",
        "--no-ff",
        "-m",
        "merge",
        "feature",
        when=_BASE_TIME + 6 * _DAY,
    )
    return tmp_path


def _files(diff: str) -> set[str]:
    return {
        line.split(" b/", 1)[1]
        for line in diff.splitlines()
        if line.startswith("diff --git ")
    }


async def test_last_n_matches_head_tilde_n_on_linear_history(linear_repo: Path):
    commit_range = await resolve_last_n_commits_range(5, linear_repo)

    assert commit_range == f"{_git(linear_repo, 'rev-parse', 'HEAD~5')}..HEAD"
    diff = await _diff_text(commit_range, linear_repo)
    for kept in ("c3.py", "c4.py", "c5.py", "c6.py", "c7.py"):
        assert f"b/{kept}" in diff
    for dropped in ("c0.py", "c1.py", "c2.py"):
        assert f"b/{dropped}" not in diff


async def test_last_one_commit_diffs_only_head(linear_repo: Path):
    commit_range = await resolve_last_n_commits_range(1, linear_repo)

    diff = await _diff_text(commit_range, linear_repo)
    assert "b/c7.py" in diff
    assert "b/c6.py" not in diff


async def test_last_n_reaches_commits_past_first_parent_depth(merge_repo: Path):
    # The old HEAD~5..HEAD does not exist: the first-parent chain is only
    # M, m3, c2, c1 deep, although 6 commits are reachable from HEAD.
    with pytest.raises(AssertionError, match="bad revision|unknown revision"):
        _git(merge_repo, "diff", "HEAD~5..HEAD")

    commit_range = await resolve_last_n_commits_range(5, merge_repo)

    diff = await _diff_text(commit_range, merge_repo)
    # Five of the six commits, with the root commit c1 as the only boundary:
    # exactly those five commits' changes, nothing from outside the window.
    assert _files(diff) == {"c2.py", "f1.py", "f2.py", "m3.py"}


async def test_window_with_two_boundary_commits_uses_their_common_ancestor(
    merge_repo: Path,
):
    # The first three commits in topological order leave two boundary commits
    # (f1 on the feature branch, c2 on main). The range then runs from their
    # common ancestor and covers every commit above it.
    commit_range = await resolve_last_n_commits_range(3, merge_repo)

    assert commit_range == f"{_git(merge_repo, 'rev-parse', 'main~2')}..HEAD"
    assert _files(await _diff_text(commit_range, merge_repo)) == {
        "f1.py",
        "f2.py",
        "m3.py",
    }


async def test_window_ignores_commit_dates_when_they_disagree_with_the_graph(
    tmp_path: Path,
):
    # The feature branch is dated after the merge (clock skew). The window is
    # chosen by topology, so every commit of it is covered and nothing older
    # than the branch point leaks in.
    _init(tmp_path)
    _commit(tmp_path, "a.py", 1)
    _commit(tmp_path, "b.py", 2)
    _git(tmp_path, "checkout", "-b", "feature")
    _commit(tmp_path, "f1.py", 30)
    _commit(tmp_path, "f2.py", 40)
    _git(tmp_path, "checkout", "main")
    _commit(tmp_path, "m1.py", 5)
    _git(
        tmp_path,
        "merge",
        "--no-ff",
        "-m",
        "M",
        "feature",
        when=_BASE_TIME + 6 * _DAY,
    )

    commit_range = await resolve_last_n_commits_range(3, tmp_path)

    # Window {M, m1, f2} has two boundary commits (b and f1); the range runs
    # from their common ancestor b, so f1 comes along with the window.
    assert _files(await _diff_text(commit_range, tmp_path)) == {
        "f1.py",
        "f2.py",
        "m1.py",
    }


async def test_shallow_clone_boundary_is_not_treated_as_the_root_commit(
    linear_repo: Path, tmp_path: Path
):
    clone = tmp_path / "shallow"
    result = subprocess.run(
        ["git", "clone", "-q", "--depth", "3", linear_repo.as_uri(), str(clone)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        commit_range = await resolve_last_n_commits_range(50, clone)
    finally:
        logger.remove(sink)

    # c5 is the edge of the fetched history; its own changes are unknown, so
    # only the commits after it appear, not the whole c5 snapshot.
    assert _files(await _diff_text(commit_range, clone)) == {"c6.py", "c7.py"}
    assert any("shallow" in m for m in messages)


@pytest.mark.parametrize("object_format", ["sha1", "sha256"])
async def test_last_n_above_history_clamps_to_full_history_with_one_warning(
    tmp_path: Path, object_format: str
):
    _init(tmp_path, object_format)
    for step in range(3):
        _commit(tmp_path, f"c{step}.py", step)
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        commit_range = await resolve_last_n_commits_range(1400, tmp_path)
    finally:
        logger.remove(sink)

    assert commit_range.endswith("..HEAD")
    diff = await _diff_text(commit_range, tmp_path)
    for name in ("c0.py", "c1.py", "c2.py"):
        assert f"b/{name}" in diff
    warnings = [m for m in messages if "last_n_commits" in m]
    assert len(warnings) == 1
    assert "1400" in warnings[0]
    assert "3 commits" in warnings[0]


async def test_last_n_equal_to_history_includes_root_commit_without_warning(
    linear_repo: Path,
):
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        commit_range = await resolve_last_n_commits_range(8, linear_repo)
    finally:
        logger.remove(sink)

    assert "c0.py" in await _diff_text(commit_range, linear_repo)
    assert not [m for m in messages if "last_n_commits" in m]


async def test_repository_without_commits_names_the_parameter(tmp_path: Path):
    _init(tmp_path)

    with pytest.raises(ValueError, match="last_n_commits"):
        await resolve_last_n_commits_range(3, tmp_path)


async def test_mcp_resolver_uses_the_real_range_for_last_n_commits(merge_repo: Path):
    # Unmocked wiring: tools._resolve_commit_range -> the repo found from the
    # DB path -> resolve_last_n_commits_range.
    from types import SimpleNamespace

    from chunkhound.mcp_server.tools import _resolve_commit_range

    db_dir = merge_repo / ".chunkhound"
    db_dir.mkdir()
    services = SimpleNamespace(provider=SimpleNamespace(db_path=db_dir / "db"))

    commit_range = await _resolve_commit_range(None, None, 5, services)  # type: ignore[arg-type]

    assert commit_range == await resolve_last_n_commits_range(5, merge_repo)
    assert _files(await _diff_text(commit_range, merge_repo)) == {
        "c2.py",
        "f1.py",
        "f2.py",
        "m3.py",
    }


async def test_mcp_resolver_finds_repo_root_from_db_in_subdirectory(
    merge_repo: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch
):
    # The DB can live below the repo root (--db flows). The repo must be found
    # by walking up from the DB path, not from the process cwd.
    from types import SimpleNamespace

    from chunkhound.mcp_server.tools import _resolve_commit_range

    db_dir = merge_repo / "data" / "index"
    db_dir.mkdir(parents=True)
    services = SimpleNamespace(provider=SimpleNamespace(db_path=db_dir / "db"))
    monkeypatch.chdir(tmp_path_factory.mktemp("unrelated_cwd"))

    commit_range = await _resolve_commit_range(None, None, 5, services)  # type: ignore[arg-type]

    assert commit_range == await resolve_last_n_commits_range(5, merge_repo)
