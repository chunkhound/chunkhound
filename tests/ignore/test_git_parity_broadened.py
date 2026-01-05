from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path


def run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.safecrlf=false",
            *args,
        ],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def git_ignored(paths: Iterable[str], repo: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for p in paths:
        proc = run_git(["check-ignore", "-q", "--no-index", p], repo)
        out[p] = proc.returncode == 0
    return out


def engine_ignored(paths: Iterable[str], repo: Path) -> dict[str, bool]:
    from chunkhound.utils.ignore_engine import build_ignore_engine  # type: ignore

    eng = build_ignore_engine(
        root=repo, sources=["gitignore"], chignore_file=".chignore", config_exclude=None
    )
    out: dict[str, bool] = {}
    for p in paths:
        out[p] = eng.matches(repo / p, is_dir=False) is not None
    return out


def test_root_dir_variants(tmp_path: Path) -> None:
    repo = tmp_path
    assert run_git(["init"], repo).returncode == 0

    (repo / ".gitignore").write_text(
        "\n".join(["dir/", "**/dir/", "/exact/", "dir/file", "**/deep/file"]) + "\n"
    )

    # layout
    (repo / "dir").mkdir(parents=True, exist_ok=True)
    (repo / "dir" / "a.txt").write_text("x")
    (repo / "sub" / "dir").mkdir(parents=True, exist_ok=True)
    (repo / "sub" / "dir" / "b.txt").write_text("x")
    (repo / "exact").mkdir(parents=True, exist_ok=True)
    (repo / "exact" / "x.txt").write_text("x")
    (repo / "dir").mkdir(parents=True, exist_ok=True)
    (repo / "dir" / "file").write_text("x")
    (repo / "sub" / "dir" / "file").write_text("x")
    (repo / "sub" / "deep" / "dir").mkdir(parents=True, exist_ok=True)
    (repo / "sub" / "deep" / "file").write_text("x")

    rels = [
        "dir/a.txt",  # dir/ should ignore
        "sub/dir/b.txt",  # dir/ should NOT ignore, **/dir/ should ignore
        "exact/x.txt",  # /exact/ should ignore
        "dir/file",  # dir/file should ignore
        "sub/dir/file",  # dir/file should NOT ignore (anchored), **/deep/file should ignore only deep/file
        "sub/deep/file",  # **/deep/file should ignore
    ]

    gmap = git_ignored(rels, repo)
    emap = engine_ignored(rels, repo)
    assert emap == gmap


def test_subdir_gitignore_anchoring(tmp_path: Path) -> None:
    repo = tmp_path
    assert run_git(["init"], repo).returncode == 0

    (repo / "sub").mkdir(parents=True, exist_ok=True)
    (repo / "sub" / ".gitignore").write_text("\n".join(["work/", "dir/file"]) + "\n")

    # create paths
    (repo / "sub" / "work").mkdir(parents=True, exist_ok=True)
    (repo / "sub" / "work" / "a.txt").write_text("x")
    (repo / "sub" / "dir").mkdir(parents=True, exist_ok=True)
    (repo / "sub" / "dir" / "file").write_text("x")
    (repo / "work").mkdir(parents=True, exist_ok=True)
    (repo / "work" / "a.txt").write_text("x")
    (repo / "dir").mkdir(parents=True, exist_ok=True)
    (repo / "dir" / "file").write_text("x")

    rels = [
        "sub/work/a.txt",  # should be ignored (anchored to sub)
        "work/a.txt",  # should NOT be ignored (rule is in sub)
        "sub/dir/file",  # should be ignored (anchored to sub)
        "dir/file",  # should NOT be ignored
    ]

    gmap = git_ignored(rels, repo)
    emap = engine_ignored(rels, repo)
    assert emap == gmap


def test_docs_recursive_build(tmp_path: Path) -> None:
    repo = tmp_path
    assert run_git(["init"], repo).returncode == 0
    (repo / ".gitignore").write_text("docs/**/build/\n")

    # Create paths
    (repo / "docs" / "build").mkdir(parents=True, exist_ok=True)
    (repo / "docs" / "build" / "x.txt").write_text("x")
    (repo / "docs" / "a" / "b" / "build").mkdir(parents=True, exist_ok=True)
    (repo / "docs" / "a" / "b" / "build" / "y.txt").write_text("y")
    (repo / "docs2" / "build").mkdir(parents=True, exist_ok=True)
    (repo / "docs2" / "build" / "z.txt").write_text("z")

    rels = [
        "docs/build/x.txt",  # ignored
        "docs/a/b/build/y.txt",  # ignored
        "docs2/build/z.txt",  # not ignored (outside docs)
    ]

    gmap = git_ignored(rels, repo)
    emap = engine_ignored(rels, repo)
    assert emap == gmap
