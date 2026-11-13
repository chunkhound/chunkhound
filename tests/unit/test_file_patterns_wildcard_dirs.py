import subprocess
from pathlib import Path

import pytest

from chunkhound.utils.file_patterns import (
    should_exclude_path,
    should_include_file,
    walk_directory_tree,
)
from chunkhound.utils.git_discovery import list_repo_files_via_git


def test_exclude_double_star_segment_wildcard_venv():
    base = Path("/workspaces/project")
    p = base / ".venv-docling" / "lib" / "python3.11" / "site-packages" / "pkg" / "mod.py"
    patterns = ["**/.venv*/**"]
    assert should_exclude_path(p, base, patterns, {}) is True


def test_exclude_double_star_segment_exact_venv():
    base = Path("/workspaces/project")
    p = base / ".venv" / "lib" / "python3.11" / "site-packages" / "pkg" / "mod.py"
    patterns = ["**/.venv/**"]
    assert should_exclude_path(p, base, patterns, {}) is True


def test_exclude_directory_suffix_phar():
    base = Path("/workspaces/project")
    p = base / "vendor.phar" / "data" / "file.txt"
    patterns = ["**/*.phar/**"]
    assert should_exclude_path(p, base, patterns, {}) is True


@pytest.mark.asyncio
async def test_walk_directory_tree_prunes_wildcard_dir_segments(tmp_path: Path):
    # Create structure
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "a.py").write_text("print('ok')\n", encoding="utf-8")

    # .venv-docling subtree (should be pruned)
    venv_pkg = tmp_path / ".venv-docling" / "lib" / "python3.11" / "site-packages" / "pkg"
    venv_pkg.mkdir(parents=True)
    (venv_pkg / "b.py").write_text("# venv\n", encoding="utf-8")

    # *.phar directory (should be pruned)
    phar_dir = tmp_path / "vendor.phar" / "data"
    phar_dir.mkdir(parents=True)
    (phar_dir / "c.py").write_text("# phar\n", encoding="utf-8")

    # site-packages directly (should be pruned)
    sp_dir = tmp_path / "site-packages" / "other"
    sp_dir.mkdir(parents=True)
    (sp_dir / "d.py").write_text("# sp\n", encoding="utf-8")

    include = ["**/*.py"]
    exclude = ["**/.venv*/**", "**/*.phar/**", "**/site-packages/**"]

    files, _ = walk_directory_tree(
        start_path=tmp_path,
        root_directory=tmp_path,
        patterns=include,
        exclude_patterns=exclude,
        parent_gitignores={},
        use_inode_ordering=False,
    )

    rels = sorted([f.relative_to(tmp_path).as_posix() for f in files])
    # Only the src/a.py file should be present
    assert rels == ["src/a.py"], f"unexpected files discovered: {rels}"


@pytest.mark.parametrize(
    ("matcher", "pattern", "relative_path", "expected"),
    [
        (should_include_file, "src/**/*.py", "src/main.py", True),
        (should_include_file, "src/**/*.py", "src/pkg/mod.py", True),
        (should_include_file, "src/**/*.py", "other/main.py", False),
        (should_exclude_path, "**/.yarn/cache/**", ".yarn/cache/file.zip", True),
        (should_exclude_path, "**/.yarn/cache/**", "pkg/.yarn/cache/file.zip", True),
        (should_exclude_path, "**/public/bundles/**", "public/bundles/app.js", True),
        (should_exclude_path, "**/public/bundles/**", "public/bundles-x/app.js", False),
    ],
)
def test_directory_globs_match_named_files(
    matcher, pattern: str, relative_path: str, expected: bool
) -> None:
    base = Path("/workspaces/project")
    assert matcher(base / relative_path, base, [pattern], {}) is expected


def test_git_discovery_keeps_files_directly_under_an_anchored_include(tmp_path: Path):
    repo = tmp_path / "repo"
    for relative_path in ("src/main.py", "src/pkg/mod.py", "other/nope.py"):
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    files, _stats = list_repo_files_via_git(
        repo, repo, ["**/src/**/*.py"], config_excludes=[]
    )
    rels = sorted(path.relative_to(repo).as_posix() for path in files)
    assert rels == ["src/main.py", "src/pkg/mod.py"], rels
