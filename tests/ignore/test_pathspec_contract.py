"""ChunkHound reads pathspec's compiled patterns to decide re-includes as git does.

pathspec has no upper version bound. If its pattern objects change shape, the
rules degrade to pathspec's own matching (no pruning, no git-exact
re-includes) and log a warning. These tests use the installed pathspec, so
they guard whichever version is installed (0.12 and 1.x spell a pattern's
regex differently). The first compares the engine with git on a layout that
pathspec alone decides differently; the second checks that a plain file
pattern leaves ignored directories pruned.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from chunkhound.utils.ignore_engine import (
    build_ignore_engine,
    build_repo_aware_ignore_engine,
)

# git ignores dir/keep.py (a file under an excluded directory cannot be
# re-included) and contents/sub/x.py (!contents/sub/ re-includes only the
# directory); pathspec alone re-includes both.
_GITIGNORE = "dir/\n!dir/keep.py\ncontents/**\n!contents/sub/\n"
_FILES = ["dir/keep.py", "contents/sub/x.py", "contents/top.py", "a.py"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
def test_ignore_engine_agrees_with_git_where_pathspec_alone_does_not(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".gitignore").write_text(_GITIGNORE, encoding="utf-8")
    for rel in _FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("X = 1\n", encoding="utf-8")

    # NUL-separated bytes: text mode would write "\r\n" on Windows, and git
    # would then read each path with a trailing "\r".
    checked = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"],
        cwd=root,
        input="\0".join(_FILES).encode(),
        capture_output=True,
    )
    assert checked.returncode in (0, 1), checked.stderr
    git_ignored = set(checked.stdout.decode().split("\0")) - {""}
    # The layout does what it is for: git decides both re-include cases.
    assert git_ignored == {"dir/keep.py", "contents/sub/x.py", "contents/top.py"}

    engine = build_ignore_engine(root, ["gitignore"], ".chignore")
    engine_ignored = {
        rel for rel in _FILES if engine.matches(root / rel, is_dir=False) is not None
    }

    assert engine_ignored == git_ignored


def test_a_plain_file_pattern_keeps_ignored_directories_pruned(tmp_path: Path) -> None:
    """A .gitignore that names a file (Thumbs.db) beside node_modules/.

    Under pathspec 1.x such a line once stopped all directory pruning: the
    walks entered node_modules, while each file in it was still excluded.
    """
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / ".gitignore").write_text(
        "Thumbs.db\n*.log\nnode_modules/\n", encoding="utf-8"
    )
    engine = build_repo_aware_ignore_engine(
        root, ["gitignore"], ".chignore", ["**/dist/**"]
    )

    assert engine.matches(root / "node_modules", is_dir=True) is not None
    assert engine.matches(root / "pkg" / "node_modules", is_dir=True) is not None
    assert engine.matches(root / "web" / "dist", is_dir=True) is not None
    assert engine.matches(root / "src", is_dir=True) is None
    assert engine.matches(root / "src" / "Thumbs.db", is_dir=False) is not None
    assert engine.matches(root / "src" / "debug.log", is_dir=False) is not None
    assert engine.matches(root / "src" / "app.py", is_dir=False) is None
