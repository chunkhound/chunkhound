from __future__ import annotations

from pathlib import Path

from chunkhound.api.cli.utils.rich_output import RichOutputFormatter
from chunkhound.utils.git_safe import run_git


def _nearest_existing_dir(path: Path) -> Path | None:
    current = path
    try:
        while True:
            if current.exists():
                return current if current.is_dir() else current.parent
            if current.parent == current:
                return None
            current = current.parent
    except OSError:
        return None


def _git_repo_root(start_dir: Path) -> Path | None:
    try:
        result = run_git(
            ["rev-parse", "--show-toplevel"],
            cwd=start_dir,
            timeout_s=5.0,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    if not root:
        return None
    return Path(root)


def _git_is_ignored(*, repo_root: Path, path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    try:
        result = run_git(
            ["check-ignore", "-q", "--no-index", rel.as_posix()],
            cwd=repo_root,
            timeout_s=5.0,
        )
    except Exception:
        return False
    return result.returncode == 0


def maybe_warn_git_output_dir(
    output_dir: Path, formatter: RichOutputFormatter
) -> None:
    start_dir = _nearest_existing_dir(output_dir)
    if start_dir is None:
        return

    repo_root = _git_repo_root(start_dir)
    if repo_root is None:
        return

    try:
        output_dir.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return

    if _git_is_ignored(repo_root=repo_root, path=output_dir):
        return

    formatter.warning(
        "Output directory appears to be inside a git repo and is not ignored; "
        "generated docs may show up in `git status`. Consider adding it to "
        ".gitignore or writing to a git-ignored directory."
    )

