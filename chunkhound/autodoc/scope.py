from __future__ import annotations

from pathlib import Path

from chunkhound.autodoc.models import HydeConfig


def iter_scope_files(scope_path: Path, project_root: Path) -> list[str]:
    """Return normalized relative file paths within scope, skipping noise dirs."""
    ignore_dirs = {
        ".git",
        ".chunkhound",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
    }
    file_paths: list[str] = []
    for path in scope_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignore_dirs for part in path.parts):
            continue
        try:
            rel = path.relative_to(project_root)
        except ValueError:
            continue
        file_paths.append(str(rel).replace("\\", "/"))
    return file_paths


def collect_scope_files(
    *,
    scope_path: Path,
    project_root: Path,
    hyde_cfg: HydeConfig,
) -> list[str]:
    """Collect file paths within the scope, relative to project_root.

    This is filesystem-only and intentionally lightweight.
    """
    file_paths = iter_scope_files(scope_path, project_root)

    try:
        if file_paths:
            from subprocess import run

            git_root: Path | None = None
            for candidate in (scope_path, project_root):
                proc_root = run(
                    ["git", "rev-parse", "--show-toplevel"],
                    cwd=str(candidate),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if proc_root.returncode == 0 and proc_root.stdout.strip():
                    git_root = Path(proc_root.stdout.strip()).resolve()
                    break

            if git_root is not None:
                rel_for_git: list[str] = []
                rel_to_original: dict[str, str] = {}

                for rel in file_paths:
                    abs_path = (project_root / rel).resolve()
                    try:
                        git_rel = abs_path.relative_to(git_root).as_posix()
                    except ValueError:
                        continue
                    rel_for_git.append(git_rel)
                    rel_to_original[git_rel] = rel

                if rel_for_git:
                    proc = run(
                        ["git", "check-ignore", "--stdin"],
                        cwd=str(git_root),
                        input="\n".join(rel_for_git),
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    if proc.returncode in (0, 1):
                        ignored_git_rel = {
                            line.strip()
                            for line in proc.stdout.splitlines()
                            if line.strip()
                        }
                        if ignored_git_rel:
                            ignored_original = {
                                rel_to_original[p]
                                for p in ignored_git_rel
                                if p in rel_to_original
                            }
                            file_paths = [
                                p for p in file_paths if p not in ignored_original
                            ]
    except Exception:
        pass

    if len(file_paths) > hyde_cfg.max_scope_files:
        file_paths = file_paths[: hyde_cfg.max_scope_files]

    return file_paths
