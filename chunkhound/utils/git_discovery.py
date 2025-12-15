"""Git-backed discovery helpers.

Enumerate files using `git ls-files` for speed and exact ignore semantics,
then apply ChunkHound include/exclude filters.

Design notes:
- We shell once per repo root (tracked + untracked non-ignored via --others --exclude-standard).
- We NUL-delimit (-z) to avoid issues with spaces and special chars.
- For non-repo directories, callers should fall back to the Python traversal.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from chunkhound.utils.file_patterns import (
    should_include_file,
    should_exclude_path,
    normalize_include_patterns,
)
from chunkhound.utils.git_safe import run_git, GitCommandError


def build_git_pathspecs(rel_prefix: str | None, include_patterns: Sequence[str]) -> list[str]:
    """Build a minimal set of Git :(glob) pathspecs from CH include patterns.

    We only push down simple, lossless patterns:
      - file extensions (e.g., **/*.py)
      - exact basenames (e.g., **/Makefile)
    Complex patterns (character classes, alternations, etc.) are ignored.
    """
    # Import summarizer lazily to avoid cycles
    from chunkhound.utils.file_patterns import _summarize_include_patterns  # type: ignore

    rel = (rel_prefix or "").strip("/")
    exts, names, _complex = _summarize_include_patterns(list(include_patterns))
    specs: list[str] = []

    def add(p: str) -> None:
        specs.append(f":(glob){p}")

    # Extensions → **/*.<ext>
    for ext in sorted(exts):
        if rel:
            add(f"{rel}/**/*{ext}")
        else:
            add(f"**/*{ext}")

    # Exact basenames → **/<name>
    for nm in sorted(names):
        if rel:
            add(f"{rel}/**/{nm}")
        else:
            add(f"**/{nm}")

    return specs


def _run_git_ls_files(repo_root: Path, pathspecs: list[str] | None = None) -> tuple[list[str], int, int]:
    """Return repo-relative paths from git ls-files (tracked + untracked non-ignored)."""
    repo_root = repo_root.resolve()
    # Tracked files
    tracked = []
    try:
        args = ["-C", str(repo_root), "ls-files", "-z"]
        if pathspecs:
            args += ["--", *pathspecs]
        res = run_git(args, cwd=repo_root, timeout_s=None)
        if res.returncode != 0:
            tracked = []
        else:
            tracked = [p for p in (res.stdout or "").split("\x00") if p]
    except GitCommandError:
        tracked = []
    # Untracked, non-ignored (exclude-standard honors .gitignore + core excludes)
    others = []
    try:
        args = ["-C", str(repo_root), "ls-files", "--others", "--exclude-standard", "-z"]
        if pathspecs:
            args += ["--", *pathspecs]
        res = run_git(args, cwd=repo_root, timeout_s=None)
        if res.returncode != 0:
            others = []
        else:
            others = [p for p in (res.stdout or "").split("\x00") if p]
    except GitCommandError:
        others = []
    # Deduplicate while preserving order (tracked typically first)
    seen = set()
    merged: list[str] = []
    for p in tracked + others:
        if p not in seen:
            seen.add(p)
            merged.append(p)
    return merged, len(tracked), len(others)


def list_repo_files_via_git(
    repo_root: Path,
    start_dir: Path,
    include_patterns: Sequence[str],
    config_excludes: Sequence[str] | None = None,
    pushdown: bool | None = None,
    filter_root: Path | None = None,
) -> tuple[list[Path], dict]:
    """Enumerate files under start_dir (inside repo_root) using git ls-files.

    - repo_root: absolute path to the Git working tree root.
    - start_dir: directory being indexed; must be repo_root or a subdirectory.
    - include_patterns: ChunkHound include globs.
    - config_excludes: ChunkHound config/default excludes (applied after git results).
    """
    repo_root = repo_root.resolve()
    start_dir = start_dir.resolve()
    rel_prefix: str | None = None
    try:
        rel_prefix = start_dir.relative_to(repo_root).as_posix()
        if rel_prefix in (".", ""):
            rel_prefix = None
    except Exception:
        # start_dir may be the repo_root itself, or outside; if outside we return empty
        if start_dir != repo_root:
            return []
        rel_prefix = None

    # Decide pushdown from env if not provided
    if pushdown is None:
        try:
            v = os.environ.get("CHUNKHOUND_INDEXING__GIT_PATHSPEC_PUSHDOWN", "")
            pushdown = (v.strip() != "0")
        except Exception:
            pushdown = True

    # Subtree restriction is included via rel_prefix. Build additional :(glob) pathspecs from includes.
    pathspecs: list[str] | None = None
    if pushdown:
        try:
            specs = build_git_pathspecs(rel_prefix, include_patterns)
            # Apply CAP to avoid excessive number of :(glob) specs
            cap_env = os.environ.get("CHUNKHOUND_INDEXING__GIT_PATHSPEC_CAP", "")
            try:
                cap = int(cap_env) if cap_env.strip() else 128
                if cap < 1:
                    cap = 128
            except Exception:
                cap = 128
            capped = False
            if specs and len(specs) > cap:
                # Fallback to subtree-only restriction to guarantee correctness
                pathspecs = [rel_prefix] if rel_prefix else None
                capped = True
            else:
                # If we produced any specs, use them; otherwise fall back to plain subtree pathspec
                if specs:
                    pathspecs = specs
                elif rel_prefix:
                    pathspecs = [rel_prefix]
        except Exception:
            # Fallback to subtree only on any error
            pathspecs = ([rel_prefix] if rel_prefix else None)
    else:
        pathspecs = ([rel_prefix] if rel_prefix else None)

    rel_paths, rows_tracked, rows_others = _run_git_ls_files(repo_root, pathspecs or None)
    if rel_prefix:
        rel_prefix_slash = rel_prefix + "/"
        rel_paths = [p for p in rel_paths if p == rel_prefix or p.startswith(rel_prefix_slash)]

    # Assume caller provides patterns already normalized where needed
    norm_includes = list(include_patterns)
    out: list[Path] = []
    pcache: dict[str, object] = {}
    # Evaluate includes/excludes relative to filter_root (CH root) when provided; otherwise start_dir
    # Don't resolve symlinks for filter base - use logical path for worktree support
    base_for_filters = filter_root or start_dir

    for rel in rel_paths:
        # Don't resolve symlinks - use logical path for worktree support
        abs_path = repo_root / rel
        # Filter to files that still exist as files (follows symlink for existence check only)
        try:
            if not abs_path.is_file():
                continue
        except Exception:
            continue

        # Apply ChunkHound config/default excludes on top of Git results
        if config_excludes and should_exclude_path(abs_path, base_for_filters, list(config_excludes), pcache):
            continue

        # Apply include patterns
        if should_include_file(abs_path, base_for_filters, list(norm_includes), pcache):
            out.append(abs_path)

    stats = {
        "git_rows_tracked": int(rows_tracked),
        "git_rows_others": int(rows_others),
        "git_rows_total": int(rows_tracked + rows_others),
        "git_pathspecs": int(len(pathspecs)) if pathspecs is not None else 0,
        "git_pushdown": bool(pushdown),
    }
    # Optionally surface whether CAP fallback was applied (best-effort)
    try:
        if "capped" in locals() and capped:
            stats["git_pathspecs_capped"] = True
    except Exception:
        pass
    return out, stats


__all__ = [
    "list_repo_files_via_git",
]
