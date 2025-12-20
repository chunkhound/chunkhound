"""IgnoreEngine: central exclusion logic with gitwildmatch semantics.

Initial implementation supports root-level .gitignore files
via the `pathspec` library using gitwildmatch patterns. This is sufficient to
make the initial tests pass; we will extend it to per-directory inheritance
and richer rule origins in follow-up steps.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Tuple
from loguru import logger

try:
    from pathspec import PathSpec
    from pathspec.patterns.gitwildmatch import GitWildMatchPattern
except Exception as e:  # pragma: no cover - import error surfaced at runtime
    PathSpec = None  # type: ignore
    GitWildMatchPattern = None  # type: ignore


@dataclass
class MatchInfo:
    matched: bool
    source: Optional[Path] = None
    pattern: Optional[str] = None


class IgnoreEngine:
    def __init__(self, root: Path, compiled_specs: list[tuple[Path, "PathSpec"]]):
        self.root = root.resolve()
        self._compiled_specs = compiled_specs

    def matches(self, path: Path, is_dir: bool) -> Optional[MatchInfo]:
        # Normalize to root-relative POSIX path
        try:
            rel = path.resolve().relative_to(self.root)
        except Exception:
            rel = path.resolve()
        rel_str = rel.as_posix()

        # Evaluate specs in precedence order; first match wins
        for src, spec in self._compiled_specs:
            if spec.match_file(rel_str) or (is_dir and spec.match_file(rel_str + "/")):
                return MatchInfo(matched=True, source=src, pattern=None)
        return None


def _compile_gitwildmatch(patterns: Iterable[str]) -> "PathSpec":
    if PathSpec is None or GitWildMatchPattern is None:
        raise RuntimeError(
            "pathspec is required for IgnoreEngine; please add dependency 'pathspec'"
        )
    return PathSpec.from_lines(GitWildMatchPattern, patterns)


def _collect_global_gitignore_patterns() -> list[str]:
    """Collect patterns from the global gitignore file.

    Reads core.excludesFile from git config or checks default locations.
    Global patterns apply at any directory level (no transformation needed).
    """
    from chunkhound.utils.git_safe import get_global_excludes_file

    global_file = get_global_excludes_file()
    if not global_file:
        return []

    try:
        lines = global_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:
        logger.debug(f"Failed to read global gitignore {global_file}: {e}")
        return []

    patterns: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Global patterns match anywhere, so prefix with **/ if not already anchored
        if line.startswith("/"):
            # Anchored to root - keep as is (rare in global gitignore)
            patterns.append(line)
        elif line.startswith("**/") or line.startswith("!"):
            # Already has recursive prefix or is negation
            patterns.append(line)
        else:
            # Make pattern match anywhere in tree
            patterns.append(f"**/{line}")
    return patterns


def build_ignore_engine(
    root: Path,
    sources: list[str],
    chignore_file: str = ".chignore",
    config_exclude: Optional[Iterable[str]] = None,
) -> IgnoreEngine:
    """Build an IgnoreEngine for the given root and sources.

    Currently supports:
    - gitignore: uses only the root-level .gitignore file
    - config: uses provided glob-like patterns (gitwildmatch semantics)

    Note: Global gitignore (core.excludesFile) is NOT loaded here because the
    IgnoreEngine cannot properly handle cross-file negations. Global patterns
    are applied via git pathspec pushdown in IndexingCoordinator instead.
    """
    compiled: list[tuple[Path, PathSpec]] = []
    root = root.resolve()

    # Always enforce config_exclude (default excludes) regardless of sources
    if config_exclude:
        compiled.append((root, _compile_gitwildmatch(config_exclude)))

    for src in sources:
        if src == "gitignore":
            # Collect and transform .gitignore rules across the tree to root-relative patterns
            pre_spec = None
            if config_exclude:
                pre_spec = _compile_gitwildmatch(config_exclude)
            pats = _collect_gitignore_patterns(root, pre_spec)
            if pats:
                compiled.append((root / ".gitignore", _compile_gitwildmatch(pats)))
        # "chignore" source is no longer supported.
        elif src == "config":
            pats = list(config_exclude or [])
            if pats:
                compiled.append((root, _compile_gitwildmatch(pats)))

    return IgnoreEngine(root, compiled)


def _collect_gitignore_patterns(root: Path, pre_exclude_spec: Optional["PathSpec"] = None) -> list[str]:
    """Return root-relative gitwildmatch patterns transformed from .gitignore files.

    We walk the directory tree top-down so that root patterns appear before
    child directory patterns; last match still wins in PathSpec.
    """
    out: list[str] = []
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dpath = Path(dirpath)
        # Prune excluded subtrees early based on config excludes (e.g., node_modules)
        if pre_exclude_spec is not None:
            rel_base = "." if dpath == root else dpath.relative_to(root).as_posix()
            # Mutate dirnames in-place to prevent descending
            to_remove = []
            for dn in dirnames:
                child = dn if rel_base == "." else f"{rel_base}/{dn}"
                if pre_exclude_spec.match_file(child) or pre_exclude_spec.match_file(child + "/"):
                    to_remove.append(dn)
            for dn in to_remove:
                dirnames.remove(dn)
        gi = dpath / ".gitignore"
        if not gi.exists():
            continue
        try:
            lines = gi.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        rel_from_root = dpath.relative_to(root)
        dir_rel = "." if str(rel_from_root) == "." else rel_from_root.as_posix()
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            out.extend(_transform_gitignore_line(dir_rel, line))
    return out


def _detect_repo_roots(root: Path, pre_exclude_spec: Optional["PathSpec"] = None) -> list[Path]:
    """Detect Git repository roots under root by looking for .git dir or file.

    Prunes excluded subtrees using pre_exclude_spec (e.g., node_modules) to
    avoid unnecessary traversal.
    """
    roots: list[Path] = []
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dpath = Path(dirpath)

        # Prune excluded dirs
        if pre_exclude_spec is not None:
            rel_base = "." if dpath == root else dpath.relative_to(root).as_posix()
            to_remove = []
            for dn in dirnames:
                child = dn if rel_base == "." else f"{rel_base}/{dn}"
                if pre_exclude_spec.match_file(child) or pre_exclude_spec.match_file(child + "/"):
                    to_remove.append(dn)
            for dn in to_remove:
                dirnames.remove(dn)

        # Repo root if .git dir exists or .git file exists (submodule)
        if (dpath / ".git").is_dir() or (dpath / ".git").is_file():
            roots.append(dpath)
    # Sort deepest first for nearest-ancestor selection convenience later
    roots.sort(key=lambda p: len(p.as_posix()))
    return roots


class RepoAwareIgnoreEvaluator:
    """Repo-boundary aware evaluator that selects per-repo engines by path.

    - For a path inside a detected repo root, use only that repo's engine.
    - For paths outside any repo, use a workspace-scoped engine.
    - Config excludes are compiled into each engine and used to prune during
      .gitignore collection.
    """

    def __init__(
        self,
        workspace_root: Path,
        repo_roots: list[Path],
        sources: list[str],
        chignore_file: str,
        config_exclude: Optional[Iterable[str]] = None,
        workspace_root_only_gitignore: bool = False,
    ) -> None:
        self.root = workspace_root.resolve()
        self.repo_roots = sorted([p.resolve() for p in repo_roots], key=lambda p: len(p.as_posix()), reverse=True)
        self.sources = sources
        self.chignore_file = chignore_file
        self.config_exclude = list(config_exclude or [])

        # Build per-repo engines
        self._per_repo: Dict[Path, IgnoreEngine] = {}
        for rr in self.repo_roots:
            self._per_repo[rr] = build_ignore_engine(rr, sources, chignore_file, self.config_exclude)
        # Workspace engine for non-repo areas
        if workspace_root_only_gitignore:
            # Workspace (non-repo) overlay: honor .gitignore files under the root
            # using the same collection logic as repo mode (root + nested files).
            # This preserves Git anchoring semantics and nested inheritance even
            # when the workspace itself is not a Git repository.
            compiled: list[tuple[Path, PathSpec]] = []
            if self.config_exclude:
                compiled.append((self.root, _compile_gitwildmatch(self.config_exclude)))
            try:
                pre = _compile_gitwildmatch(self.config_exclude) if self.config_exclude else None
            except Exception:
                pre = None
            try:
                pats = _collect_gitignore_patterns(self.root, pre)
            except Exception:
                pats = []
            if pats:
                compiled.append((self.root / ".gitignore", _compile_gitwildmatch(pats)))
            self._workspace_engine = IgnoreEngine(self.root, compiled)
        else:
            # Default: do NOT apply workspace .gitignore at all; only config_exclude
            compiled: list[tuple[Path, PathSpec]] = []
            if self.config_exclude:
                compiled.append((self.root, _compile_gitwildmatch(self.config_exclude)))
            self._workspace_engine = IgnoreEngine(self.root, compiled)

    def _nearest_repo(self, path: Path) -> Optional[Path]:
        p = path.resolve()
        for rr in self.repo_roots:
            try:
                p.relative_to(rr)
                return rr
            except Exception:
                continue
        return None

    def matches(self, path: Path, is_dir: bool) -> Optional[MatchInfo]:
        rr = self._nearest_repo(path)
        if rr is not None:
            return self._per_repo[rr].matches(path, is_dir)
        return self._workspace_engine.matches(path, is_dir)


def build_repo_aware_ignore_engine(
    root: Path,
    sources: list[str],
    chignore_file: str = ".chignore",
    config_exclude: Optional[Iterable[str]] = None,
    backend: str = "python",
    workspace_root_only_gitignore: Optional[bool] = None,
) -> RepoAwareIgnoreEvaluator:
    pre_spec = _compile_gitwildmatch(config_exclude or []) if (config_exclude) else None
    repo_roots = _detect_repo_roots(root, pre_spec)
    if backend == "libgit2":
        eng = _try_build_libgit2_repo_aware(root, repo_roots, sources, chignore_file, config_exclude)
        if eng is not None:
            return eng
    # Determine workspace-root-only behavior.
    # Priority:
    # 1) Explicit parameter from config
    # 2) If sources include gitignore and there are NO repos in the workspace,
    #    default to True to honor a root .gitignore for non‑repo trees
    # 3) Legacy ENV override (kept for backward compatibility)
    if workspace_root_only_gitignore is not None:
        wr_only = bool(workspace_root_only_gitignore)
    else:
        if ("gitignore" in (sources or [])) and (not repo_roots):
            wr_only = True
        else:
            try:
                wr_only = os.environ.get("CHUNKHOUND_INDEXING__WORKSPACE_GITIGNORE_NONREPO", "").strip() not in ("", "0", "false", "no")
            except Exception:
                wr_only = False
    return RepoAwareIgnoreEvaluator(
        root,
        repo_roots,
        sources,
        chignore_file,
        config_exclude,
        workspace_root_only_gitignore=wr_only,
    )



def _transform_gitignore_line(dir_rel: str, line: str) -> list[str]:
    """Transform a .gitignore pattern from a directory into root-relative patterns.

    Handles negation (!), anchored (/), and directory-only (trailing /) forms by
    emitting patterns that constrain the match to the originating subtree.
    """
    neg = False
    if line.startswith("!"):
        neg = True
        line = line[1:]

    # Directory-only patterns (ending with '/')
    is_dir_pat = line.endswith("/")
    if is_dir_pat:
        line = line[:-1]

    # Build base (relative to root) for anchored vs unanchored
    parts: list[str] = []

    def add(p: str) -> None:
        if is_dir_pat:
            p = f"{p}/**"
        if neg:
            p = "!" + p
        parts.append(p)

    # Resolve directory prefix with Git semantics
    # Rules (simplified from gitignore docs):
    # - Leading '/' anchors to the directory containing the .gitignore.
    # - A pattern that contains a '/' (after trimming trailing '/') is anchored to that directory.
    # - A pattern without any '/' matches in any directory under the .gitignore directory.
    core = line
    has_slash = "/" in core

    if dir_rel == ".":
        # Root-level .gitignore
        if core.startswith("/"):
            # Anchored to root; keep leading slash to prevent filename-only matches
            add(core)
        elif has_slash:
            # Contains '/', anchored to root
            add(core)
        else:
            # No '/', match anywhere (root and nested)
            add(core)
            add(f"**/{core}")
    else:
        # Subdirectory .gitignore
        if core.startswith("/"):
            # Anchored to this directory
            add(f"{dir_rel}/{core[1:]}")
        elif has_slash:
            # Contains '/', anchored to this directory
            add(f"{dir_rel}/{core}")
        else:
            # No '/', match anywhere under this directory (direct and nested)
            add(f"{dir_rel}/{core}")
            add(f"{dir_rel}/**/{core}")

    return parts


def detect_repo_roots(root: Path, config_exclude: Optional[Iterable[str]] = None) -> list[Path]:
    """Public helper to detect repo roots under a workspace root.

    Applies pruning using config_exclude (gitwildmatch semantics) to avoid
    descending into heavy trees (e.g., node_modules) while scanning.
    """
    pre_spec = _compile_gitwildmatch(config_exclude or []) if config_exclude else None
    return _detect_repo_roots(root, pre_spec)


def build_repo_aware_ignore_engine_from_roots(
    root: Path,
    repo_roots: list[Path],
    sources: list[str],
    chignore_file: str = ".chignore",
    config_exclude: Optional[Iterable[str]] = None,
    backend: str = "python",
    workspace_root_only_gitignore: Optional[bool] = None,
) -> RepoAwareIgnoreEvaluator:
    """Build a repo-aware evaluator from a precomputed list of repo roots.

    Avoids re-scanning the entire workspace per worker when running in parallel.
    """
    if backend == "libgit2":
        eng = _try_build_libgit2_repo_aware(root, repo_roots, sources, chignore_file, config_exclude)
        if eng is not None:
            return eng
    if workspace_root_only_gitignore is None:
        try:
            wr_only = os.environ.get("CHUNKHOUND_INDEXING__WORKSPACE_GITIGNORE_NONREPO", "").strip() not in ("", "0", "false", "no")
        except Exception:
            wr_only = False
    else:
        wr_only = bool(workspace_root_only_gitignore)
    return RepoAwareIgnoreEvaluator(root, repo_roots, sources, chignore_file, config_exclude, workspace_root_only_gitignore=wr_only)


# --------------------------- Optional libgit2 backend ---------------------------
class RepoAwareLibgit2Evaluator:
    """Repo-aware evaluator using libgit2 (pygit2) for gitignore decisions.

    Falls back to Python engine semantics if pygit2 isn't available or a call fails.
    Always applies config_exclude (pathspec) first as a hard exclude layer.
    """

    def __init__(
        self,
        workspace_root: Path,
        repo_roots: list[Path],
        sources: list[str],
        chignore_file: str,
        config_exclude: Optional[Iterable[str]] = None,
    ) -> None:
        self.root = workspace_root.resolve()
        self.repo_roots = sorted([p.resolve() for p in repo_roots], key=lambda p: len(p.as_posix()), reverse=True)
        self.sources = sources
        self.chignore_file = chignore_file
        self.config_exclude = list(config_exclude or [])

        # Precompile config_exclude with pathspec for fast hard excludes
        self._cfg_spec = _compile_gitwildmatch(self.config_exclude) if self.config_exclude else None

        # Open libgit2 repos
        self._repos: Dict[Path, object] = {}
        try:
            import pygit2  # type: ignore
            self._pygit2 = pygit2
        except Exception:
            self._pygit2 = None

        if self._pygit2 is not None:
            for rr in self.repo_roots:
                try:
                    # pygit2 accepts workdir path (not .git) for Repository()
                    self._repos[rr] = self._pygit2.Repository(str(rr))
                except Exception:
                    # Ignore repos we can't open; they'll be handled by cfg spec only
                    continue

    def _nearest_repo(self, path: Path) -> Optional[Path]:
        p = path.resolve()
        for rr in self.repo_roots:
            try:
                p.relative_to(rr)
                return rr
            except Exception:
                continue
        return None

    def _cfg_excluded(self, rel: str, is_dir: bool) -> bool:
        if self._cfg_spec is None:
            return False
        return self._cfg_spec.match_file(rel) or (is_dir and self._cfg_spec.match_file(rel + "/"))

    def matches(self, path: Path, is_dir: bool) -> Optional[MatchInfo]:
        # Hard exclude via config_exclude first
        try:
            rel_cfg = path.resolve().relative_to(self.root).as_posix()
        except Exception:
            rel_cfg = path.name
        if self._cfg_excluded(rel_cfg, is_dir):
            return MatchInfo(matched=True, source=self.root, pattern=None)

        rr = self._nearest_repo(path)
        if rr is None or self._pygit2 is None:
            return None
        repo = self._repos.get(rr)
        if repo is None:
            return None

        # Compute repo-relative path
        try:
            rel = path.resolve().relative_to(rr).as_posix()
        except Exception:
            rel = path.name

        # Try common pygit2 ignore API methods (varies by version)
        try:
            fn = getattr(repo, "is_path_ignored", None) or getattr(repo, "path_is_ignored", None)
            if callable(fn):
                ign = bool(fn(rel if not is_dir else (rel + "/")))
                if ign:
                    return MatchInfo(matched=True, source=rr, pattern=None)
        except Exception:
            return None
        return None


def _try_build_libgit2_repo_aware(
    root: Path,
    repo_roots: list[Path],
    sources: list[str],
    chignore_file: str,
    config_exclude: Optional[Iterable[str]] = None,
) -> Optional[RepoAwareLibgit2Evaluator]:
    # Warn exactly once per process when we cannot honor libgit2 backend
    global _LIBGIT2_WARNED
    try:
        _LIBGIT2_WARNED
    except NameError:
        _LIBGIT2_WARNED = False  # type: ignore[var-annotated]
    # Only attempt when gitignore is part of sources
    if "gitignore" not in (sources or []):
        return None
    try:
        import pygit2  # noqa: F401
    except Exception:
        if not _LIBGIT2_WARNED and not os.environ.get("CHUNKHOUND_MCP_MODE"):
            logger.warning(
                "gitignore_backend=libgit2 requested but pygit2 is not available; falling back to python backend"
            )
            _LIBGIT2_WARNED = True  # type: ignore[assignment]
        return None
    try:
        return RepoAwareLibgit2Evaluator(root, repo_roots, sources, chignore_file, config_exclude)
    except Exception:
        if not _LIBGIT2_WARNED and not os.environ.get("CHUNKHOUND_MCP_MODE"):
            logger.warning(
                "gitignore_backend=libgit2 requested but initialization failed; falling back to python backend"
            )
            _LIBGIT2_WARNED = True  # type: ignore[assignment]
        return None


__all__ = [
    "IgnoreEngine",
    "MatchInfo",
    "build_ignore_engine",
    "build_repo_aware_ignore_engine",
    "build_repo_aware_ignore_engine_from_roots",
    "detect_repo_roots",
]
