"""Ignore handling and repo-root detection with gitwildmatch semantics."""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

try:
    from pathspec import PathSpec
    from pathspec.patterns.gitwildmatch import GitWildMatchPattern
except Exception:  # pragma: no cover - import error surfaced at runtime
    PathSpec = None  # type: ignore
    GitWildMatchPattern = None  # type: ignore


@dataclass
class MatchInfo:
    matched: bool
    source: Path | None = None
    pattern: str | None = None


# A transformed .gitignore line: the root-relative pattern, and whether the
# .gitignore wrote it as a directory pattern ("foo/", emitted as "foo/**").
_GitignoreLine = tuple[str, bool]


@dataclass(frozen=True)
class _GitignorePruneState:
    patterns: tuple[_GitignoreLine, ...]
    spec: PathSpec | None
    rules: _SpecRules | None = None


def _empty_gitignore_prune_state() -> _GitignorePruneState:
    return _GitignorePruneState(patterns=(), spec=None)


def _in_tree_path(path: Path, root: Path) -> Path:
    """Return the path ignore rules are evaluated on: where it sits under ``root``.

    A file reached through a symlink or junction inside the tree is matched by
    its location in the tree, as Git matches it, so ``node_modules/`` still
    excludes a linked package. Resolving first would match the link's target
    instead, which lies outside the tree, where no root-relative rule applies.
    A path that is not lexically under ``root`` (already resolved, or spelled
    differently) is resolved as before.
    """
    in_tree = Path(os.path.abspath(path))
    try:
        in_tree.relative_to(root)
        return in_tree
    except ValueError:
        return path.resolve()


_GLOB_CHARS = "*?[\\"


def _negation_scope(body: str) -> str:
    """Return the literal path prefix a negation pattern is confined to.

    ``""`` means it can match anywhere: an unanchored name (``keep.py``,
    ``*.md``), a ``**/`` pattern, or one that starts with a glob.
    """
    body = body.rstrip("/")
    if body.startswith("**/"):
        return ""
    if body.startswith("/"):
        body = body[1:]
    elif "/" not in body:
        return ""
    for index, char in enumerate(body):
        if char in _GLOB_CHARS:
            return body[:index]
    return body


# pathspec (0.12) ends a pattern's regex with a tail, the group named ps_d,
# that makes a match on a directory also match everything beneath it. Git
# matches a pattern against the path itself, so the exact form drops the tail.
_PS_D_OPTIONAL_TAIL = "(?:(?P<ps_d>/).*)?"
_PS_D_DIR_TAIL = "(?P<ps_d>/).*"
# The .gitignore transform writes a directory pattern "foo/" as "foo/**".
_DIR_CONTENTS_SUFFIX = "/**"
_DIR_DECISIONS_MAX = 65536
_rules_degraded_warned = False


class _PathspecContractError(RuntimeError):
    """pathspec's compiled patterns no longer look as this module expects."""


@dataclass(frozen=True)
class _Rule:
    """One pattern in the order it was compiled.

    ``regex`` is pathspec's: it matches a path or anything beneath it.
    ``exact`` matches the path itself only, as git matches a pattern; None for
    a pattern that matches nothing. A ``dir_only`` pattern applies only to
    directories.
    """

    regex: re.Pattern[str]
    exact: re.Pattern[str] | None
    include: bool
    dir_only: bool


@dataclass(frozen=True, eq=False)
class _SpecRules:
    """A compiled spec plus what git-exact decisions need from its patterns.

    ``negation_scopes`` holds the literal prefix each negation is confined to
    (see ``_negation_scope``). ``degraded`` is set when the patterns could not
    be read; decisions then fall back to pathspec alone and never prune.
    ``dir_decisions`` memoizes git's answer per directory: the rules never
    change, so an answer stays valid for the spec's lifetime.
    """

    spec: PathSpec
    rules: tuple[_Rule, ...]
    negations: tuple[re.Pattern[str], ...]
    negation_scopes: tuple[str, ...]
    degraded: bool = False
    dir_decisions: dict[str, bool] = field(default_factory=dict)


def _exact_regex(pathspec_regex: str) -> re.Pattern[str]:
    exact = pathspec_regex.replace(_PS_D_OPTIONAL_TAIL, "").replace(_PS_D_DIR_TAIL, "")
    if "ps_d" in exact:
        raise _PathspecContractError(f"unexpected pathspec regex {pathspec_regex!r}")
    return re.compile(exact)


def _build_spec_rules(spec: PathSpec, dir_only: Sequence[bool] | None) -> _SpecRules:
    patterns = list(spec.patterns)
    if dir_only is not None and len(dir_only) != len(patterns):
        raise _PathspecContractError(
            f"{len(patterns)} compiled patterns for {len(dir_only)} lines"
        )
    rules: list[_Rule] = []
    negations: list[re.Pattern[str]] = []
    scopes: list[str] = []
    for index, pattern in enumerate(patterns):
        include = getattr(pattern, "include", _PathspecContractError)
        regex = getattr(pattern, "regex", _PathspecContractError)
        source = getattr(pattern, "pattern", _PathspecContractError)
        if _PathspecContractError in (include, regex, source):
            raise _PathspecContractError(f"pattern object {pattern!r} lacks fields")
        if include is None:
            continue  # a blank or comment line
        if not isinstance(regex, re.Pattern) or not isinstance(source, str):
            raise _PathspecContractError(f"pattern object {pattern!r} changed shape")
        body = source[1:] if source.startswith("!") else source
        if dir_only is None:
            # Patterns as written (config excludes): "foo/" is dir-only.
            is_dir_only = body.endswith("/")
            exact: re.Pattern[str] | None = _exact_regex(regex.pattern)
        elif dir_only[index]:
            # Transformed .gitignore "foo/" arrives as "foo/**": git matches
            # the directory against "foo" itself.
            if not body.endswith(_DIR_CONTENTS_SUFFIX):
                raise _PathspecContractError(f"dir-only line {source!r} lacks /**")
            base = GitWildMatchPattern(body[: -len(_DIR_CONTENTS_SUFFIX)])
            is_dir_only = True
            exact = None if base.regex is None else _exact_regex(base.regex.pattern)
        else:
            is_dir_only = False
            exact = _exact_regex(regex.pattern)
        rules.append(_Rule(regex, exact, bool(include), is_dir_only))
        if not include:
            negations.append(regex)
            scopes.append(_negation_scope(body))
    return _SpecRules(spec, tuple(rules), tuple(negations), tuple(scopes))


def _spec_rules(spec: PathSpec, dir_only: Sequence[bool] | None = None) -> _SpecRules:
    """Read a compiled spec's patterns for directory and re-include decisions.

    ``dir_only`` flags each compiled line that a .gitignore wrote as a
    directory pattern; without it, a trailing "/" marks one. If pathspec's
    pattern objects are not as expected, the result is degraded: files are
    decided by pathspec alone and no directory is pruned, and this is logged
    once.
    """
    global _rules_degraded_warned
    try:
        return _build_spec_rules(spec, dir_only)
    except Exception as error:
        if not _rules_degraded_warned:
            _rules_degraded_warned = True
            logger.warning(
                "Ignore rules could not be read from pathspec ({}); directories "
                "are not pruned and re-includes follow pathspec alone",
                error,
            )
        return _SpecRules(spec, (), (), (), degraded=True)


def _exact_decision(rules: _SpecRules, rel: str, is_dir: bool) -> bool | None:
    """Git's own decision for ``rel``: the last pattern matching it wins.

    None when no pattern matches the path itself.
    """
    decision: bool | None = None
    for rule in rules.rules:
        if rule.exact is None or (rule.dir_only and not is_dir):
            continue
        if rule.exact.match(rel):
            decision = rule.include
    return decision


def _git_dir_excluded(rules: _SpecRules, rel: str) -> bool:
    """Whether git excludes directory ``rel``: itself, or a directory above it.

    Git never re-includes anything beneath an excluded directory.
    """
    memo = rules.dir_decisions
    known = memo.get(rel)
    if known is not None:
        return known
    parent = rel.rpartition("/")[0]
    excluded = (bool(parent) and _git_dir_excluded(rules, parent)) or (
        _exact_decision(rules, rel, True) is True
    )
    if len(memo) >= _DIR_DECISIONS_MAX:
        memo.clear()
    memo[rel] = excluded
    return excluded


def _negation_may_match_under(rules: _SpecRules, rel: str) -> bool:
    under = rel + "/"
    for scope in rules.negation_scopes:
        if not scope or scope.startswith(under) or under.startswith(scope):
            return True
    return False


def _contents_covered(rules: _SpecRules, rel: str) -> bool:
    """Whether the last file pattern matching "rel/" excludes it.

    Such a pattern (``foo/**``, ``**/node_modules/**``) matches everything
    beneath ``rel``. Directory patterns do not count: they match directories,
    not the files under them.
    """
    under = rel + "/"
    decision: bool | None = None
    for rule in rules.rules:
        if not rule.dir_only and rule.regex.match(under):
            decision = rule.include
    return decision is True


def _spec_excludes_dir(rules: _SpecRules, rel: str) -> bool:
    """Whether this spec excludes directory ``rel`` and so its whole subtree.

    Callers prune the walk on True, so True must mean nothing below ``rel``
    can be included. Two ways to know that:

    - git excludes the directory, itself or through a directory above it
      (``foo/``, ``foo``, ``/*``, or a ``foo/**`` that covers a
      subdirectory), and git never re-includes a path beneath it;
    - a file pattern covers the directory's contents (``foo/**``, or
      ChunkHound's ``**/node_modules/**`` config excludes) and no negation in
      the spec can match below it. With such a negation (``foo/**`` then
      ``!foo/bar.py``) the directory stays walkable and each file is decided
      on its own.
    """
    if not rel or rel == "." or rules.degraded:
        return False
    if _git_dir_excluded(rules, rel):
        return True
    return _contents_covered(rules, rel) and not _negation_may_match_under(rules, rel)


def _spec_excludes_file(rules: _SpecRules, rel: str) -> bool:
    """Whether this spec excludes file ``rel``.

    Without a negation that matches the file, pathspec's answer is git's. With
    one, the file is decided as git decides it: by the last pattern matching
    the file itself, and excluded regardless when a directory above it is. A
    directory re-include (``foo/**`` then ``!foo/bar/``) re-includes the
    directory only; ``foo/**`` still matches the files in it.
    """
    if rules.degraded or not any(regex.match(rel) for regex in rules.negations):
        return bool(rules.spec.match_file(rel))
    if _exact_decision(rules, rel, False) is True:
        return True
    parent = rel.rpartition("/")[0]
    return bool(parent) and _git_dir_excluded(rules, parent)


class IgnoreEngine:
    def __init__(
        self,
        root: Path,
        compiled_specs: list[tuple[Path, PathSpec]],
        spec_rules: list[_SpecRules] | None = None,
    ):
        self.root = root.resolve()
        self._compiled_specs = compiled_specs
        if spec_rules is None:
            spec_rules = [_spec_rules(spec) for _src, spec in compiled_specs]
        self._spec_rules = [
            (src, rules)
            for (src, _spec), rules in zip(compiled_specs, spec_rules, strict=True)
        ]

    def matches(self, path: Path, is_dir: bool) -> MatchInfo | None:
        """Return a match when ``path`` is excluded.

        For a directory a match means its whole subtree is excluded, so walks
        may prune it (see ``_spec_excludes_dir``). Specs are consulted in
        precedence order and the first that excludes the path wins.
        """
        # Normalize to root-relative POSIX path
        located = _in_tree_path(path, self.root)
        try:
            rel = located.relative_to(self.root)
        except Exception:
            rel = located
        rel_str = rel.as_posix()

        decide = _spec_excludes_dir if is_dir else _spec_excludes_file
        for src, rules in self._spec_rules:
            if decide(rules, rel_str):
                return MatchInfo(matched=True, source=src, pattern=None)
        return None


def _compile_gitwildmatch(patterns: Iterable[str]) -> PathSpec:
    if PathSpec is None or GitWildMatchPattern is None:
        raise RuntimeError(
            "pathspec is required for IgnoreEngine; please add dependency 'pathspec'"
        )
    return PathSpec.from_lines(GitWildMatchPattern, patterns)


def _compile_gitignore_lines(
    lines: Sequence[_GitignoreLine],
) -> tuple[PathSpec, _SpecRules]:
    """Compile transformed .gitignore lines, keeping which were dir patterns."""
    spec = _compile_gitwildmatch([pattern for pattern, _dir_only in lines])
    return spec, _spec_rules(spec, [dir_only for _pattern, dir_only in lines])


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
    config_exclude: Iterable[str] | None = None,
) -> IgnoreEngine:
    """Build an IgnoreEngine for the given root and sources.

    Currently supports:
    - gitignore: collects transformed .gitignore rules across the tree
    - config: uses provided glob-like patterns (gitwildmatch semantics)

    Note: Global gitignore (core.excludesFile) is NOT loaded here because the
    IgnoreEngine cannot properly handle cross-file negations. Global patterns
    are applied via git pathspec pushdown in IndexingCoordinator instead.
    """
    compiled: list[tuple[Path, PathSpec]] = []
    rules: list[_SpecRules] = []
    root = root.resolve()

    # Always enforce config_exclude (default excludes) regardless of sources
    if config_exclude:
        spec = _compile_gitwildmatch(config_exclude)
        compiled.append((root, spec))
        rules.append(_spec_rules(spec))

    for src in sources:
        if src == "gitignore":
            # Collect and transform .gitignore rules across the tree to
            # root-relative patterns.
            pre_spec = None
            if config_exclude:
                pre_spec = _compile_gitwildmatch(config_exclude)
            lines = _collect_gitignore_patterns(root, pre_spec)
            if lines:
                spec, spec_rules = _compile_gitignore_lines(lines)
                compiled.append((root / ".gitignore", spec))
                rules.append(spec_rules)
        # "chignore" source is no longer supported.
        elif src == "config":
            pats = list(config_exclude or [])
            if pats:
                spec = _compile_gitwildmatch(pats)
                compiled.append((root, spec))
                rules.append(_spec_rules(spec))

    return IgnoreEngine(root, compiled, rules)


def _collect_gitignore_patterns(
    root: Path, pre_exclude_spec: PathSpec | None = None
) -> list[_GitignoreLine]:
    """Return root-relative gitwildmatch patterns transformed from .gitignore files.

    We walk the directory tree top-down so that root patterns appear before
    child directory patterns; last match still wins in PathSpec.
    """
    out: list[_GitignoreLine] = []
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
                if pre_exclude_spec.match_file(child) or pre_exclude_spec.match_file(
                    child + "/"
                ):
                    to_remove.append(dn)
            for dn in to_remove:
                dirnames.remove(dn)
        out.extend(_read_transformed_gitignore_patterns(root, dpath))
    return out


def _read_transformed_gitignore_patterns(
    root: Path, dir_path: Path
) -> list[_GitignoreLine]:
    """Load a directory's .gitignore and transform it to root-relative patterns."""
    gi = dir_path / ".gitignore"
    if not gi.exists():
        return []

    try:
        lines = gi.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []

    try:
        rel_from_root = dir_path.relative_to(root)
    except ValueError:
        return []
    dir_rel = "." if str(rel_from_root) == "." else rel_from_root.as_posix()

    out: list[_GitignoreLine] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.extend(_transform_gitignore_line(dir_rel, line))
    return out


def _extend_gitignore_prune_state(
    root: Path,
    dir_path: Path,
    parent_state: _GitignorePruneState | None,
) -> _GitignorePruneState | None:
    """Build prune state for a directory from inherited + local .gitignore rules."""
    base_state = parent_state or _empty_gitignore_prune_state()
    inherited = base_state.patterns

    local_patterns = _read_transformed_gitignore_patterns(root, dir_path)
    if not local_patterns:
        return base_state

    combined = inherited + tuple(local_patterns)
    spec, rules = _compile_gitignore_lines(combined)
    return _GitignorePruneState(patterns=combined, spec=spec, rules=rules)


def _detect_repo_roots(
    root: Path,
    pre_exclude_spec: PathSpec | None = None,
    *,
    prune_ignored_gitfile_roots: bool = False,
    workspace_root_only_gitignore: bool = False,
) -> list[Path]:
    """Detect Git repository roots under root by looking for .git dir or file.

    Prunes excluded subtrees using pre_exclude_spec (e.g., node_modules) and,
    when enabled, prunes directories ignored by the active repo's .gitignore
    rules or the workspace-root non-repo overlay while preserving direct child
    repo boundaries.
    """
    roots: list[Path] = []
    root = root.resolve()
    ignored_cache: dict[tuple[str, str], bool] = {}
    gitignore_state_by_dir: dict[Path, _GitignorePruneState | None] = {}

    git_available = False
    git_check_ignored_fn = None
    if prune_ignored_gitfile_roots:
        try:
            git_available = shutil.which("git") is not None
        except Exception as e:
            git_available = False
            logger.debug(
                "Worktree ignore pruning: failed to check git availability: {}",
                e,
            )
        if git_available:
            try:
                from chunkhound.utils.git_safe import (
                    git_check_ignored as _git_check_ignored,
                )

                git_check_ignored_fn = _git_check_ignored
            except Exception as e:
                git_check_ignored_fn = None
                logger.debug(
                    "Worktree ignore pruning: failed to import git_check_ignored: {}",
                    e,
                )
        if (
            (root / ".git").is_dir()
            or (root / ".git").is_file()
            or workspace_root_only_gitignore
        ):
            gitignore_state_by_dir[root] = _extend_gitignore_prune_state(
                root,
                root,
                None,
            )
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dpath = Path(dirpath)
        dpath_resolved = dpath.resolve()
        current_gitignore_state = gitignore_state_by_dir.get(dpath_resolved)

        # Prune excluded dirs
        if pre_exclude_spec is not None:
            rel_base = "." if dpath == root else dpath.relative_to(root).as_posix()
            to_remove = []
            for dn in dirnames:
                child = dn if rel_base == "." else f"{rel_base}/{dn}"
                if pre_exclude_spec.match_file(child) or pre_exclude_spec.match_file(
                    child + "/"
                ):
                    to_remove.append(dn)
            for dn in to_remove:
                dirnames.remove(dn)

        # When gitignore is an active ignore source, avoid descending into
        # gitignored directories during repo-root detection. This closes the
        # gap where backend selection scans large ignored trees before actual
        # file discovery starts.
        if (
            current_gitignore_state is not None
            and current_gitignore_state.spec is not None
        ):
            rel_base = "." if dpath == root else dpath.relative_to(root).as_posix()
            to_remove = []
            for dn in dirnames:
                child_path = dpath / dn
                child_resolved = child_path.resolve()
                # Preserve direct child repo / worktree boundaries even when
                # the parent repo ignores the directory name.
                if (child_resolved / ".git").is_dir() or (
                    child_resolved / ".git"
                ).is_file():
                    continue
                child = dn if rel_base == "." else f"{rel_base}/{dn}"
                if current_gitignore_state.rules is not None and _spec_excludes_dir(
                    current_gitignore_state.rules, child
                ):
                    to_remove.append(dn)
            for dn in to_remove:
                dirnames.remove(dn)

        # Propagate gitignore prune state to directories that remain after
        # pruning. Repo boundaries start a fresh state from that repository's
        # own .gitignore lineage so parent rules do not leak into nested repos.
        if prune_ignored_gitfile_roots:
            for dn in dirnames:
                child_logical_path = dpath / dn
                if child_logical_path.is_symlink():
                    continue
                child_resolved = child_logical_path.resolve()
                child_is_repo = (child_resolved / ".git").is_dir() or (
                    child_resolved / ".git"
                ).is_file()
                if child_is_repo:
                    gitignore_state_by_dir[
                        child_resolved
                    ] = _extend_gitignore_prune_state(
                        root,
                        child_logical_path,
                        None,
                    )
                elif current_gitignore_state is not None:
                    gitignore_state_by_dir[
                        child_resolved
                    ] = _extend_gitignore_prune_state(
                        root,
                        child_logical_path,
                        current_gitignore_state,
                    )

        # Repo root if .git dir exists or .git file exists (worktree/submodule)
        git_dir = (dpath / ".git").is_dir()
        git_file = (dpath / ".git").is_file()
        if git_dir or git_file:
            if git_dir:
                # Full nested repos remain strict boundaries: do not consult parent
                # ignore rules here (by design; see repo boundary tests).
                roots.append(dpath)
                continue

            # `.git` file (linked worktree/submodule): optional guard. When enabled,
            # if this repo root directory is ignored by an ancestor repo's gitignore
            # rules, do NOT treat it as a boundary.
            if (
                not prune_ignored_gitfile_roots
                or (not git_available)
                or git_check_ignored_fn is None
            ):
                roots.append(dpath)
                continue

            # This avoids surprising discovery behavior when a parent repo ignores
            # a subtree (e.g. `.gitignored/`) that contains linked worktrees.
            parent: Path | None = None
            for rr in sorted(roots, key=lambda p: len(p.as_posix()), reverse=True):
                try:
                    dpath.resolve().relative_to(rr.resolve())
                    parent = rr
                    break
                except (ValueError, OSError):
                    continue

            if parent is not None:
                try:
                    try:
                        rel = dpath.resolve().relative_to(parent.resolve()).as_posix()
                    except (ValueError, OSError):
                        rel = ""
                    if rel:
                        try:
                            parent_key = str(parent.resolve())
                        except OSError:
                            parent_key = str(parent)
                        key = (parent_key, rel)
                        ign = ignored_cache.get(key)
                        if ign is None:
                            ign = git_check_ignored_fn(
                                repo_root=parent,
                                rel_path=rel,
                                timeout_s=5.0,
                                on_error=lambda e,
                                _dpath=dpath,
                                _parent=parent: logger.debug(
                                    "Worktree ignore pruning: git check failed "
                                    "for {} (parent={}): {}",
                                    _dpath,
                                    _parent,
                                    e,
                                ),
                            )
                            ignored_cache[key] = ign
                        if ign:
                            # Prune traversal below this directory as well.
                            dirnames[:] = []
                            continue
                except Exception as e:
                    logger.debug(
                        "Worktree ignore pruning: unexpected failure for {} "
                        "(parent={}): {}",
                        dpath,
                        parent,
                        e,
                    )

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
        config_exclude: Iterable[str] | None = None,
        workspace_root_only_gitignore: bool = False,
    ) -> None:
        self.root = workspace_root.resolve()
        self.repo_roots = sorted(
            [p.resolve() for p in repo_roots],
            key=lambda p: len(p.as_posix()),
            reverse=True,
        )
        self.sources = sources
        self.chignore_file = chignore_file
        self.config_exclude = list(config_exclude or [])

        # Build per-repo engines
        self._per_repo: dict[Path, IgnoreEngine] = {}
        for rr in self.repo_roots:
            self._per_repo[rr] = build_ignore_engine(
                rr, sources, chignore_file, self.config_exclude
            )
        # Workspace engine for non-repo areas
        if workspace_root_only_gitignore:
            # Workspace (non-repo) overlay: honor .gitignore files under the root
            # using the same collection logic as repo mode (root + nested files).
            # This preserves Git anchoring semantics and nested inheritance even
            # when the workspace itself is not a Git repository.
            compiled: list[tuple[Path, PathSpec]] = []
            rules: list[_SpecRules] = []
            if self.config_exclude:
                cfg_spec = _compile_gitwildmatch(self.config_exclude)
                compiled.append((self.root, cfg_spec))
                rules.append(_spec_rules(cfg_spec))
            try:
                pre = (
                    _compile_gitwildmatch(self.config_exclude)
                    if self.config_exclude
                    else None
                )
            except Exception:
                pre = None
            try:
                pats = _collect_gitignore_patterns(self.root, pre)
            except Exception:
                pats = []
            if pats:
                gi_spec, gi_rules = _compile_gitignore_lines(pats)
                compiled.append((self.root / ".gitignore", gi_spec))
                rules.append(gi_rules)
            self._workspace_engine = IgnoreEngine(self.root, compiled, rules)
        else:
            # Default: do NOT apply workspace .gitignore at all; only config_exclude
            compiled: list[tuple[Path, PathSpec]] = []
            if self.config_exclude:
                compiled.append((self.root, _compile_gitwildmatch(self.config_exclude)))
            self._workspace_engine = IgnoreEngine(self.root, compiled)

    def _nearest_repo(self, path: Path) -> Path | None:
        p = _in_tree_path(path, self.root)
        for rr in self.repo_roots:
            try:
                p.relative_to(rr)
                return rr
            except Exception:
                continue
        return None

    def matches(self, path: Path, is_dir: bool) -> MatchInfo | None:
        rr = self._nearest_repo(path)
        if rr is not None:
            return self._per_repo[rr].matches(path, is_dir)
        return self._workspace_engine.matches(path, is_dir)


def build_repo_aware_ignore_engine(
    root: Path,
    sources: list[str],
    chignore_file: str = ".chignore",
    config_exclude: Iterable[str] | None = None,
    backend: str = "python",
    workspace_root_only_gitignore: bool | None = None,
) -> RepoAwareIgnoreEvaluator:
    pre_spec = _compile_gitwildmatch(config_exclude or []) if (config_exclude) else None

    if workspace_root_only_gitignore is not None:
        prune_workspace_root_only = bool(workspace_root_only_gitignore)
    else:
        try:
            prune_workspace_root_only = os.environ.get(
                "CHUNKHOUND_INDEXING__WORKSPACE_GITIGNORE_NONREPO", ""
            ).strip() not in ("", "0", "false", "no")
        except Exception:
            prune_workspace_root_only = False

    repo_roots = _detect_repo_roots(
        root,
        pre_spec,
        prune_ignored_gitfile_roots=("gitignore" in (sources or [])),
        workspace_root_only_gitignore=prune_workspace_root_only,
    )
    if backend == "libgit2":
        eng = _try_build_libgit2_repo_aware(
            root, repo_roots, sources, chignore_file, config_exclude
        )
        if eng is not None:
            return eng
    # Determine workspace-root-only behavior.
    # Priority:
    # 1) Explicit parameter from config
    # 2) If sources include gitignore and there are NO repos in the workspace,
    #    default to True to honor a root .gitignore for non‑repo trees
    # 3) Legacy ENV override (kept for backward compatibility)
    if workspace_root_only_gitignore is not None:
        wr_only = prune_workspace_root_only
    else:
        if ("gitignore" in (sources or [])) and (not repo_roots):
            wr_only = True
        else:
            wr_only = prune_workspace_root_only
    return RepoAwareIgnoreEvaluator(
        root,
        repo_roots,
        sources,
        chignore_file,
        config_exclude,
        workspace_root_only_gitignore=wr_only,
    )


def _transform_gitignore_line(dir_rel: str, line: str) -> list[_GitignoreLine]:
    """Transform a .gitignore pattern from a directory into root-relative patterns.

    Handles negation (!), anchored (/), and directory-only (trailing /) forms by
    emitting patterns that constrain the match to the originating subtree.
    Each pattern comes with whether it was a directory pattern: "foo/" is
    emitted as "foo/**", which matches the same files, and the flag lets
    directory decisions tell it from a "foo/**" the .gitignore wrote.
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
    parts: list[_GitignoreLine] = []

    def add(p: str) -> None:
        if is_dir_pat:
            p = f"{p}/**"
        if neg:
            p = "!" + p
        parts.append((p, is_dir_pat))

    # Resolve directory prefix with Git semantics
    # Rules (simplified from gitignore docs):
    # - Leading '/' anchors to the directory containing the .gitignore.
    # - A pattern that contains a '/' (after trimming trailing '/') is
    #   anchored to that directory.
    # - A pattern without any '/' matches in any directory under the
    #   .gitignore directory.
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


def detect_repo_roots(
    root: Path,
    config_exclude: Iterable[str] | None = None,
    *,
    prune_ignored_gitfile_roots: bool = False,
    workspace_root_only_gitignore: bool = False,
) -> list[Path]:
    """Public helper to detect repo roots under a workspace root.

    Applies pruning using config_exclude (gitwildmatch semantics) to avoid
    descending into heavy trees (e.g., node_modules) while scanning. When
    ``prune_ignored_gitfile_roots`` is enabled, directories ignored by the
    active repo's .gitignore rules or the workspace-root non-repo overlay are
    also pruned during the scan.
    """
    pre_spec = _compile_gitwildmatch(config_exclude or []) if config_exclude else None
    return _detect_repo_roots(
        root,
        pre_spec,
        prune_ignored_gitfile_roots=prune_ignored_gitfile_roots,
        workspace_root_only_gitignore=workspace_root_only_gitignore,
    )


def build_repo_aware_ignore_engine_from_roots(
    root: Path,
    repo_roots: list[Path],
    sources: list[str],
    chignore_file: str = ".chignore",
    config_exclude: Iterable[str] | None = None,
    backend: str = "python",
    workspace_root_only_gitignore: bool | None = None,
) -> RepoAwareIgnoreEvaluator:
    """Build a repo-aware evaluator from a precomputed list of repo roots.

    Avoids re-scanning the entire workspace per worker when running in parallel.
    """
    if backend == "libgit2":
        eng = _try_build_libgit2_repo_aware(
            root, repo_roots, sources, chignore_file, config_exclude
        )
        if eng is not None:
            return eng
    if workspace_root_only_gitignore is None:
        try:
            wr_only = os.environ.get(
                "CHUNKHOUND_INDEXING__WORKSPACE_GITIGNORE_NONREPO", ""
            ).strip() not in ("", "0", "false", "no")
        except Exception:
            wr_only = False
    else:
        wr_only = bool(workspace_root_only_gitignore)
    return RepoAwareIgnoreEvaluator(
        root,
        repo_roots,
        sources,
        chignore_file,
        config_exclude,
        workspace_root_only_gitignore=wr_only,
    )


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
        config_exclude: Iterable[str] | None = None,
    ) -> None:
        self.root = workspace_root.resolve()
        self.repo_roots = sorted(
            [p.resolve() for p in repo_roots],
            key=lambda p: len(p.as_posix()),
            reverse=True,
        )
        self.sources = sources
        self.chignore_file = chignore_file
        self.config_exclude = list(config_exclude or [])

        # Precompile config_exclude with pathspec for fast hard excludes
        self._cfg_spec = (
            _compile_gitwildmatch(self.config_exclude) if self.config_exclude else None
        )
        self._cfg_rules = (
            _spec_rules(self._cfg_spec) if self._cfg_spec is not None else None
        )

        # Open libgit2 repos
        self._repos: dict[Path, object] = {}
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

    def _nearest_repo(self, path: Path) -> Path | None:
        p = _in_tree_path(path, self.root)
        for rr in self.repo_roots:
            try:
                p.relative_to(rr)
                return rr
            except Exception:
                continue
        return None

    def _cfg_excluded(self, rel: str, is_dir: bool) -> bool:
        if self._cfg_rules is None:
            return False
        # Same rule as IgnoreEngine.matches: a directory match means its whole
        # subtree is excluded. libgit2 decides .gitignore with git's semantics.
        if is_dir:
            return _spec_excludes_dir(self._cfg_rules, rel)
        return _spec_excludes_file(self._cfg_rules, rel)

    def matches(self, path: Path, is_dir: bool) -> MatchInfo | None:
        # Hard exclude via config_exclude first
        try:
            rel_cfg = _in_tree_path(path, self.root).relative_to(self.root).as_posix()
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
            rel = _in_tree_path(path, rr).relative_to(rr).as_posix()
        except Exception:
            rel = path.name

        # Try common pygit2 ignore API methods (varies by version)
        try:
            fn = getattr(repo, "is_path_ignored", None) or getattr(
                repo, "path_is_ignored", None
            )
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
    config_exclude: Iterable[str] | None = None,
) -> RepoAwareLibgit2Evaluator | None:
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
                "gitignore_backend=libgit2 requested but pygit2 is not "
                "available; falling back to python backend"
            )
            _LIBGIT2_WARNED = True  # type: ignore[assignment]
        return None
    try:
        return RepoAwareLibgit2Evaluator(
            root, repo_roots, sources, chignore_file, config_exclude
        )
    except Exception:
        if not _LIBGIT2_WARNED and not os.environ.get("CHUNKHOUND_MCP_MODE"):
            logger.warning(
                "gitignore_backend=libgit2 requested but initialization "
                "failed; falling back to python backend"
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
