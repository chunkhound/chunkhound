"""Git-exact ignore decisions read from pathspec's compiled patterns.

pathspec answers whether a path matches a spec. Git also needs to know when a
directory is excluded as a whole, so a walk can prune it, and how a negation
re-includes a file. This module derives those answers from the compiled
patterns. ``ignore_engine`` builds the specs and calls these functions.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from loguru import logger

try:
    from pathspec import PathSpec
    from pathspec.patterns.gitwildmatch import GitWildMatchPattern
except Exception:  # pragma: no cover - import error surfaced at runtime
    PathSpec = None  # type: ignore
    GitWildMatchPattern = None  # type: ignore

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


# pathspec ends a pattern's regex with a tail, the group named ps_d, that
# makes a match on a directory also match everything beneath it. Git matches
# a pattern against the path itself, so the exact form ends at the path
# instead. The tail is spelled per pathspec version:
#   0.12: "(?:(?P<ps_d>/).*)?$" (file or directory), "(?P<ps_d>/).*$" (directory)
#   1.x:  "(?:(?P<ps_d>/)|$)" and "(?P<ps_d>/)"
# pathspec 1.x applies its regexes with re.search, and some are unanchored
# ("*" compiles to "."). 0.12 anchors every regex, so re.search is right for
# both; every regex below is applied with it.
_PS_D_TAILS = (
    "(?:(?P<ps_d>/).*)?$",
    "(?P<ps_d>/).*$",
    "(?:(?P<ps_d>/)|$)",
    "(?P<ps_d>/)",
)
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


def _exact_regex(pathspec_regex: str, source: str) -> re.Pattern[str]:
    exact = pathspec_regex
    for tail in _PS_D_TAILS:
        if exact.endswith(tail):
            exact = exact[: -len(tail)] + "$"
            break
    if "ps_d" in exact:
        import pathspec as installed_pathspec

        version = getattr(installed_pathspec, "__version__", "?")
        raise _PathspecContractError(
            f"pathspec {version} compiled {source!r} to an unexpected regex "
            f"{pathspec_regex!r}"
        )
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
            exact: re.Pattern[str] | None = _exact_regex(regex.pattern, source)
        elif dir_only[index]:
            # Transformed .gitignore "foo/" arrives as "foo/**": git matches
            # the directory against "foo" itself.
            if not body.endswith(_DIR_CONTENTS_SUFFIX):
                raise _PathspecContractError(f"dir-only line {source!r} lacks /**")
            base = GitWildMatchPattern(body[: -len(_DIR_CONTENTS_SUFFIX)])
            is_dir_only = True
            exact = (
                None if base.regex is None else _exact_regex(base.regex.pattern, source)
            )
        else:
            is_dir_only = False
            exact = _exact_regex(regex.pattern, source)
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
        if rule.exact.search(rel):
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
        if not rule.dir_only and rule.regex.search(under):
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
    if rules.degraded or not any(regex.search(rel) for regex in rules.negations):
        return bool(rules.spec.match_file(rel))
    if _exact_decision(rules, rel, False) is True:
        return True
    parent = rel.rpartition("/")[0]
    return bool(parent) and _git_dir_excluded(rules, parent)
