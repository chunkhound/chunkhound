"""ChunkHound reads pathspec's compiled patterns to decide directories as git does.

pathspec has no upper version bound. If its pattern objects change shape, the
rules degrade to pathspec's own matching (no pruning, no git-exact
re-includes) and log a warning; these tests fail first so the change is seen.
They compile with the installed pathspec, so they guard whichever version the
lock resolves (0.12 spells a regex's tail "(?:(?P<ps_d>/).*)?$", 1.x
"(?:(?P<ps_d>/)|$)" and matches with re.search).
"""

from __future__ import annotations

from pathlib import Path

from chunkhound.utils.ignore_engine import (
    _compile_gitignore_lines,
    _compile_gitwildmatch,
    _spec_rules,
    _transform_gitignore_line,
    build_repo_aware_ignore_engine,
)

# Every form a .gitignore line takes: plain file, glob, anchored, directory,
# negation, and the whole-tree wildcards.
_GITIGNORE_FORMS = [
    "Thumbs.db",
    "*.log",
    "/build",
    "build/",
    "/build/",
    "node_modules/",
    "foo/**",
    "**/node_modules/**",
    "!keep.py",
    "!/src/",
    "src/**/x.py",
    "**/foo",
    "a/b/",
    "*",
    "/*",
    "foo/*",
    "**",
    "**/",
    "/**/",
    "a/**/b",
    "dir/**/",
    "[abc].py",
    "?x.py",
    ".env*",
    "!.impeccable/live/config.json",
]


def test_ignore_rules_are_read_from_compiled_pathspec_patterns() -> None:
    lines = [
        *_transform_gitignore_line(".", "foo/"),
        *_transform_gitignore_line(".", "!foo/bar.py"),
        *_transform_gitignore_line("src", "!/keep/"),
    ]
    _spec, rules = _compile_gitignore_lines(lines)

    assert not rules.degraded
    assert [rule.dir_only for rule in rules.rules] == [d for _p, d in lines]
    assert [rule.include for rule in rules.rules] == [
        not p.startswith("!") for p, _d in lines
    ]
    assert all(rule.exact is not None for rule in rules.rules)

    config_rules = _spec_rules(_compile_gitwildmatch(["**/node_modules/**", "dist/"]))
    assert not config_rules.degraded
    assert [rule.dir_only for rule in config_rules.rules] == [False, True]


def test_every_gitignore_pattern_form_is_read_from_the_installed_pathspec() -> None:
    lines = [
        line
        for form in _GITIGNORE_FORMS
        for line in _transform_gitignore_line(".", form)
    ]
    _spec, rules = _compile_gitignore_lines(lines)
    assert not rules.degraded

    excludes = [form for form in _GITIGNORE_FORMS if not form.startswith("!")]
    assert not _spec_rules(_compile_gitwildmatch(excludes)).degraded


def test_a_plain_file_pattern_keeps_ignored_directories_pruned(tmp_path: Path) -> None:
    """A .gitignore that names a file (Thumbs.db) beside node_modules/ (TFS 65863).

    With pathspec 1.1 such a line once stopped all directory pruning: the walks
    entered node_modules, while each file in it was still excluded.
    """
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / ".gitignore").write_text(
        "Thumbs.db\n*.log\nnode_modules/\n", encoding="utf-8"
    )
    engine = build_repo_aware_ignore_engine(
        root, ["gitignore"], ".chignore", ["**/dist/**"]
    )

    assert engine.matches(root / "node_modules", is_dir=True)
    assert engine.matches(root / "pkg" / "node_modules", is_dir=True)
    assert engine.matches(root / "web" / "dist", is_dir=True)
    assert not engine.matches(root / "src", is_dir=True)
    assert engine.matches(root / "src" / "Thumbs.db", is_dir=False)
    assert engine.matches(root / "src" / "debug.log", is_dir=False)
    assert not engine.matches(root / "src" / "app.py", is_dir=False)
