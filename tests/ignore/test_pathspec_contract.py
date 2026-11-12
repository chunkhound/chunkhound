"""ChunkHound reads pathspec's compiled patterns to decide directories as git does.

pathspec has no upper version bound. If its pattern objects change shape, the
rules degrade to pathspec's own matching (no pruning, no git-exact
re-includes) and log a warning; this test fails first so the change is seen.
"""

from __future__ import annotations

from chunkhound.utils.ignore_engine import (
    _compile_gitignore_lines,
    _compile_gitwildmatch,
    _spec_rules,
    _transform_gitignore_line,
)


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
