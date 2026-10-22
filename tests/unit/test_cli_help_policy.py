"""Static guard: help and version output is tested in-process, not via a child.

Subprocesses are reserved for end-to-end workflows. One real process start is
allowed to prove the entry point still launches.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parents[1]
_CLI_HELPERS = frozenset({"run_cli", "run_command", "cli_command"})
_HELP_FLAGS = frozenset({"--help", "--version"})
# The one real child process: proves interpreter, imports, and native load.
_REAL_START_ALLOWLIST = frozenset({("test_smoke.py", "test_cli_entry_point_starts")})


def _has_flag(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Constant) and child.value in _HELP_FLAGS
        for child in ast.walk(node)
    )


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    """Map local names of the CLI subprocess helpers to their real names."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for imported in node.names:
                if imported.name in _CLI_HELPERS:
                    aliases[imported.asname or imported.name] = imported.name
    return aliases


def _helper_name(node: ast.expr, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        name = aliases.get(node.id, node.id)
    elif isinstance(node, ast.Attribute):
        name = node.attr
    else:
        return None
    return name if name in _CLI_HELPERS else None


class _HelpCallVisitor(ast.NodeVisitor):
    """Flag CLI subprocess calls that exercise help or version output."""

    def __init__(self, aliases: dict[str, str]) -> None:
        self._aliases = aliases
        self._functions: list[tuple[str, bool]] = []
        self.offenders: list[tuple[int, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        decorator_mentions_help = any(_has_flag(d) for d in node.decorator_list)
        self._functions.append((node.name, decorator_mentions_help))
        self.generic_visit(node)
        self._functions.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if _helper_name(node.func, self._aliases) and self._is_offender(node):
            function = self._functions[-1][0] if self._functions else ""
            self.offenders.append((node.lineno, function))
        self.generic_visit(node)

    def _is_offender(self, node: ast.Call) -> bool:
        # A literal flag anywhere in the call catches variable argv, tuples,
        # nested cli_command() calls, and keyword arguments.
        if _has_flag(node):
            return True
        if not self._functions:
            return False
        name, decorator_mentions_help = self._functions[-1]
        if decorator_mentions_help:
            return True
        lowered = name.lower()
        tokens = set(re.split(r"[^a-z0-9]+", lowered))
        return "help" in tokens or "version" in tokens


def _offenders(root: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        visitor = _HelpCallVisitor(_import_aliases(tree))
        visitor.visit(tree)
        found.extend(
            f"{path.relative_to(root)}:{line}"
            for line, function in visitor.offenders
            if (path.name, function) not in _REAL_START_ALLOWLIST
        )
    return found


def test_help_and_version_run_in_process() -> None:
    offenders = _offenders(_TESTS_ROOT)
    assert not offenders, (
        "Help/version is parser work: use run_cli_in_process() instead of a "
        "cold subprocess: " + ", ".join(offenders)
    )


def test_guard_detects_indirect_help(tmp_path: Path) -> None:
    source = tmp_path / "test_indirect.py"
    source.write_text(
        "def test_search_help():\n    args = ['search', '--help']\n    run_cli(args)\n",
        encoding="utf-8",
    )

    assert _offenders(tmp_path) == ["test_indirect.py:3"]


def test_guard_allows_in_process_help(tmp_path: Path) -> None:
    source = tmp_path / "test_in_process.py"
    source.write_text(
        "def test_search_help():\n    run_cli_in_process(['search', '--help'])\n",
        encoding="utf-8",
    )

    assert _offenders(tmp_path) == []
