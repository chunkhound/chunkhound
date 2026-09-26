"""Guard: site tests must decode child output as UTF-8.

WHY: ``subprocess.run(text=True)`` decodes pipes with the host locale code page
(cp1252 on Windows), which silently corrupts the UTF-8 bytes Node/npm emit and
makes the same test pass on Linux and fail on Windows. All site-test child
processes must go through ``tests/site/process_runner.run_text_process``, which
pins UTF-8. This guard is the single source of truth for that rule.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SITE_DIR = Path(__file__).resolve().parent
HELPER = SITE_DIR / "process_runner.py"

# subprocess entry points that read child text; direct use bypasses the helper.
_TEXT_ENTRY_POINTS = frozenset(
    {
        "run",
        "Popen",
        "call",
        "check_call",
        "check_output",
        "getoutput",
        "getstatusoutput",
    }
)


def _violations(path: Path, tree: ast.AST) -> list[int]:
    """Line numbers breaking the rule: direct subprocess text entry points, or
    import forms the call scanner below cannot see through (from-import, alias).
    Only plain ``import subprocess`` is allowed."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.ImportFrom) and node.module == "subprocess") or (
            isinstance(node, ast.Import)
            and any(a for a in node.names if a.name == "subprocess" and a.asname)
        ):
            lines.add(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr in _TEXT_ENTRY_POINTS
        ):
            lines.add(node.lineno)
    return sorted(lines)


def test_site_tests_run_children_only_through_utf8_helper() -> None:
    offenders = []
    for path in sorted(SITE_DIR.rglob("*.py")):
        if path == HELPER:
            continue
        for line in _violations(path, ast.parse(path.read_text(encoding="utf-8"))):
            offenders.append(f"{path.relative_to(SITE_DIR)}:{line}")
    assert not offenders, (
        "call tests/site/process_runner.run_text_process instead of subprocess "
        f"directly: {offenders}"
    )


def test_utf8_helper_round_trips_non_ascii_bytes() -> None:
    """A cp1252 decode would mangle these bytes; the helper must not."""
    from tests.site.process_runner import run_text_process

    payload = "\u2014\u00b7\u2192"  # em dash, middle dot, right arrow
    code = f"import sys; sys.stdout.buffer.write({payload!r}.encode('utf-8'))"
    result = run_text_process([sys.executable, "-c", code], check=True)
    assert result.stdout == payload
