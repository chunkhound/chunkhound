"""Guard against printf-style args in loguru calls (chunkhound#458).

Loguru interpolates only ``{}`` placeholders (or lazy callables). A call
such as ``logger.warning("missing for %s", path)`` therefore logs the
literal text ``%s`` instead of the value. This test fails on any such
call in the ``chunkhound`` package so the style cannot drift back.

Files that use the standard library ``logging`` module (e.g.
``chunkhound/parsers/rapid_yaml_parser.py``) are intentionally excluded:
there ``%s`` formatting is correct and is handled by ``logging`` itself.
Only files that import loguru are scanned, and only real call nodes are
considered, so comments and docstrings never trip the guard.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# A printf conversion spec, e.g. %s, %d, %i, %f, %.2f, %05d -- but not
# an escaped literal "%%".
_PRINTF_SPEC = re.compile(r"%(?!%)[#0\- +]*\d*(?:\.\d+)?[sdifgeEra]")

_LOGURU_LEVELS = {
    "trace",
    "debug",
    "info",
    "success",
    "warning",
    "error",
    "critical",
}


def _is_loguru_call(node: ast.Call) -> bool:
    """True for ``logger.<level>(...)`` and ``logger.opt(...).<level>(...)``."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in _LOGURU_LEVELS:
        return False
    value = func.value
    if isinstance(value, ast.Name):
        return value.id == "logger"
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
        inner = value.func.value
        return isinstance(inner, ast.Name) and inner.id == "logger"
    return False


def _is_loguru_log_call(node: ast.Call) -> bool:
    """True for ``logger.opt(...).log(level, message, ...)``."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "log":
        return False
    value = func.value
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
        inner = value.func.value
        return isinstance(inner, ast.Name) and inner.id == "logger"
    return False


def find_printf_loguru_calls(source: str) -> list[str]:
    """Return ``line: format`` entries for printf-style loguru calls."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fmt_arg: ast.expr | None = None
        if _is_loguru_call(node) and node.args:
            fmt_arg = node.args[0]
        elif _is_loguru_log_call(node) and len(node.args) >= 2:
            fmt_arg = node.args[1]
        if fmt_arg is None:
            continue
        if isinstance(fmt_arg, ast.Constant) and isinstance(fmt_arg.value, str):
            if _PRINTF_SPEC.search(fmt_arg.value):
                violations.append(f"{node.lineno}: {fmt_arg.value!r}")
        elif isinstance(fmt_arg, ast.BinOp) and isinstance(fmt_arg.op, ast.Mod):
            # logger.debug("... %s ..." % value) -- also never interpolated
            # by loguru; flag it by its line regardless of the format text.
            violations.append(f"{node.lineno}: printf-style % formatting")
    return violations


def test_scanner_catches_printf_style_and_ignores_braces() -> None:
    assert find_printf_loguru_calls('logger.warning("missing for %s", path)\n') == [
        "1: 'missing for %s'"
    ]
    assert find_printf_loguru_calls('logger.warning("missing for {}", path)\n') == []
    assert find_printf_loguru_calls('logger.debug("after %ds", timeout)\n') != []
    # %r (repr) is also a printf-style specifier the guard must catch.
    assert find_printf_loguru_calls('logger.info("value=%r", x)\n') == [
        "1: 'value=%r'"
    ]
    # A literal "%%" is an escaped percent, not a placeholder.
    assert find_printf_loguru_calls('logger.info("100%% complete")\n') == []


def test_no_printf_style_loguru_calls_in_source() -> None:
    package_root = Path(__file__).resolve().parents[1] / "chunkhound"
    assert package_root.is_dir(), f"chunkhound package not found at {package_root}"
    violations: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8", errors="ignore")
        if "loguru" not in source:
            # stdlib-logging modules (e.g. rapid_yaml_parser.py) are out
            # of scope: %s formatting is correct there.
            continue
        for entry in find_printf_loguru_calls(source):
            violations.append(f"{path.relative_to(package_root.parent)}:{entry}")
    assert not violations, (
        "printf-style loguru calls found (loguru only interpolates {}):\n"
        + "\n".join(violations)
    )
