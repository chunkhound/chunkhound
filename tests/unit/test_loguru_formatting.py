"""Prevent printf-style placeholders from reaching Loguru unchanged.

Loguru uses str.format placeholders, unlike the stdlib logging module.
This checks only calls whose logger name is imported from Loguru, so existing
logging.getLogger calls can continue to use percent-style formatting.
"""

import ast
import re
from pathlib import Path


_PRINTF_PLACEHOLDER = re.compile(r"%(?:\d+(?:\.\d+)?)?[sdrf]\b")
_LOG_METHODS = {"trace", "debug", "info", "success", "warning", "error", "critical", "exception", "log"}


def test_loguru_messages_do_not_use_printf_placeholders() -> None:
    root = Path(__file__).resolve().parents[2] / "chunkhound"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "loguru" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        logger_names = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "loguru"
            for alias in node.names
            if alias.name == "logger"
        }
        if not logger_names:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            method = node.func
            if not isinstance(method, ast.Attribute) or method.attr not in _LOG_METHODS:
                continue
            if not isinstance(method.value, ast.Name) or method.value.id not in logger_names:
                continue
            template = node.args[0]
            if not isinstance(template, ast.Constant) or not isinstance(template.value, str):
                continue
            if _PRINTF_PLACEHOLDER.search(template.value):
                offenders.append(f"{path.relative_to(root.parent)}:{node.lineno}")

    assert not offenders, "Loguru calls use printf-style placeholders: " + ", ".join(offenders)
