"""Enforces the single-dispatch-point design: exactly
chunkhound/mcp_server/base.py imports the mcp SDK — no more, no less."""

import ast
from pathlib import Path

import pytest

WRAPPER_FILES = [
    "chunkhound/daemon/server.py",
    "chunkhound/daemon/client_proxy.py",
    "chunkhound/mcp_server/stdio.py",
    "chunkhound/mcp_server/http_server.py",
]
DISPATCH_POINT_FILE = "chunkhound/mcp_server/base.py"


def _mcp_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [
                a.name
                for a in node.names
                if a.name == "mcp" or a.name.startswith("mcp.")
            ]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "mcp" or node.module.startswith("mcp."):
                hits.append(node.module)
    return hits


@pytest.mark.parametrize("rel_path", WRAPPER_FILES)
def test_wrapper_does_not_import_mcp_sdk(rel_path: str) -> None:
    hits = _mcp_imports(Path(rel_path))
    assert not hits, (
        f"{rel_path} imports the mcp SDK directly: {hits} — route through "
        "MCPServerBase.serve_connection()/build_streamable_http_app() instead"
    )


def test_dispatch_point_file_does_import_mcp_sdk() -> None:
    """The flip side of the invariant: exactly one true owner, not just four
    non-owners — guards against this test passing vacuously if base.py ever
    stopped importing mcp too."""
    assert _mcp_imports(Path(DISPATCH_POINT_FILE)), (
        f"{DISPATCH_POINT_FILE} no longer imports mcp — the single dispatch "
        "point has disappeared; something is structurally wrong"
    )


def test_dispatch_point_uses_only_public_mcp_modules() -> None:
    """Private SDK modules (``mcp.*._x``) may change in any minor release of
    the ``mcp>=2.3,<3`` range we accept; base.py must not depend on them."""
    private = [
        m
        for m in _mcp_imports(Path(DISPATCH_POINT_FILE))
        if any(part.startswith("_") for part in m.split("."))
    ]
    assert not private, f"{DISPATCH_POINT_FILE} imports private mcp modules: {private}"
