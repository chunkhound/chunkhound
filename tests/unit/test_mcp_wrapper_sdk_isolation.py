"""Enforces the single-dispatch-point design for the MCP SDK.

Every module directly under chunkhound/mcp_server/ and chunkhound/daemon/
must not import the MCP SDK (``mcp``/``mcp.*`` or ``mcp_types``/``mcp_types.*``),
with two exemptions:

- chunkhound/mcp_server/base.py is the single dispatch point and owns all SDK
  usage (it must import the SDK, and only public SDK modules).
- chunkhound/mcp_server/common.py imports ``mcp.types`` for type helpers only.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DISPATCH_POINT_FILE = "chunkhound/mcp_server/base.py"
EXEMPT_FILES = {DISPATCH_POINT_FILE, "chunkhound/mcp_server/common.py"}
SDK_PACKAGES = ("mcp", "mcp_types")


def _discover_wrapper_files() -> list[str]:
    paths = [
        *REPO_ROOT.glob("chunkhound/mcp_server/*.py"),
        *REPO_ROOT.glob("chunkhound/daemon/*.py"),
    ]
    rel = (p.relative_to(REPO_ROOT).as_posix() for p in paths)
    return sorted(r for r in rel if r not in EXEMPT_FILES)


WRAPPER_FILES = _discover_wrapper_files()


def _is_sdk_module(name: str) -> bool:
    return any(name == pkg or name.startswith(f"{pkg}.") for pkg in SDK_PACKAGES)


def _mcp_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names if _is_sdk_module(a.name)]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if _is_sdk_module(node.module):
                hits.append(node.module)
    return hits


def test_wrapper_discovery_is_not_empty() -> None:
    assert WRAPPER_FILES, "no wrapper files discovered — glob paths are stale"


@pytest.mark.parametrize("rel_path", WRAPPER_FILES)
def test_wrapper_does_not_import_mcp_sdk(rel_path: str) -> None:
    hits = _mcp_imports(REPO_ROOT / rel_path)
    assert not hits, (
        f"{rel_path} imports the mcp SDK directly: {hits} — route through "
        "MCPServerBase.serve_connection()/build_streamable_http_app() instead"
    )


def test_dispatch_point_file_does_import_mcp_sdk() -> None:
    """The flip side of the invariant: exactly one true owner, not just
    non-owners — guards against this test passing vacuously if base.py ever
    stopped importing mcp too."""
    assert _mcp_imports(REPO_ROOT / DISPATCH_POINT_FILE), (
        f"{DISPATCH_POINT_FILE} no longer imports mcp — the single dispatch "
        "point has disappeared; something is structurally wrong"
    )


def test_dispatch_point_uses_only_public_mcp_modules() -> None:
    """Private SDK modules (``mcp.*._x`` / ``mcp_types.*._x``) may change in
    any minor release of the range we accept; base.py must not depend on
    them."""
    private = [
        m
        for m in _mcp_imports(REPO_ROOT / DISPATCH_POINT_FILE)
        if any(part.startswith("_") for part in m.split("."))
    ]
    assert not private, f"{DISPATCH_POINT_FILE} imports private mcp modules: {private}"
