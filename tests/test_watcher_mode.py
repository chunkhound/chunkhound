import ast
from pathlib import Path

import pytest

from tests import conftest as tests_conftest
from tests.utils.windows_compat import (
    get_configured_watcher_mode,
    normalize_watcher_mode,
    should_use_polling,
)


class _FakeNode:
    def __init__(self, *, native: bool = False, polling: bool = False) -> None:
        self.nodeid = "tests/test_watcher_mode.py::fake"
        self._markers = {
            "native_watcher": [object()] if native else [],
            "polling_watcher": [object()] if polling else [],
        }

    def iter_markers(self, name: str):
        return iter(self._markers.get(name, ()))


class _FakeRequest:
    def __init__(self, *, native: bool = False, polling: bool = False) -> None:
        self.node = _FakeNode(native=native, polling=polling)


def _resolve_watcher_mode(request: _FakeRequest) -> str:
    return tests_conftest.watcher_mode.__wrapped__(request)


_WATCHER_MARKS = frozenset({"pytest.mark.native_watcher", "pytest.mark.polling_watcher"})


def _decorator_name(node: ast.expr) -> str | None:
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Attribute):
        return ast.unparse(target)
    return None


def _has_watcher_marker(node: ast.ClassDef | ast.AsyncFunctionDef | ast.FunctionDef) -> bool:
    return any(_decorator_name(d) in _WATCHER_MARKS for d in node.decorator_list)


def _module_has_watcher_pytestmark(module: ast.Module) -> bool:
    """Check module-level pytestmark = [...] for a watcher marker."""
    for stmt in module.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets):
            marks = stmt.value.elts if isinstance(stmt.value, ast.List) else [stmt.value]
            if any(_decorator_name(m) in _WATCHER_MARKS for m in marks):
                return True
    return False


def _contains_watcher_backend_ops(node: ast.AsyncFunctionDef | ast.FunctionDef) -> bool:
    # "start" is intentionally excluded — too generic (MCP clients, threads also call .start()).
    # The wait_for_* methods are distinctive enough to identify watcher backend usage.
    # Keep in sync with public wait_for_* methods on RealtimeIndexingService.
    backend_calls = {
        "wait_for_monitoring_ready",
        "wait_for_file_indexed",
        "wait_for_file_removed",
    }
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            if child.func.attr in backend_calls:
                return True
        if isinstance(child, ast.Assign):
            for target in child.targets:
                # Sentinel: prevent re-introducing the removed _force_polling=True hack
                # without an explicit watcher marker. Remove this check if the attribute
                # is renamed or deleted from the production code.
                if isinstance(target, ast.Attribute) and target.attr == "_force_polling":
                    return True
    return False


def test_normalize_watcher_mode_defaults_to_native() -> None:
    assert normalize_watcher_mode(None) == "native"
    assert normalize_watcher_mode("") == "native"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("native", False), ("polling", True), (" Native ", False), ("POLLING", True)],
)
def test_should_use_polling_depends_only_on_explicit_mode(
    mode: str, expected: bool
) -> None:
    assert should_use_polling(mode) is expected


def test_get_configured_watcher_mode_reads_optional_env(monkeypatch) -> None:
    monkeypatch.setenv("CHUNKHOUND_TEST_WATCHER_MODE", "polling")
    assert get_configured_watcher_mode() == "polling"


def test_get_configured_watcher_mode_defaults_to_native(monkeypatch) -> None:
    monkeypatch.delenv("CHUNKHOUND_TEST_WATCHER_MODE", raising=False)
    assert get_configured_watcher_mode() == "native"


def test_watcher_mode_fixture_prefers_explicit_native_marker(monkeypatch) -> None:
    monkeypatch.setenv("CHUNKHOUND_TEST_WATCHER_MODE", "polling")
    assert _resolve_watcher_mode(_FakeRequest(native=True)) == "native"


def test_watcher_mode_fixture_prefers_explicit_polling_marker(monkeypatch) -> None:
    monkeypatch.setenv("CHUNKHOUND_TEST_WATCHER_MODE", "native")
    assert _resolve_watcher_mode(_FakeRequest(polling=True)) == "polling"


def test_watcher_mode_fixture_uses_env_without_markers(monkeypatch) -> None:
    monkeypatch.setenv("CHUNKHOUND_TEST_WATCHER_MODE", "polling")
    assert _resolve_watcher_mode(_FakeRequest()) == "polling"


def test_watcher_mode_fixture_rejects_conflicting_markers() -> None:
    with pytest.raises(pytest.UsageError):
        _resolve_watcher_mode(_FakeRequest(native=True, polling=True))


def test_generic_watcher_markers_are_registered(pytestconfig) -> None:
    marker_lines = pytestconfig.getini("markers")
    assert any(line.startswith("native_watcher:") for line in marker_lines)
    assert any(line.startswith("polling_watcher:") for line in marker_lines)
    assert not any(line.startswith("windows_native_watcher:") for line in marker_lines)
    assert not any(line.startswith("windows_polling:") for line in marker_lines)


def test_watcher_backend_operations_require_explicit_marker() -> None:
    tests_dir = Path(__file__).resolve().parent
    # Scan all test files — ones without watcher ops will produce 0 offenders naturally.
    target_files = sorted(
        p for p in tests_dir.rglob("test_*.py")
        if p.name != "test_watcher_mode.py"  # exclude self (no watcher ops in here)
    )
    offenders: list[str] = []

    for path in target_files:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_marked = _module_has_watcher_pytestmark(module)

        for top in module.body:
            if isinstance(top, ast.ClassDef):
                class_marked = module_marked or _has_watcher_marker(top)
                candidates = [
                    m for m in top.body
                    if isinstance(m, (ast.AsyncFunctionDef, ast.FunctionDef))
                    and m.name.startswith("test_")
                ]
                for method in candidates:
                    marked = class_marked or _has_watcher_marker(method)
                    if not marked and _contains_watcher_backend_ops(method):
                        offenders.append(f"{path.name}:{method.lineno} {method.name}")
            elif isinstance(top, (ast.AsyncFunctionDef, ast.FunctionDef)):
                if top.name.startswith("test_"):
                    if not (module_marked or _has_watcher_marker(top)) and _contains_watcher_backend_ops(top):
                        offenders.append(f"{path.name}:{top.lineno} {top.name}")

    assert offenders == [], (
        "Watcher-backed realtime tests must declare native_watcher or polling_watcher: "
        + ", ".join(offenders)
    )
