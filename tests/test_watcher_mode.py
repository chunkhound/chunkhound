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
