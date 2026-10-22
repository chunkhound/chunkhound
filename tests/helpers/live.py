"""Explicit live inputs and a default Python socket boundary.

Only Linux CI adds a native/subprocess firewall (network namespace).
macOS and Windows get only the in-process Python socket guard; subprocess
and native traffic there is not isolated.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.network import restore_network, restrict_network

LIVE_KEY = pytest.StashKey[dict[str, Any]]()
_RUN_CONFIG: pytest.Config | None = None


def add_live_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("live")
    group.addoption("--run-live", action="store_true", help="Run marked live tests")
    group.addoption(
        "--live-config", help="Absolute JSON file with explicit live inputs"
    )


def configure_live(config: pytest.Config) -> None:
    global _RUN_CONFIG
    _RUN_CONFIG = config
    worker = getattr(config, "workerinput", {})
    config.stash[LIVE_KEY] = (
        worker["live_inputs"] if "live_inputs" in worker else _read_inputs(config)
    )
    config.addinivalue_line("markers", "live: real external provider or CLI request")
    if config.getoption("force_enable_socket"):
        raise pytest.UsageError("External sockets require a live test and --run-live")
    # Own the full lifetime, including collection and session finalizers.
    config.pluginmanager.unregister(name="socket")
    restrict_network()
    config.add_cleanup(restore_network)


def _read_inputs(config: pytest.Config) -> dict[str, Any]:
    filename = config.getoption("live_config")
    if not filename:
        return {}
    path = Path(filename)
    if not path.is_absolute():
        raise pytest.UsageError("--live-config must be an absolute JSON path")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise pytest.UsageError(f"Cannot read --live-config: {exc}") from exc
    if not isinstance(data, dict):
        raise pytest.UsageError("--live-config must contain a JSON object")
    return data


def live_inputs(section: str) -> dict[str, Any]:
    if _RUN_CONFIG is None or not _RUN_CONFIG.getoption("run_live"):
        raise RuntimeError("Live inputs require --run-live")
    data = _RUN_CONFIG.stash[LIVE_KEY].get(section)
    if not isinstance(data, dict) or not data:
        pytest.fail(f"Live prerequisite missing: --live-config section '{section}'")
    return dict(data)


def select_live(config: pytest.Config, items: list[pytest.Function]) -> None:
    if config.getoption("run_live"):
        return
    selected = [item for item in items if item.get_closest_marker("live") is None]
    deselected = [item for item in items if item.get_closest_marker("live") is not None]
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def prepare_live_test(item: pytest.Function) -> None:
    if item.get_closest_marker("live") is not None:
        if not item.config.getoption("run_live"):
            pytest.fail("Live tests require --run-live")
        if not item.config.stash[LIVE_KEY]:
            pytest.fail("Live prerequisite missing: supply --live-config absolute JSON")
        restore_network()
    elif (
        item.get_closest_marker("enable_socket")
        or "socket_enabled" in item.fixturenames
    ):
        pytest.fail("Only live-marked tests may enable external sockets")
    else:
        restrict_network()


def apply_live_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only explicitly listed credential values enter the private test home."""
    data = live_inputs("environment")
    allowed = ("OPENAI_", "ANTHROPIC_", "OPENCODE_", "CHUNKHOUND_LLM_")
    for key, value in data.items():
        if not key.startswith(allowed) or not isinstance(value, str):
            pytest.fail(f"Unsupported live environment input: {key}")
        monkeypatch.setenv(key, value)


def require_cli(name: str) -> dict[str, Any]:
    import shutil

    data = live_inputs(name)
    binary = data.get("binary", name)
    if not isinstance(binary, str) or not shutil.which(binary):
        pytest.fail(f"Live prerequisite missing: {name} executable")
    return data
