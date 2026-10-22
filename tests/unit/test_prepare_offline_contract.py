"""Contract for offline artifact staging (no network, no real downloads)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest


def _load_prepare() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "prepare_offline_tests.py"
    spec = importlib.util.spec_from_file_location("prepare_offline_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_offline_tests"] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_installs_vss_and_materializes_watchman(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VSS install/load runs; Watchman materializes only when supported."""
    module = _load_prepare()
    calls: list[str] = []
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = False
    connection.execute.side_effect = lambda sql: calls.append(sql)
    monkeypatch.setattr(module.duckdb, "connect", lambda: connection)
    monkeypatch.setattr(module, "is_packaged_watchman_runtime_supported", lambda: True)
    materialize = MagicMock()
    monkeypatch.setattr(module, "materialize_watchman_binary", materialize)
    module.main()
    assert "INSTALL vss" in calls
    assert "LOAD vss" in calls
    materialize.assert_called_once_with()


def test_prepare_skips_watchman_when_unsupported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unsupported platforms log and skip instead of failing."""
    module = _load_prepare()
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = False
    monkeypatch.setattr(module.duckdb, "connect", lambda: connection)
    monkeypatch.setattr(module, "is_packaged_watchman_runtime_supported", lambda: False)
    materialize = MagicMock()
    monkeypatch.setattr(module, "materialize_watchman_binary", materialize)
    module.main()
    materialize.assert_not_called()
    assert "not supported" in capsys.readouterr().out
