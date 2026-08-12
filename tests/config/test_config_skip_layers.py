"""Contract tests for ``Config(skip_layers=...)``.

Locks the public per-layer skip contract before the remote-config pipeline
begins consuming it to construct restricted-merge / half-merged / post-rules
snapshots. Regressions in layer gating would silently corrupt the terminal
delta gate, so the contract lives in tests rather than in comments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.core.config.config import Config


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize CHUNKHOUND_* env vars and point HOME at a scratch dir so
    the developer environment cannot leak into layer merging.
    """
    for key in list(os.environ):
        if key.startswith("CHUNKHOUND_"):
            monkeypatch.delenv(key, raising=False)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


@pytest.fixture
def proj(tmp_path: Path) -> Path:
    """A concrete project directory. `Config.validate_config` sys.exits when
    target_dir does not exist, so every test needs a real path.
    """
    path = tmp_path / "proj"
    path.mkdir()
    return path


def test_default_skip_layers_none_runs_env_layer(
    proj: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHUNKHOUND_DEBUG", "true")
    config = Config(target_dir=proj)
    assert config.debug is True


def test_skip_env(proj: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNKHOUND_DEBUG", "true")
    config = Config(target_dir=proj, skip_layers={"env"})
    assert config.debug is False


def test_skip_cli(proj: Path) -> None:
    args = SimpleNamespace(debug=True, verbose=False)
    without_skip = Config(args=args, target_dir=proj)
    with_skip = Config(args=args, target_dir=proj, skip_layers={"cli"})
    assert without_skip.debug is True
    assert with_skip.debug is False


def test_skip_local_config(proj: Path) -> None:
    (proj / ".chunkhound.json").write_text(
        json.dumps({"debug": True}), encoding="utf-8"
    )
    without_skip = Config(target_dir=proj)
    with_skip = Config(target_dir=proj, skip_layers={"local_config"})
    assert without_skip.debug is True
    assert with_skip.debug is False


def test_skip_global(tmp_path: Path, proj: Path) -> None:
    (tmp_path / "home" / ".chunkhound.json").write_text(
        json.dumps({"debug": True}), encoding="utf-8"
    )
    without_skip = Config(target_dir=proj)
    with_skip = Config(target_dir=proj, skip_layers={"global"})
    assert without_skip.debug is True
    assert with_skip.debug is False


def test_skip_config_file(tmp_path: Path, proj: Path) -> None:
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps({"debug": True}), encoding="utf-8")
    args = SimpleNamespace(config=str(explicit), debug=False, verbose=False)
    without_skip = Config(args=args, target_dir=proj)
    with_skip = Config(args=args, target_dir=proj, skip_layers={"config_file"})
    assert without_skip.debug is True
    assert with_skip.debug is False


def test_skip_all_layers_yields_defaults(
    tmp_path: Path, proj: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every configurable layer set to True, then all layers skipped. The
    resulting Config must equal the model defaults — proves the kwarg gates
    every layer and nothing sneaks through.
    """
    monkeypatch.setenv("CHUNKHOUND_DEBUG", "true")
    (tmp_path / "home" / ".chunkhound.json").write_text(
        json.dumps({"debug": True}), encoding="utf-8"
    )
    (proj / ".chunkhound.json").write_text(
        json.dumps({"debug": True}), encoding="utf-8"
    )
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps({"debug": True}), encoding="utf-8")
    args = SimpleNamespace(config=str(explicit), debug=True, verbose=False)

    config = Config(
        args=args,
        target_dir=proj,
        skip_layers={"env", "global", "local_config", "config_file", "cli"},
    )
    assert config.debug is False


def test_target_dir_from_args_path_survives_layer_skip(proj: Path) -> None:
    """``target_dir`` is extracted from ``args.path`` outside the layer gates,
    so skipping ``local_config`` and ``config_file`` (the exact shape the
    remote-config pipeline's URL-discovery step uses) must still route
    ``args.path`` to ``target_dir``. A future refactor that folded target_dir
    extraction into a layer-application helper would silently break that step.
    """
    args = SimpleNamespace(path=str(proj), config=None, debug=False, verbose=False)
    config = Config(args=args, skip_layers={"local_config", "config_file"})
    assert config.target_dir == proj.resolve()


def test_skip_all_layers_still_applies_direct_kwargs(proj: Path) -> None:
    """Direct kwargs bypass the skip gate — they are merged unconditionally
    after every layer, so callers using ``Config(**overrides)`` for tests or
    programmatic construction retain that path even when the layered inputs
    are fully suppressed.
    """
    config = Config(
        target_dir=proj,
        skip_layers={"env", "global", "local_config", "config_file", "cli"},
        debug=True,
    )
    assert config.debug is True
