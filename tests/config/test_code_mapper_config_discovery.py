from __future__ import annotations

from pathlib import Path

import pytest

from chunkhound.core.config.config import Config


def test_config_does_not_treat_code_mapper_scope_as_target_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_root = tmp_path / "workspace"
    scope = workspace_root / "scope"
    scope.mkdir(parents=True, exist_ok=True)

    # Ensure project root detection can succeed.
    (workspace_root / ".chunkhound.json").write_text("{}", encoding="utf-8")

    monkeypatch.chdir(workspace_root)

    class Args:
        def __init__(self) -> None:
            self.command = "map"
            self.path = Path("scope")
            self.config = None

    cfg = Config(args=Args())
    assert cfg.target_dir == workspace_root


def test_config_uses_config_file_dir_as_code_mapper_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    cfg_path = workspace_root / ".chunkhound.json"
    cfg_path.write_text("{}", encoding="utf-8")

    other_dir = tmp_path / "other"
    other_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(other_dir)

    class Args:
        def __init__(self) -> None:
            self.command = "map"
            self.path = Path("scope")
            self.config = cfg_path

    cfg = Config(args=Args())
    assert cfg.target_dir == workspace_root
