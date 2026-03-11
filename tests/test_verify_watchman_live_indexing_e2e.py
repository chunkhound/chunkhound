from __future__ import annotations

import os
import sys
from pathlib import Path

from scripts import verify_watchman_live_indexing_e2e as live_verifier


def test_mcp_env_prefers_installed_venv_and_clears_repo_python_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_venv = tmp_path / "repo-venv"
    repo_bin = repo_venv / ("Scripts" if os.name == "nt" else "bin")
    repo_bin.mkdir(parents=True)
    installed_venv = tmp_path / "installed-venv"
    installed_bin = installed_venv / ("Scripts" if os.name == "nt" else "bin")
    installed_bin.mkdir(parents=True)

    monkeypatch.setenv("PATH", os.pathsep.join([str(repo_bin), "/usr/bin", "/bin"]))
    monkeypatch.setenv("VIRTUAL_ENV", str(repo_venv))
    monkeypatch.setenv("PYTHONPATH", "/tmp/repo-pythonpath")
    monkeypatch.setenv("PYTHONHOME", "/tmp/repo-pythonhome")

    env = live_verifier._mcp_env(installed_venv)

    assert env["VIRTUAL_ENV"] == str(installed_venv)
    assert env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert env["PATH"].split(os.pathsep)[0] == str(installed_bin)
    assert str(repo_bin) not in env["PATH"].split(os.pathsep)

    if sys.prefix:
        assert env["PATH"].split(os.pathsep)[0] != str(Path(sys.prefix) / "bin")
