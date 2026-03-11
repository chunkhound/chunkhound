from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_resolve_bridge_process_accepts_windows_cmd_wrapper(monkeypatch) -> None:
    bridge_child = SimpleNamespace(
        cmdline=lambda: ["python", "-m", "chunkhound.watchman_runtime.bridge"],
    )
    parent = SimpleNamespace(
        cmdline=lambda: ["cmd.exe", "/c", "watchman.cmd", "--foreground"],
        children=lambda recursive=True: [bridge_child],
    )

    monkeypatch.setattr(live_verifier.os, "name", "nt", raising=False)
    monkeypatch.setattr(
        live_verifier.psutil,
        "Process",
        lambda pid: parent if pid == 123 else None,
    )

    resolved = live_verifier._resolve_bridge_process(123)

    assert resolved is bridge_child


def test_remove_tree_with_retries_terminates_windows_processes_using_root(
    tmp_path: Path, monkeypatch
) -> None:
    locked_root = tmp_path / "locked-root"
    locked_root.mkdir()
    terminated: list[int] = []
    original_rmtree = live_verifier.shutil.rmtree
    attempts = {"count": 0}

    class FakeProcess:
        def __init__(self, pid: int, cwd: str | None, cmdline: list[str]) -> None:
            self.info = {"pid": pid, "cwd": cwd, "cmdline": cmdline}

    def flaky_rmtree(path: Path) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise PermissionError("simulated Windows handle delay")
        original_rmtree(path)

    monkeypatch.setattr(live_verifier.os, "name", "nt", raising=False)
    monkeypatch.setattr(live_verifier.shutil, "rmtree", flaky_rmtree)
    monkeypatch.setattr(live_verifier.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(
        live_verifier.psutil,
        "process_iter",
        lambda *_args, **_kwargs: iter(
            [
                FakeProcess(101, str(locked_root), []),
                FakeProcess(202, None, [str(locked_root / "daemon.log")]),
                FakeProcess(303, str(tmp_path / "other"), []),
            ]
        ),
    )
    monkeypatch.setattr(
        live_verifier, "_terminate_process_tree", lambda pid: terminated.append(pid)
    )

    live_verifier._remove_tree_with_retries(locked_root, attempts=2)

    assert attempts["count"] == 2
    assert terminated == [101, 202]
