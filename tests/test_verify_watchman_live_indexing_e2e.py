from __future__ import annotations

import os
import sys
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

from scripts import verify_watchman_live_indexing_e2e as live_verifier


def test_prepare_release_runs_watchman_release_verifiers_in_order() -> None:
    prepare_release = (
        Path(__file__).resolve().parents[1] / "scripts" / "prepare_release.sh"
    )
    script_text = prepare_release.read_text(encoding="utf-8")
    runtime_call = (
        'uv run python scripts/verify_watchman_runtime_resources.py "${WHEEL_PATHS[@]}"'
    )
    live_call = (
        'uv run python scripts/verify_watchman_live_indexing_e2e.py "${WHEEL_PATHS[@]}"'
    )

    assert runtime_call in script_text
    assert live_call in script_text
    assert script_text.index(runtime_call) < script_text.index(live_call)


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


def test_resolve_native_watchman_process_accepts_matching_binary_path(
    monkeypatch,
) -> None:
    process = SimpleNamespace(
        cmdline=lambda: ["/tmp/runtime/watchman", "--foreground"],
        environ=lambda: {"PATH": "/tmp/venv/bin:/usr/bin", "VIRTUAL_ENV": "/tmp/venv"},
    )
    monkeypatch.setattr(
        live_verifier.psutil,
        "Process",
        lambda pid: process if pid == 123 else None,
    )

    resolved = live_verifier._resolve_native_watchman_process(
        123,
        expected_binary_path="/tmp/runtime/watchman",
    )

    assert resolved is process


def test_resolve_native_watchman_process_normalizes_path_variants(
    monkeypatch,
) -> None:
    process = SimpleNamespace(
        cmdline=lambda: [r"C:\Runtime\WATCHMAN.EXE", "--foreground"],
        environ=lambda: {
            "PATH": r"C:\venv\Scripts;C:\Windows\System32",
            "VIRTUAL_ENV": r"C:\venv",
        },
    )
    monkeypatch.setattr(
        live_verifier.psutil,
        "Process",
        lambda pid: process if pid == 456 else None,
    )
    monkeypatch.setattr(
        live_verifier.os.path,
        "normcase",
        lambda value: str(value).replace("/", "\\").lower(),
    )
    monkeypatch.setattr(
        live_verifier.os.path,
        "normpath",
        lambda value: str(value).replace("/", "\\"),
    )

    resolved = live_verifier._resolve_native_watchman_process(
        456,
        expected_binary_path="C:/Runtime/watchman.exe",
    )

    assert resolved is process


def test_assert_sidecar_uses_installed_runtime_accepts_runtime_bin_then_venv_on_windows(
    monkeypatch,
) -> None:
    process = SimpleNamespace(
        environ=lambda: {
            "PATH": (
                r"C:\runtime\bin;C:\venv\Scripts;C:\Windows\System32"
            ),
            "VIRTUAL_ENV": r"C:\venv",
        }
    )
    monkeypatch.setattr(
        live_verifier,
        "_resolve_native_watchman_process",
        lambda pid, expected_binary_path: process,
    )
    monkeypatch.setattr(
        live_verifier,
        "_python_path",
        lambda venv_dir: PureWindowsPath(r"C:\venv\Scripts\python.exe"),
    )
    monkeypatch.setattr(live_verifier.os, "pathsep", ";", raising=False)
    monkeypatch.setattr(
        live_verifier.os.path,
        "normcase",
        lambda value: str(value).replace("/", "\\").lower(),
    )
    monkeypatch.setattr(
        live_verifier.os.path,
        "normpath",
        lambda value: str(value).replace("/", "\\"),
    )

    live_verifier._assert_sidecar_uses_installed_runtime(
        {
            "watchman_pid": 123,
            "watchman_binary_path": r"C:\runtime\bin\watchman.exe",
        },
        venv_dir=Path(r"C:\venv"),
    )


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
