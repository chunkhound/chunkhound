"""The release wheel gate installs only downloaded wheels outside the checkout."""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/validate_release_wheels.py"


@pytest.mark.parametrize("custom_directories", [False, True])
def test_combined_wheel_gate_uses_isolated_python_and_real_smoke_script(
    tmp_path, monkeypatch, custom_directories
):
    main_dir = tmp_path / ("main wheels" if custom_directories else "main-dist")
    native_dir = tmp_path / ("native wheels" if custom_directories else "native-dist")
    main_dir.mkdir()
    native_dir.mkdir()
    main_wheel = main_dir / "chunkhound-6.2.0-py3-none-any.whl"
    native_wheel = native_dir / "chunkhound_native-6.2.0-cp310-abi3-win_amd64.whl"
    main_wheel.touch()
    native_wheel.touch()
    (main_dir / "README.txt").touch()
    (native_dir / "nested").mkdir()
    (native_dir / "nested" / "old.whl").touch()

    argv = [str(SCRIPT)]
    if custom_directories:
        argv += ["--main-dist", str(main_dir), "--native-dist", str(native_dir)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.chdir(tmp_path)
    calls = []

    def run(command, *, cwd, check):
        # uv is the external boundary; no installation or network in this test.
        assert check is True
        directory = Path(cwd)
        assert directory.is_dir()
        assert not directory.is_relative_to(ROOT)
        assert directory != tmp_path
        calls.append((command, directory))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    runpy.run_path(str(SCRIPT), run_name="__main__")

    assert len(calls) == 2
    assert calls[0][0] == ["uv", "python", "install", "3.12"]
    assert calls[1][0] == [
        "uv",
        "run",
        "--no-project",
        "--isolated",
        "--python",
        "3.12",
        "--with",
        str(main_wheel),
        "--with",
        str(native_wheel),
        "python",
        str(ROOT / "scripts/smoke_test_native_wheel.py"),
    ]
    assert calls[0][1] == calls[1][1]
    assert not calls[1][1].exists()


@pytest.mark.parametrize(
    ("label", "count"), [("main", 0), ("main", 2), ("native", 0), ("native", 2)]
)
def test_combined_wheel_gate_rejects_missing_or_ambiguous_artifacts(
    tmp_path, monkeypatch, label, count
):
    for kind in ("main", "native"):
        directory = tmp_path / f"{kind}-dist"
        directory.mkdir()
        for index in range(count if kind == label else 1):
            (directory / f"artifact-{index}.whl").touch()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])

    def unexpected_install(*args, **kwargs):
        pytest.fail("Invalid artifact sets must fail before invoking uv")

    monkeypatch.setattr(subprocess, "run", unexpected_install)
    with pytest.raises(
        SystemExit,
        match=f"Expected exactly one matching {label} wheel, found {count}",
    ):
        runpy.run_path(str(SCRIPT), run_name="__main__")


@pytest.mark.parametrize("failed_command", ["install", "smoke"])
def test_combined_wheel_gate_propagates_install_or_smoke_failure(
    tmp_path, monkeypatch, failed_command
):
    for kind in ("main", "native"):
        directory = tmp_path / f"{kind}-dist"
        directory.mkdir()
        (directory / "artifact.whl").touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    calls = []

    def run(command, *, cwd, check):
        calls.append(Path(cwd))
        if failed_command == "install" or command[1] == "run":
            raise subprocess.CalledProcessError(17, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(SystemExit) as failure:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert failure.value.code == 17
    assert len(calls) == (1 if failed_command == "install" else 2)
    assert all(not directory.exists() for directory in calls)
