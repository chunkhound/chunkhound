from __future__ import annotations

import json
import os
import sys
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

from scripts import verify_watchman_live_indexing_e2e as live_verifier


def test_prepare_release_runs_watchman_release_verifiers_in_order() -> None:
    prepare_release = (
        Path(__file__).resolve().parents[1] / "scripts" / "prepare_release.sh"
    )
    script_text = prepare_release.read_text(encoding="utf-8")
    runtime_call = (
        "uv run python scripts/verify_watchman_runtime_resources.py "
        '--require-supported-matrix "${WHEEL_PATHS[@]}"'
    )
    installed_live_call = (
        'uv run python scripts/verify_watchman_live_indexing_e2e.py "${WHEEL_PATHS[@]}"'
    )
    source_fallback_call = (
        "uv run python scripts/verify_watchman_live_indexing_e2e.py "
        '--verify-source-fallback --source-root "$PROJECT_ROOT"'
    )

    assert runtime_call in script_text
    assert installed_live_call in script_text
    assert source_fallback_call in script_text
    assert script_text.index(runtime_call) < script_text.index(source_fallback_call)
    assert script_text.index(source_fallback_call) < script_text.index(
        installed_live_call
    )


def test_prepare_release_enforces_supported_matrix_for_runtime_verifier() -> None:
    prepare_release = (
        Path(__file__).resolve().parents[1] / "scripts" / "prepare_release.sh"
    )
    script_text = prepare_release.read_text(encoding="utf-8")

    assert (
        "uv run python scripts/verify_watchman_runtime_resources.py "
        '--require-supported-matrix "${WHEEL_PATHS[@]}"'
    ) in script_text


@pytest.mark.asyncio
async def test_wait_for_ready_requires_nested_watchman_health(monkeypatch) -> None:
    responses = [
        {
            "status": "ready",
            "scan_progress": {
                "realtime": {
                    "watchman_connection_state": "connected",
                    "watchman_subscription_count": 1,
                }
            },
        },
        {
            "status": "ready",
            "scan_progress": {
                "realtime": {
                    "watchman_connection_state": "connected",
                    "watchman_subscription_count": 1,
                    "watchman_subscription_names": ["chunkhound-live-indexing"],
                    "watchman_scopes": [
                        {
                            "subscription_name": "chunkhound-live-indexing",
                            "scope_kind": "primary",
                            "requested_path": "/repo",
                            "watch_root": "/repo",
                            "relative_root": None,
                        }
                    ],
                    "watchman_loss_of_sync": {
                        "count": 0,
                        "fresh_instance_count": 0,
                        "recrawl_count": 0,
                        "disconnect_count": 0,
                        "last_reason": None,
                        "last_at": None,
                        "last_details": None,
                    },
                }
            },
        },
    ]

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        async def send_request(
            self, method: str, params: dict[str, object], timeout: float
        ) -> dict[str, object]:
            del method, params, timeout
            response = responses[self.calls]
            self.calls += 1
            return {"content": [{"type": "text", "text": json.dumps(response)}]}

    async def fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(live_verifier.asyncio, "sleep", fake_sleep)

    client = FakeClient()
    status = await live_verifier._wait_for_ready(client)

    assert client.calls == 2
    assert status["scan_progress"]["realtime"]["watchman_scopes"] == [
        {
            "subscription_name": "chunkhound-live-indexing",
            "scope_kind": "primary",
            "requested_path": "/repo",
            "watch_root": "/repo",
            "relative_root": None,
        }
    ]


def test_source_tree_copy_ignore_excludes_transient_repo_state() -> None:
    ignored = live_verifier._source_tree_copy_ignore(
        ".",
        [
            ".cache",
            ".chunkhound",
            ".git",
            ".ruff_cache",
            ".uv-cache",
            ".uv_cache",
            ".uvcache",
            ".venv",
            "build",
            "dist",
            "keep-me",
        ],
    )

    assert ignored == {
        ".cache",
        ".chunkhound",
        ".git",
        ".ruff_cache",
        ".uv-cache",
        ".uv_cache",
        ".uvcache",
        ".venv",
        "build",
        "dist",
    }


def test_main_runs_host_compatible_wheels_and_source_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    linux_wheel = tmp_path / "chunkhound-0.0.0-py3-none-manylinux_2_34_x86_64.whl"
    windows_wheel = tmp_path / "chunkhound-0.0.0-py3-none-win_amd64.whl"
    linux_wheel.write_text("wheel", encoding="utf-8")
    windows_wheel.write_text("wheel", encoding="utf-8")
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        live_verifier.runtime_verifier,
        "_verify_supported_wheel_matrix",
        lambda paths: {
            "linux-x86_64": Path(paths[0]),
            "windows-x86_64": Path(paths[1]),
        },
    )
    monkeypatch.setattr(
        live_verifier.runtime_verifier,
        "_runtime_platform_for_wheel",
        lambda wheel_path: (
            "linux-x86_64" if "manylinux" in wheel_path.name else "windows-x86_64"
        ),
    )
    monkeypatch.setattr(
        live_verifier.runtime_verifier,
        "_should_verify_runtime_reads",
        lambda runtime_platform: runtime_platform == "linux-x86_64",
    )

    async def fake_verify_wheel(wheel_path: Path) -> None:
        calls.append(("wheel", wheel_path.name))

    async def fake_verify_source_fallback(source_root: Path) -> None:
        calls.append(("source", str(source_root)))

    monkeypatch.setattr(live_verifier, "_verify_wheel", fake_verify_wheel)
    monkeypatch.setattr(
        live_verifier,
        "_verify_source_fallback",
        fake_verify_source_fallback,
    )

    assert (
        live_verifier.main(
            [
                "--require-supported-matrix",
                "--verify-source-fallback",
                "--source-root",
                str(tmp_path),
                str(linux_wheel),
                str(windows_wheel),
            ]
        )
        == 0
    )
    assert calls == [("wheel", linux_wheel.name), ("source", str(tmp_path))]


@pytest.mark.asyncio
async def test_verify_source_fallback_ignores_transient_state_and_checks_contract(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "source-root"
    manifest_root = (
        source_root / "chunkhound" / "watchman_runtime" / "platforms" / "linux-x86_64"
    )
    manifest_root.mkdir(parents=True)
    manifest_root.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "platform": "linux-x86_64",
                "source_url": "https://example.invalid/runtime.deb",
                "support_file_sources": {
                    "lib/example.so": {
                        "source_url": "https://example.invalid/support.deb"
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (source_root / ".chunkhound" / "watchman").mkdir(parents=True)
    (source_root / ".chunkhound" / "watchman" / "sock").write_text(
        "socket-ish",
        encoding="utf-8",
    )
    (source_root / ".uvcache" / "wheels-v5" / "pypi" / "pkg").mkdir(parents=True)
    (source_root / ".uvcache" / "wheels-v5" / "pypi" / "pkg" / "stale").write_text(
        "stale",
        encoding="utf-8",
    )
    (source_root / ".uv-cache").mkdir()
    (source_root / ".uv_cache").mkdir()
    (source_root / ".cache").mkdir()
    (source_root / ".ruff_cache").mkdir()

    work_root = tmp_path / "work"
    editable_source_roots: list[Path] = []
    source_install_roots: list[Path] = []
    sdist_build_roots: list[Path] = []
    sdist_artifacts: list[Path] = []
    default_project_dirs: list[Path] = []
    watchman_project_dirs: list[Path] = []
    daemon_status_descriptions: list[str] = []
    cleanup_roots: list[Path] = []

    monkeypatch.setattr(
        live_verifier.hatch_build,
        "_load_supported_watchman_platforms",
        lambda: {"linux-x86_64"},
    )
    monkeypatch.setattr(
        live_verifier.tempfile,
        "mkdtemp",
        lambda prefix: str(work_root),
    )

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> SimpleNamespace:
        assert check
        assert capture_output
        assert text
        command = tuple(args[:3])
        if command[:2] == ("uv", "venv"):
            venv_dir = Path(args[2])
            python_path = live_verifier._python_path(venv_dir)
            chunkhound_path = live_verifier._chunkhound_path(venv_dir)
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")
            chunkhound_path.write_text("", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command == ("uv", "build", "--sdist"):
            assert cwd is not None
            build_root = Path(cwd)
            sdist_build_roots.append(build_root)
            rewritten_manifest = json.loads(
                (
                    build_root
                    / "chunkhound"
                    / "watchman_runtime"
                    / "platforms"
                    / "linux-x86_64"
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            assert rewritten_manifest["source_url"].startswith(
                live_verifier._DETERMINISTIC_FAILURE_URL_BASE
            )
            dist_dir = Path(args[-1])
            dist_dir.mkdir(parents=True, exist_ok=True)
            sdist_path = dist_dir / "chunkhound-0.0.0.tar.gz"
            sdist_path.write_text("sdist", encoding="utf-8")
            sdist_artifacts.append(sdist_path)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command == ("uv", "pip", "install"):
            assert env is not None
            assert env["SETUPTOOLS_SCM_PRETEND_VERSION"] == "0.0.0"
            assert env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CHUNKHOUND"] == "0.0.0"
            install_target = Path(args[-1])
            if "-e" in args:
                editable_source_roots.append(install_target)
                assert not (install_target / ".chunkhound").exists()
                assert not (install_target / ".uvcache").exists()
                assert not (install_target / ".uv-cache").exists()
                assert not (install_target / ".uv_cache").exists()
                assert not (install_target / ".cache").exists()
                assert not (install_target / ".ruff_cache").exists()
                rewritten_manifest = json.loads(
                    (
                        install_target
                        / "chunkhound"
                        / "watchman_runtime"
                        / "platforms"
                        / "linux-x86_64"
                        / "manifest.json"
                    ).read_text(encoding="utf-8")
                )
                assert rewritten_manifest["source_url"].startswith(
                    live_verifier._DETERMINISTIC_FAILURE_URL_BASE
                )
            elif install_target.suffix == ".gz":
                assert install_target in sdist_artifacts
            else:
                source_install_roots.append(install_target)
                rewritten_manifest = json.loads(
                    (
                        install_target
                        / "chunkhound"
                        / "watchman_runtime"
                        / "platforms"
                        / "linux-x86_64"
                        / "manifest.json"
                    ).read_text(encoding="utf-8")
                )
                assert rewritten_manifest["source_url"].startswith(
                    live_verifier._DETERMINISTIC_FAILURE_URL_BASE
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"Unexpected subprocess.run invocation: {args}")

    monkeypatch.setattr(live_verifier.subprocess, "run", fake_run)

    class FakeReader:
        async def read(self) -> bytes:
            return b""

    class FakeClient:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class FakeProc:
        def __init__(self) -> None:
            self.stderr = FakeReader()

    default_client = FakeClient()

    async def fake_start_proxy_client(
        *,
        chunkhound_exe: Path,
        project_dir: Path,
        env: dict[str, str],
    ) -> tuple[FakeProc, FakeClient]:
        default_project_dirs.append(project_dir)
        assert chunkhound_exe.is_file()
        config = json.loads(
            (project_dir / ".chunkhound.json").read_text(encoding="utf-8")
        )
        assert "realtime_backend" not in config["indexing"]
        assert env["CHUNKHOUND_WATCHMAN_RUNTIME_CACHE_DIR"].endswith(
            str(Path("cache") / "default")
        )
        return FakeProc(), default_client

    async def fake_wait_for_daemon_status(
        client: FakeClient,
        *,
        predicate,
        description: str,
        timeout: float = live_verifier._READY_TIMEOUT_SECONDS,
    ) -> dict[str, object]:
        assert client is default_client
        daemon_status_descriptions.append(description)
        assert timeout == live_verifier._READY_TIMEOUT_SECONDS
        status = {
            "status": "ready",
            "scan_progress": {
                "realtime": {
                    "configured_backend": "watchdog",
                    "effective_backend": "watchdog",
                }
            },
        }
        assert predicate(status)
        return status

    async def fake_run_proxy_to_failure(
        *,
        chunkhound_exe: Path,
        project_dir: Path,
        env: dict[str, str],
        timeout: float = live_verifier._SOURCE_FALLBACK_FAILURE_TIMEOUT_SECONDS,
    ) -> tuple[int, str, str]:
        watchman_project_dirs.append(project_dir)
        assert chunkhound_exe.is_file()
        assert timeout == live_verifier._SOURCE_FALLBACK_FAILURE_TIMEOUT_SECONDS
        config = json.loads(
            (project_dir / ".chunkhound.json").read_text(encoding="utf-8")
        )
        assert config["indexing"]["realtime_backend"] == "watchman"
        assert env["CHUNKHOUND_WATCHMAN_RUNTIME_CACHE_DIR"].endswith(
            str(Path("cache") / "watchman")
        )
        assert env["CHUNKHOUND_WATCHMAN_RUNTIME_DOWNLOAD_TIMEOUT_SECONDS"] == "1"
        assert env["CHUNKHOUND_WATCHMAN_RUNTIME_DOWNLOAD_RETRIES"] == "1"
        daemon_log_path = project_dir / ".chunkhound" / "daemon.log"
        daemon_log_path.parent.mkdir(parents=True, exist_ok=True)
        daemon_log_path.write_text(
            "Watchman runtime archive download failed during source fallback test\n",
            encoding="utf-8",
        )
        return 1, "", "Recent daemon log output\nsimulated watchman failure"

    monkeypatch.setattr(
        live_verifier,
        "_start_proxy_client",
        fake_start_proxy_client,
    )
    monkeypatch.setattr(
        live_verifier,
        "_wait_for_daemon_status",
        fake_wait_for_daemon_status,
    )
    monkeypatch.setattr(
        live_verifier,
        "_run_proxy_to_failure",
        fake_run_proxy_to_failure,
    )
    monkeypatch.setattr(
        live_verifier,
        "_lock_path_for_runtime",
        lambda project_dir, runtime_dir: work_root / "missing.lock",
    )
    monkeypatch.setattr(
        live_verifier,
        "_terminate_processes_using_root",
        lambda root: cleanup_roots.append(root),
    )
    monkeypatch.setattr(
        live_verifier,
        "_remove_tree_with_retries",
        lambda root: cleanup_roots.append(root),
    )

    await live_verifier._verify_source_fallback(source_root)

    assert len(sdist_build_roots) == 1
    assert len(sdist_artifacts) == 1
    assert len(source_install_roots) == 1
    assert len(editable_source_roots) == 1
    assert len(default_project_dirs) == 3
    assert len(watchman_project_dirs) == 3
    assert daemon_status_descriptions == [
        "sdist install watchdog fallback readiness",
        "source install watchdog fallback readiness",
        "editable install watchdog fallback readiness",
    ]
    assert default_client.closed is True
    assert cleanup_roots == [work_root, work_root]


def test_main_requires_wheels_or_source_fallback(capsys) -> None:
    try:
        live_verifier.main([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError(
            "Expected parser failure when no verification mode was supplied"
        )


def test_rewrite_watchman_source_urls_rewrites_manifest_sources(tmp_path: Path) -> None:
    manifest_root = (
        tmp_path / "chunkhound" / "watchman_runtime" / "platforms" / "linux-x86_64"
    )
    manifest_root.mkdir(parents=True)
    manifest_path = manifest_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "platform": "linux-x86_64",
                "source_url": "https://example.invalid/runtime.deb",
                "support_file_sources": {
                    "lib/example.so": {
                        "source_url": "https://example.invalid/support.deb"
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch_platforms = {"linux-x86_64"}
    original = live_verifier.hatch_build._load_supported_watchman_platforms
    live_verifier.hatch_build._load_supported_watchman_platforms = (
        lambda: monkeypatch_platforms
    )
    try:
        live_verifier._rewrite_watchman_source_urls(tmp_path)
    finally:
        live_verifier.hatch_build._load_supported_watchman_platforms = original

    rewritten = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rewritten["source_url"].startswith(
        live_verifier._DETERMINISTIC_FAILURE_URL_BASE
    )
    assert rewritten["support_file_sources"]["lib/example.so"]["source_url"].startswith(
        live_verifier._DETERMINISTIC_FAILURE_URL_BASE
    )


def test_clean_room_env_injects_runtime_dir(tmp_path: Path, monkeypatch) -> None:
    venv_dir = tmp_path / "venv"
    runtime_dir = tmp_path / "runtime"
    bin_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    monkeypatch.setenv("PATH", os.pathsep.join([str(bin_dir), "/usr/bin"]))

    env = live_verifier._clean_room_env(
        venv_dir,
        runtime_dir=runtime_dir,
        extra_env={"TEST_FLAG": "1"},
    )

    assert env["CHUNKHOUND_DAEMON_RUNTIME_DIR"] == str(runtime_dir)
    assert env["TEST_FLAG"] == "1"


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


def test_mcp_command_args_disable_embeddings_for_regex_only_validation(
    tmp_path: Path,
) -> None:
    assert live_verifier._mcp_command_args(tmp_path) == (
        "mcp",
        "--no-embeddings",
        str(tmp_path),
    )


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
            "PATH": (r"C:\runtime\bin;C:\venv\Scripts;C:\Windows\System32"),
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
