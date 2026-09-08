from __future__ import annotations

import asyncio
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
        "uv run python scripts/verify_watchman_live_indexing_e2e.py "
        '--native-wheel "$NATIVE_WHEEL_PATH" "${WHEEL_PATHS[@]}"'
    )
    source_fallback_call = (
        "uv run python scripts/verify_watchman_live_indexing_e2e.py "
        '--native-wheel "$NATIVE_WHEEL_PATH" --verify-source-fallback '
        '--source-root "$PROJECT_ROOT"'
    )

    assert runtime_call in script_text
    assert installed_live_call in script_text
    assert source_fallback_call in script_text
    assert script_text.index(runtime_call) < script_text.index(source_fallback_call)
    assert script_text.index(source_fallback_call) < script_text.index(
        installed_live_call
    )


def test_rollout_gate_workflow_runs_single_aggregate_fallback_proof() -> None:
    """The supported rollout contract (Step 105) is a single aggregate
    fallback proof on one host, not a per-host duplication. Lock the
    workflow shape so doc/workflow drift is caught in CI."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("PyYAML not available")

    workflow_path = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    )
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]

    assert "watchman-runtime-validation" in jobs
    assert "watchman-rollout-gate" in jobs

    runtime_validation = jobs["watchman-runtime-validation"]
    runtime_oses = runtime_validation["strategy"]["matrix"]["os"]
    assert "ubuntu-latest" in runtime_oses
    assert "windows-latest" in runtime_oses

    rollout_gate = jobs["watchman-rollout-gate"]
    assert rollout_gate.get("needs") == "watchman-runtime-validation"
    assert "strategy" not in rollout_gate, (
        "watchman-rollout-gate must stay a single aggregate job; the "
        "fallback proof is intentionally platform-neutral and not matrixed"
    )
    runs_on = rollout_gate["runs-on"]
    assert runs_on == "ubuntu-latest", (
        f"rollout gate should run on a single aggregate host, got {runs_on}"
    )

    step_runs = [
        (step.get("name", ""), step.get("run", "")) for step in rollout_gate["steps"]
    ]
    joined_runs = "\n".join(run for _, run in step_runs)
    assert "--require-supported-matrix" in joined_runs, (
        "rollout gate must enforce the full supported wheel matrix"
    )
    assert "--verify-source-fallback" in joined_runs, (
        "rollout gate must invoke the aggregate sdist/source/editable "
        "fallback proof exactly once"
    )
    assert joined_runs.count("--verify-source-fallback") == 1, (
        "fallback proof should run exactly once in the aggregate job"
    )

    for name, run in step_runs:
        if "--verify-source-fallback" in run:
            assert "--source-root" in run, (
                f"fallback proof step {name!r} must pin --source-root"
            )


def test_ci_wires_repaired_native_wheel_into_watchman_verification() -> None:
    """Watchman verification must run against the full repaired
    chunkhound_native wheel (never a source-tree extension patch): each
    consuming job must build it into a dedicated out-dir and pass it via
    --native-wheel, and macOS PR coverage must smoke the repaired wheel."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("PyYAML not available")

    workflow_path = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    )
    jobs = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))["jobs"]

    def native_wheel_builds(job: dict) -> list[dict]:
        return [
            step
            for step in job["steps"]
            if step.get("uses", "").endswith("/build-native-wheel")
        ]

    runtime_validation = jobs["watchman-runtime-validation"]
    runtime_builds = native_wheel_builds(runtime_validation)
    assert len(runtime_builds) == 1
    assert runtime_builds[0]["with"]["out-dir"] == "native-dist"
    e2e_runs = [
        step.get("run", "")
        for step in runtime_validation["steps"]
        if "--native-wheel" in step.get("run", "")
    ]
    assert len(e2e_runs) == 1
    assert "native-dist/chunkhound_native-*.whl" in e2e_runs[0]
    assert "Expected exactly one repaired native wheel" in e2e_runs[0], (
        "installed-wheel e2e must guard against missing or ambiguous "
        "repaired-wheel globs"
    )

    rollout_gate = jobs["watchman-rollout-gate"]
    gate_builds = native_wheel_builds(rollout_gate)
    assert len(gate_builds) == 1
    assert gate_builds[0]["with"]["out-dir"] == "native-dist"
    gate_fallback_runs = [
        step.get("run", "")
        for step in rollout_gate["steps"]
        if "--verify-source-fallback" in step.get("run", "")
    ]
    assert len(gate_fallback_runs) == 1
    assert "--native-wheel" in gate_fallback_runs[0]
    assert "native-dist/chunkhound_native-*.whl" in gate_fallback_runs[0]
    assert "Expected exactly one repaired native wheel" in gate_fallback_runs[0], (
        "fallback proof must guard against missing or ambiguous repaired-wheel globs"
    )

    macos_builds = native_wheel_builds(jobs["native-dev-macos"])
    assert len(macos_builds) == 1
    assert macos_builds[0]["with"]["out-dir"] == "native-dist"


def test_build_native_wheel_action_defaults_out_dir_to_dist() -> None:
    """release.yml/release-rc.yml call build-native-wheel without out-dir and
    upload dist/*.whl afterwards -- the input default must stay "dist" and
    every output location (maturin --out, smoke-test wheel dir) must follow
    inputs.out-dir."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("PyYAML not available")

    action_path = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "actions"
        / "build-native-wheel"
        / "action.yml"
    )
    action = yaml.safe_load(action_path.read_text(encoding="utf-8"))
    out_dir = action["inputs"]["out-dir"]
    assert out_dir["required"] is False
    assert out_dir["default"] == "dist"

    runs = {step["name"]: step.get("run", "") for step in action["runs"]["steps"]}
    build_run = runs["Build native wheel"]
    smoke_run = runs["Smoke test native wheel API"]
    assert build_run.count('--out "${{ inputs.out-dir }}/"') == 2, (
        "both Linux and non-Linux maturin invocations must write into out-dir"
    )
    assert 'WHEEL_DIR="$WORKSPACE_DIR/${{ inputs.out-dir }}"' in smoke_run


def test_prepare_release_enforces_supported_matrix_for_runtime_verifier() -> None:
    prepare_release = (
        Path(__file__).resolve().parents[1] / "scripts" / "prepare_release.sh"
    )
    script_text = prepare_release.read_text(encoding="utf-8")

    assert (
        "uv run python scripts/verify_watchman_runtime_resources.py "
        '--require-supported-matrix "${WHEEL_PATHS[@]}"'
    ) in script_text


def test_prepare_release_keeps_ci_owned_publish_contract() -> None:
    prepare_release = (
        Path(__file__).resolve().parents[1] / "scripts" / "prepare_release.sh"
    )
    script_text = prepare_release.read_text(encoding="utf-8")

    assert "gh release create <tag>" in script_text
    assert "--generate-notes" in script_text
    assert "gh release edit <tag> --draft=false" in script_text
    assert "Do not run uv publish manually" in script_text


def test_prepare_release_requires_matching_native_wheel_env() -> None:
    """The watchman verifiers now require a complete repaired
    chunkhound-native wheel; prepare_release.sh must fail loudly unless
    CHUNKHOUND_NATIVE_WHEEL points at an existing .whl file."""
    prepare_release = (
        Path(__file__).resolve().parents[1] / "scripts" / "prepare_release.sh"
    )
    script_text = prepare_release.read_text(encoding="utf-8")

    assert 'NATIVE_WHEEL_PATH="${CHUNKHOUND_NATIVE_WHEEL:-}"' in script_text
    assert '[[ ! -f "$NATIVE_WHEEL_PATH" || "$NATIVE_WHEEL_PATH" != *.whl ]]' in (
        script_text
    )
    assert (
        "❌ Set CHUNKHOUND_NATIVE_WHEEL to the matching repaired "
        "chunkhound-native wheel." in script_text
    )
    assert '--native-wheel "$NATIVE_WHEEL_PATH"' in script_text


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
                    "watchman_sidecar_state": "running",
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
                        "translation_failure_count": 0,
                        "subscription_pdu_dropped_count": 0,
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


@pytest.mark.asyncio
async def test_wait_for_ready_accepts_only_polling_fallback_mode(
    monkeypatch,
) -> None:
    response = {
        "status": "ready",
        "scan_progress": {
            "realtime": {
                "watchman_sidecar_state": "stopped",
                "effective_backend": "polling",
            }
        },
    }

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        async def send_request(
            self, method: str, params: dict[str, object], timeout: float
        ) -> dict[str, object]:
            del method, params, timeout
            self.calls += 1
            return {"content": [{"type": "text", "text": json.dumps(response)}]}

    async def fake_sleep(_delay: float) -> None:
        raise AssertionError("Polling fallback mode should return immediately")

    monkeypatch.setattr(live_verifier.asyncio, "sleep", fake_sleep)

    client = FakeClient()
    status = await live_verifier._wait_for_ready(client)

    assert client.calls == 1
    assert status["scan_progress"]["realtime"]["effective_backend"] == "polling"


@pytest.mark.asyncio
async def test_json_rpc_eof_awaits_process_exit_before_reporting_returncode() -> None:
    class EofStdout:
        def __init__(self) -> None:
            self._release = asyncio.Event()

        def release(self) -> None:
            self._release.set()

        async def readline(self) -> bytes:
            await self._release.wait()
            return b""

    class FakeStdin:
        def __init__(self, stdout: EofStdout) -> None:
            self._stdout = stdout
            self.writes: list[bytes] = []

        def write(self, data: bytes) -> None:
            self.writes.append(data)

        async def drain(self) -> None:
            self._stdout.release()

        def close(self) -> None:
            return None

        async def wait_closed(self) -> None:
            return None

    class FakeStderr:
        async def read(self) -> bytes:
            return b"daemon failed\n"

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdout = EofStdout()
            self.stdin = FakeStdin(self.stdout)
            self.stderr = FakeStderr()
            self.wait_calls = 0

        async def wait(self) -> int:
            self.wait_calls += 1
            self.returncode = 7
            return self.returncode

        def terminate(self) -> None:
            raise AssertionError("process already exited after EOF wait")

        def kill(self) -> None:
            raise AssertionError("process already exited after EOF wait")

    process = FakeProcess()
    client = live_verifier.SubprocessJsonRpcClient(process)  # type: ignore[arg-type]
    await client.start()
    try:
        with pytest.raises(RuntimeError) as exc_info:
            await client.send_request("tools/call", {}, timeout=1.0)
    finally:
        await client.close()

    message = str(exc_info.value)
    assert process.wait_calls == 1
    assert "rc=7" in message
    assert "rc=None" not in message
    assert "stderr_tail='daemon failed'" in message


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
    native_wheel = tmp_path / "chunkhound_native-0.0.0.whl"
    native_wheel.write_text("wheel", encoding="utf-8")
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

    async def fake_verify_wheel(wheel_path: Path, native_wheel: Path) -> None:
        calls.append(("wheel", f"{wheel_path.name}:{native_wheel.name}"))

    async def fake_verify_source_fallback(
        source_root: Path, native_wheel: Path
    ) -> None:
        calls.append(("source", f"{source_root}:{native_wheel.name}"))

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
                "--native-wheel",
                str(native_wheel),
                str(linux_wheel),
                str(windows_wheel),
            ]
        )
        == 0
    )
    assert calls == [
        ("wheel", f"{linux_wheel.name}:{native_wheel.name}"),
        ("source", f"{tmp_path}:{native_wheel.name}"),
    ]


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
    native_wheel = tmp_path / "chunkhound_native-0.0.0.whl"
    native_wheel.write_text("wheel", encoding="utf-8")
    editable_source_roots: list[Path] = []
    source_install_roots: list[Path] = []
    sdist_build_roots: list[Path] = []
    sdist_artifacts: list[Path] = []
    native_install_wheels: list[Path] = []
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
            if "--no-deps" in args:
                assert "--force-reinstall" in args
                assert Path(args[-1]) == native_wheel
                native_install_wheels.append(native_wheel)
                return SimpleNamespace(returncode=0, stdout="", stderr="")
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

    await live_verifier._verify_source_fallback(source_root, native_wheel)

    assert len(sdist_build_roots) == 1
    assert len(sdist_artifacts) == 1
    assert len(source_install_roots) == 1
    assert len(editable_source_roots) == 1
    assert native_install_wheels == [native_wheel, native_wheel, native_wheel]
    assert len(default_project_dirs) == 3
    assert len(watchman_project_dirs) == 3
    assert daemon_status_descriptions == [
        "sdist install watchdog fallback readiness",
        "source install watchdog fallback readiness",
        "editable install watchdog fallback readiness",
    ]
    assert default_client.closed is True
    assert cleanup_roots == [work_root, work_root]


def test_install_native_wheel_uses_force_reinstall_no_deps_contract(
    tmp_path: Path, monkeypatch
) -> None:
    """The repaired native wheel must fully replace the PyPI-resolved
    chunkhound-native inside the target venv: --force-reinstall (uv would
    otherwise no-op on the same version), --no-deps (the branch wheel can be
    ahead of pyproject's version constraint), targeting the venv's own
    interpreter."""
    captured: list[list[str]] = []

    def fake_run(args, **kwargs):
        captured.append(list(args))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(live_verifier.subprocess, "run", fake_run)
    python_path = tmp_path / "venv" / "bin" / "python"
    native_wheel = tmp_path / "chunkhound_native-0.0.0-cp312-abi3.whl"
    native_wheel.write_text("wheel", encoding="utf-8")

    live_verifier._install_native_wheel(
        python_path=python_path, native_wheel=native_wheel
    )

    assert captured == [
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_path),
            "--force-reinstall",
            "--no-deps",
            str(native_wheel),
        ]
    ]


def test_install_native_wheel_surfaces_uv_failure_on_error(
    tmp_path: Path, monkeypatch
) -> None:
    """uv runs with capture_output=True, so a failed install must re-raise
    with uv's own diagnostics embedded -- otherwise the CI log would only
    show a bare return code with no reason for the failure."""

    def fake_run(args, **kwargs):
        raise live_verifier.subprocess.CalledProcessError(
            returncode=1,
            cmd=list(args),
            output="",
            stderr="uv error: wheel is not compatible with the interpreter",
        )

    monkeypatch.setattr(live_verifier.subprocess, "run", fake_run)
    python_path = tmp_path / "venv" / "bin" / "python"
    native_wheel = tmp_path / "chunkhound_native-0.0.0-cp312-abi3.whl"
    native_wheel.write_text("wheel", encoding="utf-8")

    with pytest.raises(RuntimeError, match="wheel is not compatible") as exc_info:
        live_verifier._install_native_wheel(
            python_path=python_path, native_wheel=native_wheel
        )

    assert isinstance(
        exc_info.value.__cause__, live_verifier.subprocess.CalledProcessError
    )
    assert str(native_wheel) in str(exc_info.value)


@pytest.mark.parametrize(
    ("native_wheel_name", "create_file"),
    [("chunkhound_native-0.0.0.whl", False), ("not-a-wheel.txt", True)],
)
def test_main_rejects_missing_or_non_wheel_native_wheel(
    tmp_path: Path, native_wheel_name: str, create_file: bool
) -> None:
    """``--native-wheel`` must point at an existing repaired .whl file; any
    other value must fail loudly before any install runs."""
    native_wheel = tmp_path / native_wheel_name
    if create_file:
        native_wheel.write_text("wheel", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as exc_info:
        live_verifier.main(
            ["--native-wheel", str(native_wheel), "--verify-source-fallback"]
        )

    assert str(native_wheel) in str(exc_info.value)


def test_main_fails_when_no_host_compatible_wheels(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """``main`` must fail loudly when wheel paths produce no compatible runs."""
    macos_wheel = tmp_path / "chunkhound-0.0.0-py3-none-macosx_11_0_arm64.whl"
    macos_wheel.write_text("wheel", encoding="utf-8")

    monkeypatch.setattr(
        live_verifier.runtime_verifier,
        "_verify_supported_wheel_matrix",
        lambda paths: {},
    )
    monkeypatch.setattr(
        live_verifier,
        "_host_compatible_wheels",
        lambda paths: [],
    )

    def _should_not_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("verifier must not run when host has no compatible wheels")

    monkeypatch.setattr(live_verifier, "_verify_wheel", _should_not_run)

    native_wheel = tmp_path / "chunkhound_native-0.0.0.whl"
    native_wheel.write_text("wheel", encoding="utf-8")
    exit_code = live_verifier.main(
        ["--native-wheel", str(native_wheel), str(macos_wheel)]
    )
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "no host-compatible wheels" in captured.err


def test_main_source_fallback_only_returns_zero_with_no_compatible_wheels(
    tmp_path: Path, monkeypatch
) -> None:
    """A ``--verify-source-fallback``-only invocation must remain unaffected."""
    monkeypatch.setattr(
        live_verifier,
        "_host_compatible_wheels",
        lambda paths: [],
    )

    async def fake_verify_source_fallback(
        source_root: Path, native_wheel: Path
    ) -> None:
        return None

    monkeypatch.setattr(
        live_verifier,
        "_verify_source_fallback",
        fake_verify_source_fallback,
    )

    native_wheel = tmp_path / "chunkhound_native-0.0.0.whl"
    native_wheel.write_text("wheel", encoding="utf-8")
    exit_code = live_verifier.main(
        [
            "--native-wheel",
            str(native_wheel),
            "--verify-source-fallback",
            "--source-root",
            str(tmp_path),
        ]
    )
    assert exit_code == 0


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
    assert env["CHUNKHOUND_DAEMON_REGISTRY_DIR"] == str(
        runtime_dir / "daemon-user-registry"
    )
    assert env["TEST_FLAG"] == "1"


@pytest.mark.asyncio
async def test_verify_wheel_uses_clean_room_runtime_env(
    tmp_path: Path, monkeypatch
) -> None:
    wheel_path = tmp_path / "chunkhound.whl"
    wheel_path.write_text("wheel", encoding="utf-8")
    work_root = tmp_path / "work-root"
    captured_env: dict[str, str] = {}
    cleanup_roots: list[Path] = []

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
        del check, capture_output, text, env, cwd
        if args[:2] == ["uv", "venv"]:
            venv_dir = Path(args[2])
            bin_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
            bin_dir.mkdir(parents=True, exist_ok=True)
            exe_name = "chunkhound.exe" if os.name == "nt" else "chunkhound"
            python_name = "python.exe" if os.name == "nt" else "python"
            (bin_dir / exe_name).write_text("", encoding="utf-8")
            (bin_dir / python_name).write_text("", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:3] == ["uv", "pip", "install"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"Unexpected subprocess call: {args}")

    class FakePipe:
        async def read(self) -> bytes:
            return b""

    class FakeProcess:
        def __init__(self) -> None:
            self.stderr = FakePipe()

    class FakeClient:
        def __init__(self, process: object) -> None:
            del process

        async def start(self) -> None:
            return None

        async def send_request(
            self,
            method: str,
            params: dict[str, object] | None = None,
            timeout: float = 5.0,
        ) -> dict[str, object]:
            del method, params, timeout
            return {}

        async def send_notification(
            self, method: str, params: dict[str, object] | None = None
        ) -> None:
            del method, params

        async def close(self) -> None:
            return None

    async def fake_create_subprocess_exec_safe(
        *args: str,
        stdin: object = None,
        stdout: object = None,
        stderr: object = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> FakeProcess:
        del args, stdin, stdout, stderr, cwd
        assert env is not None
        captured_env.update(env)
        return FakeProcess()

    ready_status = {
        "scan_progress": {
            "realtime": {
                "watchman_sidecar_state": "running",
                "watchman_pid": 1,
                "watchman_binary_path": "/tmp/watchman",
                "watchman_subscription_names": ["chunkhound-live-indexing"],
                "watchman_subscription_pdu_count": 1,
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
                    "translation_failure_count": 0,
                    "subscription_pdu_dropped_count": 0,
                    "last_reason": None,
                    "last_at": None,
                    "last_details": None,
                },
            }
        }
    }

    async def fake_wait_for_ready(client: object) -> dict[str, object]:
        del client
        return ready_status

    async def fake_wait_for_search_hit(client: object, query: str) -> None:
        del client, query
        return None

    monkeypatch.setattr(live_verifier.subprocess, "run", fake_run)
    monkeypatch.setattr(
        live_verifier,
        "_create_subprocess_exec_safe",
        fake_create_subprocess_exec_safe,
    )
    monkeypatch.setattr(live_verifier, "SubprocessJsonRpcClient", FakeClient)
    monkeypatch.setattr(live_verifier, "_wait_for_ready", fake_wait_for_ready)
    monkeypatch.setattr(live_verifier, "_wait_for_search_hit", fake_wait_for_search_hit)
    monkeypatch.setattr(
        live_verifier,
        "_assert_sidecar_uses_installed_runtime",
        lambda realtime, *, venv_dir: None,
    )
    monkeypatch.setattr(
        live_verifier,
        "_parse_tool_json",
        lambda result: ready_status,
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

    await live_verifier._verify_wheel(wheel_path, wheel_path)

    runtime_dir = work_root / "runtime"
    assert captured_env["CHUNKHOUND_DAEMON_RUNTIME_DIR"] == str(runtime_dir)
    assert captured_env["CHUNKHOUND_DAEMON_REGISTRY_DIR"] == str(
        runtime_dir / "daemon-user-registry"
    )
    assert cleanup_roots == [work_root, work_root]


@pytest.mark.asyncio
async def test_verify_wheel_proves_live_searchability_in_polling_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    wheel_path = tmp_path / "chunkhound.whl"
    wheel_path.write_text("wheel", encoding="utf-8")
    work_root = tmp_path / "fallback-work-root"
    wrote_live_file = False
    search_queries: list[str] = []

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
        del check, capture_output, text, env, cwd
        if args[:2] == ["uv", "venv"]:
            venv_dir = Path(args[2])
            bin_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
            bin_dir.mkdir(parents=True, exist_ok=True)
            exe_name = "chunkhound.exe" if os.name == "nt" else "chunkhound"
            python_name = "python.exe" if os.name == "nt" else "python"
            (bin_dir / exe_name).write_text("", encoding="utf-8")
            (bin_dir / python_name).write_text("", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:3] == ["uv", "pip", "install"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"Unexpected subprocess call: {args}")

    class FakePipe:
        async def read(self) -> bytes:
            return b""

    class FakeProcess:
        def __init__(self) -> None:
            self.stderr = FakePipe()

    class FakeClient:
        def __init__(self, process: object) -> None:
            del process
            self.requests = 0

        async def start(self) -> None:
            return None

        async def send_request(
            self,
            method: str,
            params: dict[str, object] | None = None,
            timeout: float = 5.0,
        ) -> dict[str, object]:
            del method, params, timeout
            self.requests += 1
            return {}

        async def send_notification(
            self, method: str, params: dict[str, object] | None = None
        ) -> None:
            del method, params

        async def close(self) -> None:
            return None

    async def fake_create_subprocess_exec_safe(
        *args: str,
        stdin: object = None,
        stdout: object = None,
        stderr: object = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> FakeProcess:
        del args, stdin, stdout, stderr, env, cwd
        return FakeProcess()

    ready_status = {
        "scan_progress": {
            "realtime": {
                "watchman_sidecar_state": "stopped",
                "effective_backend": "polling",
            }
        }
    }

    async def fake_wait_for_ready(client: object) -> dict[str, object]:
        del client
        return ready_status

    async def fake_wait_for_search_hit(client: object, query: str) -> None:
        del client
        search_queries.append(query)
        return None

    def fake_parse_tool_json(_result: object) -> dict[str, object]:
        return ready_status

    def fake_write_project(project_dir: Path) -> tuple[Path, str]:
        live_file = project_dir / "pkg" / "live.py"
        return live_file, "fallback_live_symbol"

    monkeypatch.setattr(live_verifier.subprocess, "run", fake_run)
    monkeypatch.setattr(
        live_verifier,
        "_create_subprocess_exec_safe",
        fake_create_subprocess_exec_safe,
    )
    monkeypatch.setattr(live_verifier, "SubprocessJsonRpcClient", FakeClient)
    monkeypatch.setattr(live_verifier, "_wait_for_ready", fake_wait_for_ready)
    monkeypatch.setattr(live_verifier, "_wait_for_search_hit", fake_wait_for_search_hit)
    monkeypatch.setattr(live_verifier, "_parse_tool_json", fake_parse_tool_json)
    monkeypatch.setattr(live_verifier, "_write_project", fake_write_project)
    monkeypatch.setattr(
        live_verifier,
        "_terminate_processes_using_root",
        lambda root: None,
    )
    monkeypatch.setattr(
        live_verifier,
        "_remove_tree_with_retries",
        lambda root: None,
    )

    original_write_text = Path.write_text

    def recording_write_text(
        self: Path,
        data: str,
        *args: object,
        **kwargs: object,
    ) -> int:
        nonlocal wrote_live_file
        if self.name == "live.py" and "fallback_live_symbol" in data:
            wrote_live_file = True
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", recording_write_text)

    await live_verifier._verify_wheel(wheel_path, wheel_path)

    assert wrote_live_file
    assert search_queries == ["fallback_live_symbol"]


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


def test_remove_tree_with_retries_terminates_windows_processes_with_open_file_handles(
    tmp_path: Path, monkeypatch
) -> None:
    locked_root = tmp_path / "locked-root"
    locked_root.mkdir()
    terminated: list[int] = []
    original_rmtree = live_verifier.shutil.rmtree
    attempts = {"count": 0}

    class FakeOpenFile:
        def __init__(self, path: str) -> None:
            self.path = path

    class FakeProcess:
        def __init__(
            self,
            pid: int,
            cwd: str | None,
            cmdline: list[str],
            open_files: list[FakeOpenFile],
        ) -> None:
            self.info = {"pid": pid, "cwd": cwd, "cmdline": cmdline}
            self._open_files = open_files

        def open_files(self) -> list[FakeOpenFile]:
            return list(self._open_files)

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
                FakeProcess(
                    101,
                    str(tmp_path / "elsewhere"),
                    ["python", "--flag"],
                    [
                        FakeOpenFile(
                            str(locked_root / "project" / ".chunkhound" / "daemon.log")
                        )
                    ],
                ),
                FakeProcess(202, str(tmp_path / "other"), [], []),
            ]
        ),
    )
    monkeypatch.setattr(
        live_verifier, "_terminate_process_tree", lambda pid: terminated.append(pid)
    )

    live_verifier._remove_tree_with_retries(locked_root, attempts=2)

    assert attempts["count"] == 2
    assert terminated == [101]
