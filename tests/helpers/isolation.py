"""Declared pytest inputs and private writable user state; no artifact downloads."""

import os
import shutil
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any

import execnet
import pytest
from loguru import logger

# Runner switches are inputs, not application settings. Consumers use the snapshot.
CONTROL_NAMES = frozenset(
    {
        "CHUNKHOUND_RUN_HEAVY_TESTS",
        "CHUNKHOUND_RUN_WATCHMAN_RUNTIME_VALIDATION",
        "CHUNKHOUND_ENABLE_PERF_TESTS",
        "CH_SKIP_HCL_TESTS",
        "CHUNKHOUND_YAML_ENGINE",
        "CHUNKHOUND_REQUIRE_HNSW",
        "CHUNKHOUND_SKIP_HNSW_PROBE",
        "CHUNKHOUND_USE_EXISTING_SITE_DIST",
        "CHUNKHOUND_TEST_WATCHMAN_MOUNT_PARENT",
        "CHUNKHOUND_TEST_WATCHMAN_MOUNT_CHILD",
        "CHUNKHOUND_ALLOW_PROCESSPOOL",
        "CHUNKHOUND_CODEX_BIN",
        "CHUNKHOUND_DOCS_VERSION",
    }
)
CONTROLS_KEY = pytest.StashKey[Mapping[str, str]]()
ARTIFACTS_KEY = pytest.StashKey[Path]()
SESSION_KEY = pytest.StashKey[tuple[Path, pytest.MonkeyPatch]]()
_RUN_CONFIG: pytest.Config | None = None


def bind_run_config(config: pytest.Config) -> None:
    global _RUN_CONFIG
    _RUN_CONFIG = config


def run_control(name: str, default: str | None = None) -> str | None:
    """Collection/helpers read only the declared snapshot, never ambient env."""
    if _RUN_CONFIG is None:
        raise RuntimeError("Test isolation has not been configured")
    return _RUN_CONFIG.stash[CONTROLS_KEY].get(name, default)


# Provider environment variable prefixes for test isolation.
# Add new prefixes here when adding providers to prevent credential leakage.
_PROVIDER_PREFIXES = (
    "CHUNKHOUND_",
    "CH_",
    "OPENAI_",
    "ANTHROPIC_",
    "VOYAGE_",
    "AZURE_OPENAI_",
    "COHERE_",
    "GEMINI_",
    "GOOGLE_",
    "OPENROUTER_",
    "OLLAMA_",
    "CODEX_",
    "CLAUDE_",
    "OPENCODE_",
    "WATCHMAN_",
    "LITELLM_",
    "AWS_",
    "BEDROCK_",
    "HF_",
    "HUGGING_FACE_",
    "HUGGINGFACE_",
    "LANGCHAIN_",
    "LANGSMITH_",
    "GROQ_",
    "MISTRAL_",
    "DEEPSEEK_",
    "TOGETHER_",
    "FIREWORKS_",
    "JINA_",
    "PERPLEXITY_",
    "GIT_CONFIG_",
    "NETRC",
)

# Ambient proxy settings route subprocess traffic around the Python socket
# guard. They are not declared test inputs, so clear them with the provider
# credentials. Case-insensitive matching is handled by the caller.
_AMBIENT_NETWORK_NAMES = frozenset(
    {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "FTP_PROXY"}
)


def capture_controls(environment: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(
        {
            key.upper(): value
            for key, value in environment.items()
            if key.upper() in CONTROL_NAMES
        }
    )


def bootstrap_isolation(config: pytest.Config) -> None:
    """Run before collection; worker inputs come from the controller snapshot."""
    bind_run_config(config)
    worker_inputs = getattr(config, "workerinput", {})
    config.stash[CONTROLS_KEY] = capture_controls(
        worker_inputs.get("isolation_controls", os.environ)
    )
    scratch = tempfile.TemporaryDirectory(prefix="chunkhound-pytest-")
    patch = pytest.MonkeyPatch()
    config.add_cleanup(scratch.cleanup)
    config.add_cleanup(patch.undo)
    root = Path(scratch.name).resolve()
    artifacts = _session_artifacts(root, worker_inputs)
    config.stash[ARTIFACTS_KEY] = artifacts
    session_root = root / "session"
    config.stash[SESSION_KEY] = session_root, patch
    # Configure-time plugins must resolve their relative output paths first.
    activate_environment(session_root, artifacts, patch)


def isolate_session_cwd(config: pytest.Config) -> None:
    """Enter private cwd after plugin configuration, before collection imports."""
    root, patch = config.stash[SESSION_KEY]
    _enter_project(root, patch)


def prepare_local_worker(gateway: execnet.Gateway) -> None:
    """Keep CLI/plugin path parsing relative to the original local invocation."""
    if gateway.spec.popen and not gateway.spec.chdir:
        if _RUN_CONFIG is None:
            raise RuntimeError("Test isolation has not been configured")
        channel = gateway.remote_exec("import os; os.chdir(channel.receive())")
        channel.send(str(_RUN_CONFIG.invocation_params.dir))
        channel.waitclose()


def _session_artifacts(root: Path, worker_inputs: Mapping[str, Any]) -> Path:
    if "isolation_artifacts" in worker_inputs:
        return Path(worker_inputs["isolation_artifacts"])
    artifacts = root / "artifacts"
    stage_artifacts(artifacts)
    return artifacts


def clear_application_environment(patch: pytest.MonkeyPatch) -> None:
    """Keep OS/process discovery intact while removing undeclared application input."""
    for key in tuple(os.environ):
        upper = key.upper()
        if upper.startswith(_PROVIDER_PREFIXES) or upper in _AMBIENT_NETWORK_NAMES:
            patch.delenv(key)


def _copy_tree(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)


def stage_artifacts(destination: Path) -> None:
    """Copy only preinstalled runtime artifacts, never user credentials/config.

    The VSS tree is frozen because every test HOME links to this one copy.
    A writable shared copy would let one test corrupt the extension for all
    later tests.
    """
    from chunkhound.watchman_runtime.loader import _default_runtime_cache_dir

    extensions = Path.home() / ".duckdb" / "extensions"
    for source in extensions.glob("*/*/vss.duckdb_extension*"):
        target = destination / "vss" / source.relative_to(extensions)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _copy_tree(_default_runtime_cache_dir(), destination / "watchman")
    _freeze_runtime(destination / "vss")


def _freeze_runtime(root: Path) -> None:
    """Make shared native code read-only for every sandbox that links to it.

    macOS also needs this to keep one dlopen identity: writable copies
    exhaust dyld TLS slots and abort after ~500 loads. Linux and Windows use
    it so a test cannot mutate the shared VSS image.
    """
    if root.is_dir():
        for path in root.rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        root.chmod(0o555)


def sandbox_paths(root: Path) -> dict[str, Path]:
    home = root / "home"
    return {
        "HOME": home,
        "USERPROFILE": home,
        "XDG_CONFIG_HOME": root / "config",
        "XDG_CACHE_HOME": root / "cache",
        "XDG_DATA_HOME": root / "data",
        "XDG_STATE_HOME": root / "state",
        "XDG_RUNTIME_DIR": root / "runtime",
        "APPDATA": root / "appdata",
        "LOCALAPPDATA": root / "localappdata",
    } | _tool_paths(root, home)


def _tool_paths(root: Path, home: Path) -> dict[str, Path]:
    return {
        "CODEX_HOME": home / ".codex",
        "CLAUDE_CONFIG_DIR": home / ".claude",
        "NPM_CONFIG_CACHE": root / "npm-cache",
        "CHUNKHOUND_WATCHMAN_RUNTIME_CACHE_DIR": root / "watchman",
    }


def _file_paths(root: Path) -> dict[str, Path]:
    return {
        "NPM_CONFIG_USERCONFIG": root / "npmrc",
        "GIT_CONFIG_GLOBAL": root / "gitconfig",
        "NETRC": root / "netrc",
        "CHUNKHOUND_LLM_CAPABILITY_CACHE": root / "llm-capabilities.json",
    }


def _activate_paths(root: Path, patch: pytest.MonkeyPatch) -> dict[str, Path]:
    files = _file_paths(root)
    paths = sandbox_paths(root) | files
    for key, path in paths.items():
        if key not in files:
            path.mkdir(parents=True, exist_ok=True)
        patch.setenv(key, str(path))
    paths["XDG_RUNTIME_DIR"].chmod(0o700)
    return paths


def _install_artifacts(root: Path, artifacts: Path) -> None:
    """Link shared native code; copy writable runtime state per sandbox."""
    extension_dir = root / "home" / ".duckdb" / "extensions"
    source = artifacts / "vss"
    if source.is_dir():
        extension_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            # One dlopen identity across private HOME directories. The shared
            # tree is read-only; tests that need a writable copy must ask for
            # one explicitly.
            extension_dir.symlink_to(source, target_is_directory=True)
        except OSError:
            # Windows can deny symlink creation without developer mode.
            _copy_tree(source, extension_dir)
    _copy_tree(artifacts / "watchman", root / "watchman")


@contextmanager
def sandbox_context(root: Path, artifacts: Path) -> Iterator[Path]:
    """Keep writable artifacts out of pytest's retained test-output directories."""
    with tempfile.TemporaryDirectory(prefix=".sandbox-", dir=root) as scratch:
        # Restore cwd/env after dependent teardown, before deleting private state.
        with pytest.MonkeyPatch.context() as patch:
            try:
                yield activate_sandbox(Path(scratch), artifacts, patch)
            finally:
                _close_parse_workers()


def _close_parse_workers() -> None:
    """Workers inherit private cwd/env; stop them before deleting that state.

    Uses the module's public shutdown so the pool's creation lock and atexit
    registration stay owned by ``pipeline_bridge``. A renamed or removed
    module leaves the old pool running; the debug line makes that visible.
    """
    bridge = sys.modules.get("chunkhound.pipeline_bridge")
    if bridge is None:
        logger.debug("Parse worker cleanup: pipeline_bridge module not loaded")
        return
    bridge.shutdown_parse_pool()


def activate_sandbox(root: Path, artifacts: Path, patch: pytest.MonkeyPatch) -> Path:
    """Private paths remain active until dependent resources have torn down."""
    home = activate_environment(root, artifacts, patch)
    _enter_project(root, patch)
    return home


def activate_environment(
    root: Path, artifacts: Path, patch: pytest.MonkeyPatch
) -> Path:
    clear_application_environment(patch)
    paths = _activate_paths(root, patch)
    _install_artifacts(root, artifacts)
    return paths["HOME"]


def _enter_project(root: Path, patch: pytest.MonkeyPatch) -> None:
    project = root / ".hermetic-project"
    project.mkdir(parents=True, exist_ok=True)
    # Discovery needs a deliberate project marker, not the checkout's config.
    (project / ".git").mkdir(exist_ok=True)
    patch.chdir(project)
