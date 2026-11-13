import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator

import pytest
from loguru import logger

from chunkhound.watchman_runtime.loader import is_packaged_watchman_runtime_available

logger.remove()

_WATCHMAN_RUNTIME_VALIDATION_ENV = "CHUNKHOUND_RUN_WATCHMAN_RUNTIME_VALIDATION"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "heavy: mark tests that generate large synthetic trees (skipped by default)",
    )
    config.addinivalue_line(
        "markers",
        "requires_native_watchman: mark tests that require either a packaged "
        "native Watchman runtime or the explicit runtime-validation opt-in",
    )


def pytest_collection_modifyitems(config, items):
    run_heavy = os.getenv("CHUNKHOUND_RUN_HEAVY_TESTS") == "1"
    native_watchman_ready = is_packaged_watchman_runtime_available() or (
        os.getenv(_WATCHMAN_RUNTIME_VALIDATION_ENV) == "1"
    )
    if run_heavy:
        skip_heavy = None
    else:
        skip_heavy = pytest.mark.skip(
            reason=(
                "heavy tests skipped by default "
                "(set CHUNKHOUND_RUN_HEAVY_TESTS=1 to run)"
            )
        )
    skip_native_watchman = pytest.mark.skip(
        reason=(
            "native Watchman runtime is unavailable for this source install "
            f"(set {_WATCHMAN_RUNTIME_VALIDATION_ENV}=1 in the dedicated "
            "validation lane to exercise hydration)"
        )
    )
    for item in items:
        if skip_heavy is not None and "heavy" in item.keywords:
            item.add_marker(skip_heavy)
        if not native_watchman_ready and "requires_native_watchman" in item.keywords:
            item.add_marker(skip_native_watchman)


@pytest.fixture(scope="session", autouse=True)
def _ignore_user_global_config(tmp_path_factory):
    """Keep the developer's user-wide config out of every test.

    ``Config()`` merges the first file it finds among
    ``Config.get_global_config_candidates()`` (``~/.config/chunkhound/...``),
    so a test that builds a Config without an ``embedding`` section inherits
    whatever provider the machine is configured with. On a machine whose
    global config selects ``voyageai`` with the key in the environment, every
    realtime test that calls ``create_services`` then fails in
    ``EmbeddingProviderFactory`` (or, with the key present, builds a real
    provider). Pinning ``CHUNKHOUND_GLOBAL_CONFIG_FILE`` to an existing empty
    file makes the global layer empty, as it is in CI. Tests that exercise
    global-config discovery set or delete the variable themselves.
    """
    empty = tmp_path_factory.mktemp("global-config") / "chunkhound.json"
    empty.write_text("{}", encoding="utf-8")
    previous = os.environ.get("CHUNKHOUND_GLOBAL_CONFIG_FILE")
    os.environ["CHUNKHOUND_GLOBAL_CONFIG_FILE"] = str(empty)
    yield empty
    if previous is None:
        os.environ.pop("CHUNKHOUND_GLOBAL_CONFIG_FILE", None)
    else:
        os.environ["CHUNKHOUND_GLOBAL_CONFIG_FILE"] = previous


@pytest.fixture(autouse=True)
def _clear_transient_diff_chunk_cache() -> Iterator[None]:
    """Isolate the process-wide diff chunk cache across tests."""
    yield
    tools_mod = sys.modules.get("chunkhound.mcp_server.tools")
    cache = getattr(tools_mod, "_TRANSIENT_DIFF_CHUNK_CACHE", None)
    if cache is not None:
        cache.clear()


def _discover_free_opencode_models() -> list[str]:
    """Discover currently free OpenCode model slugs."""
    try:
        result = subprocess.run(
            ["opencode", "models"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return []

        models = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            if "free" in line.lower():
                # Extract just the model slug (first whitespace-delimited token)
                slug = line.split()[0]
                if "/" not in slug:
                    continue
                models.append(slug)
        return models
    except (subprocess.SubprocessError, FileNotFoundError):
        return []


# Probe timeout: below the integration test's 30s to bound fixture latency.
_OPENCODE_PROBE_TIMEOUT = 20
# Tests use models[0] only; probe up to 4 candidates to find one servable model.
_MAX_OPENCODE_PROBES = 4


def _probe_opencode_model(slug: str) -> tuple[bool, str | None]:
    """Return (servable, reason) by running one minimal JSON-mode request.

    Invokes the CLI the same way OpenCodeCLIProvider does (temp cwd,
    --format json) and reuses its NDJSON classifier to check the success path.
    A catalog entry marked free is not proof the
    upstream serves it, so tests skip rather than fail when no candidate is
    servable.
    """
    from chunkhound.providers.llm.opencode_cli_provider import (
        OpenCodeCLIProvider,
    )

    provider = OpenCodeCLIProvider(model=slug, max_retries=1)
    try:
        result = subprocess.run(
            ["opencode", "run", "--model", slug, "--format", "json"],
            input=b"ping\n",
            capture_output=True,
            timeout=_OPENCODE_PROBE_TIMEOUT,
            cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired:
        return False, f"probe timed out after {_OPENCODE_PROBE_TIMEOUT}s"
    except OSError as exc:
        return False, f"probe could not run the opencode CLI: {exc}"

    stderr_msg = result.stderr.decode("utf-8", errors="replace")
    # intentional: reuse production parser so probe and provider agree on "usable"
    parsed = provider._parse_json_output(
        result.stdout, stderr_msg, result.returncode, model=slug
    )
    if parsed.action == "success":
        return True, None
    if parsed.action == "retry_plain":
        return False, "JSON probe unsupported or returned no text"
    return False, parsed.error_message or "JSON probe failed without a message"


@pytest.fixture(scope="session")
def free_opencode_models() -> list[str]:
    """Discover free OpenCode model slugs that actually serve requests."""
    models = _discover_free_opencode_models()
    if not models:
        pytest.skip("No free OpenCode models available")

    servable: list[str] = []
    rejected: list[tuple[str, str]] = []
    for slug in models:
        if len(servable) >= 1 or len(servable) + len(rejected) >= _MAX_OPENCODE_PROBES:
            break
        ok, reason = _probe_opencode_model(slug)
        if ok:
            servable.append(slug)
        else:
            rejected.append((slug, reason))

    if not servable:
        reasons = "; ".join(f"{slug}: {reason}" for slug, reason in rejected)
        pytest.skip(f"No servable free OpenCode models ({reasons})")
    return servable


@pytest.fixture
def isolated_mcp_mode() -> Iterator[None]:
    """Restore CHUNKHOUND_MCP_MODE after tests that run an MCP entry point.

    mcp_command and async_main set it for the process lifetime. Restore it by
    hand: monkeypatch.delenv records no undo when the variable is unset, so it
    would let those writes leak into later tests.
    """
    saved = os.environ.get("CHUNKHOUND_MCP_MODE")
    yield
    if saved is None:
        os.environ.pop("CHUNKHOUND_MCP_MODE", None)
    else:
        os.environ["CHUNKHOUND_MCP_MODE"] = saved


@pytest.fixture
def clean_environment(monkeypatch):
    """Ensure tests run with a clean environment.

    - Unset CHUNKHOUND_* variables that can alter discovery/backends.
    - Unset common embedding API keys to avoid accidental network init.
    """
    to_clear = [k for k in os.environ.keys() if k.startswith("CHUNKHOUND_")]
    to_clear += [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "VOYAGE_API_KEY",
        "ANTHROPIC_API_KEY",
    ]
    for k in to_clear:
        monkeypatch.delenv(k, raising=False)
    yield
