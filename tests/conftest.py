import os
import subprocess
import tempfile

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
