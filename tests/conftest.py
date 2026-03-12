import os

import pytest

from chunkhound.watchman_runtime.loader import is_packaged_watchman_runtime_supported


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "heavy: mark tests that generate large synthetic trees (skipped by default)",
    )
    config.addinivalue_line(
        "markers",
        "requires_native_watchman: mark tests that require a packaged native "
        "Watchman runtime on the host platform",
    )


def pytest_collection_modifyitems(config, items):
    run_heavy = os.getenv("CHUNKHOUND_RUN_HEAVY_TESTS") == "1"
    native_watchman_supported = is_packaged_watchman_runtime_supported()
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
        reason="native Watchman runtime is unsupported on this platform"
    )
    for item in items:
        if skip_heavy is not None and "heavy" in item.keywords:
            item.add_marker(skip_heavy)
        if (
            not native_watchman_supported
            and "requires_native_watchman" in item.keywords
        ):
            item.add_marker(skip_native_watchman)


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
