import os

import pytest
from loguru import logger

logger.remove()

from tests.utils.windows_compat import get_configured_watcher_mode


def pytest_collection_modifyitems(config, items):
    run_heavy = os.getenv("CHUNKHOUND_RUN_HEAVY_TESTS") == "1"
    if run_heavy:
        return
    skip_heavy = pytest.mark.skip(
        reason=(
            "heavy tests skipped by default "
            "(set CHUNKHOUND_RUN_HEAVY_TESTS=1 to run)"
        )
    )
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)


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


@pytest.fixture
def watcher_mode(request) -> str:
    """Resolve watcher mode from explicit markers before falling back to env."""
    native_markers = list(request.node.iter_markers(name="native_watcher"))
    polling_markers = list(request.node.iter_markers(name="polling_watcher"))
    if native_markers and polling_markers:
        raise pytest.UsageError(
            f"{request.node.nodeid} cannot request both native "
            "and polling watcher modes"
        )
    if polling_markers:
        return "polling"
    if native_markers:
        return "native"
    return get_configured_watcher_mode()
