import os
import subprocess

import pytest
from loguru import logger

from chunkhound.watchman_runtime.loader import is_packaged_watchman_runtime_available

logger.remove()

_WATCHMAN_RUNTIME_VALIDATION_ENV = "CHUNKHOUND_RUN_WATCHMAN_RUNTIME_VALIDATION"
_HNSW_REQUIRED_ENV = "CHUNKHOUND_REQUIRE_HNSW"
_HNSW_SKIP_PROBE_ENV = "CHUNKHOUND_SKIP_HNSW_PROBE"


def _probe_hnsw_capability() -> tuple[bool, str]:
    """Verify that the locally installed DuckDB VSS extension supports HNSW.

    Collection must remain offline. The required CI lane installs VSS before
    pytest starts; ordinary environments skip HNSW tests when ``LOAD`` fails.
    """
    connection = None
    try:
        import duckdb

        connection = duckdb.connect(":memory:")
        connection.execute("SET autoinstall_known_extensions = false")
        connection.execute("LOAD vss")
        connection.execute("CREATE TABLE hnsw_probe (embedding FLOAT[3])")
        connection.execute(
            "CREATE INDEX hnsw_probe_index ON hnsw_probe USING HNSW "
            "(embedding) WITH (metric = 'cosine')"
        )
        row = connection.execute(
            "SELECT index_name FROM duckdb_indexes() "
            "WHERE index_name = 'hnsw_probe_index'"
        ).fetchone()
        return row is not None, "HNSW index probe returned no catalog entry"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if connection is not None:
            connection.close()


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
    hnsw_items = [item for item in items if "hnsw" in item.keywords]
    hnsw_required = os.getenv(_HNSW_REQUIRED_ENV) == "1"
    if hnsw_required and not hnsw_items:
        pytest.fail("CHUNKHOUND_REQUIRE_HNSW=1 but no tests are marked hnsw")
    if hnsw_items:
        if os.getenv(_HNSW_SKIP_PROBE_ENV) == "1" and not hnsw_required:
            hnsw_ready, hnsw_reason = False, f"{_HNSW_SKIP_PROBE_ENV}=1"
        else:
            hnsw_ready, hnsw_reason = _probe_hnsw_capability()
        if not hnsw_ready:
            if hnsw_required:
                pytest.fail(
                    "CHUNKHOUND_REQUIRE_HNSW=1 but DuckDB HNSW is unavailable: "
                    f"{hnsw_reason}",
                    pytrace=False,
                )
            skip_hnsw = pytest.mark.skip(
                reason=f"DuckDB HNSW is unavailable: {hnsw_reason}"
            )
            for item in hnsw_items:
                item.add_marker(skip_hnsw)
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


@pytest.fixture(scope="session")
def free_opencode_models() -> list[str]:
    """Discover all free OpenCode model slugs for integration tests."""
    models = _discover_free_opencode_models()
    if not models:
        pytest.skip("No free OpenCode models available")
    return models


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
