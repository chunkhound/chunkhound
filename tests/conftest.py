import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow stress tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "heavy: mark tests with large synthetic trees (skipped by default)"
    )
    config.addinivalue_line(
        "markers", "slow: mark slow stress tests (use --run-slow)"
    )


def pytest_collection_modifyitems(config, items):
    # Heavy tests - env var only
    run_heavy = os.getenv("CHUNKHOUND_RUN_HEAVY_TESTS") == "1"
    if not run_heavy:
        skip_heavy = pytest.mark.skip(
            reason="heavy tests skipped (set CHUNKHOUND_RUN_HEAVY_TESTS=1)"
        )
        for item in items:
            if "heavy" in item.keywords:
                item.add_marker(skip_heavy)

    # Slow tests - CLI flag or env var
    run_slow = (
        config.getoption("--run-slow")
        or os.getenv("CHUNKHOUND_RUN_SLOW_TESTS") == "1"
    )
    if not run_slow:
        skip_slow = pytest.mark.skip(
            reason="slow tests skipped (use --run-slow or CHUNKHOUND_RUN_SLOW_TESTS=1)"
        )
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


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


def _cleanup_registry():
    """Clean up registry and database connections between tests.

    Based on pattern from test_config_integration.py. This prevents
    registry state pollution where :memory: database providers leak
    into subsequent tests that expect file-based databases.
    """
    try:
        from chunkhound.registry import get_registry
        registry = get_registry()

        # Try to close database provider if registered
        try:
            db_provider = registry.get_provider("database")
            if hasattr(db_provider, 'close'):
                db_provider.close()
            elif hasattr(db_provider, 'disconnect'):
                db_provider.disconnect()
            elif hasattr(db_provider, '_executor') and hasattr(db_provider._executor, '_connection'):
                if hasattr(db_provider._executor._connection, 'close'):
                    db_provider._executor._connection.close()
        except (ValueError, AttributeError):
            # No database provider or connection to clean up
            pass

        # Clear registry state
        registry._providers.clear()
        registry._language_parsers.clear()
        registry._config = None

    except Exception:
        # Best effort cleanup - don't fail tests if cleanup fails
        pass


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset global registry state between tests.

    This autouse fixture ensures test isolation by clearing the global
    registry singleton before and after each test. This prevents state
    pollution where providers registered by one test affect subsequent tests.

    Critical for tests that create their own DuckDBProvider instances,
    as SerialDatabaseProvider.connect() automatically registers with
    the global registry via _initialize_shared_instances().
    """
    # Clean before test to ensure clean slate
    _cleanup_registry()

    yield

    # Clean after test to prevent pollution
    _cleanup_registry()
