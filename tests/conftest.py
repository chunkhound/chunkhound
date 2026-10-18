from collections.abc import Iterator
from pathlib import Path
from typing import Any

import execnet
import pytest
from loguru import logger

from chunkhound.watchman_runtime.loader import is_packaged_watchman_runtime_available
from tests.helpers.isolation import (
    ARTIFACTS_KEY,
    CONTROLS_KEY,
    bootstrap_isolation,
    isolate_session_cwd,
    prepare_local_worker,
    sandbox_context,
)
from tests.helpers.live import (
    LIVE_KEY,
    add_live_options,
    apply_live_environment,
    configure_live,
    prepare_live_test,
    require_cli,
    restrict_network,
    select_live,
)

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


def pytest_addoption(parser: pytest.Parser) -> None:
    add_live_options(parser)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    configure_live(config)
    bootstrap_isolation(config)
    config.addinivalue_line(
        "markers",
        "heavy: mark tests that generate large synthetic trees (skipped by default)",
    )
    config.addinivalue_line(
        "markers",
        "requires_native_watchman: mark tests that require either a packaged "
        "native Watchman runtime or the explicit runtime-validation opt-in",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    isolate_session_cwd(session.config)


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_newgateway(gateway: execnet.Gateway) -> None:
    prepare_local_worker(gateway)


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: Any) -> None:
    # Workers inherit sanitized env; send declared inputs through xdist instead.
    node.workerinput["live_inputs"] = node.config.stash[LIVE_KEY]
    node.workerinput["isolation_controls"] = dict(node.config.stash[CONTROLS_KEY])
    node.workerinput["isolation_artifacts"] = str(node.config.stash[ARTIFACTS_KEY])


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Function]
) -> None:
    select_live(config, items)
    controls = config.stash[CONTROLS_KEY]
    run_heavy = controls.get("CHUNKHOUND_RUN_HEAVY_TESTS") == "1"
    native_watchman_ready = is_packaged_watchman_runtime_available() or (
        controls.get(_WATCHMAN_RUNTIME_VALIDATION_ENV) == "1"
    )
    hnsw_items = [item for item in items if "hnsw" in item.keywords]
    hnsw_required = controls.get(_HNSW_REQUIRED_ENV) == "1"
    if hnsw_required and not hnsw_items:
        pytest.fail("CHUNKHOUND_REQUIRE_HNSW=1 but no tests are marked hnsw")
    if hnsw_items:
        if controls.get(_HNSW_SKIP_PROBE_ENV) == "1" and not hnsw_required:
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


@pytest.fixture(scope="session")
def opencode_models() -> list[str]:
    """Use only the deliberately supplied live model list."""
    data = require_cli("opencode")
    if data.get("binary", "opencode") != "opencode":
        pytest.fail("OpenCode must be available as 'opencode' on PATH")
    models = data.get("models")
    if (
        not isinstance(models, list)
        or not models
        or not all(isinstance(m, str) for m in models)
    ):
        pytest.fail("Live prerequisite missing: opencode.models list")
    return models


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Function) -> None:
    prepare_live_test(item)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown(
    item: pytest.Item, nextitem: pytest.Item | None
) -> Iterator[None]:
    try:
        return (yield)
    finally:
        restrict_network()


@pytest.fixture(scope="session")
def test_run_controls(pytestconfig: pytest.Config) -> Any:
    """Declared runner inputs captured before environment sanitization."""
    return pytestconfig.stash[CONTROLS_KEY]


@pytest.fixture(autouse=True)
def live_environment(
    request: pytest.FixtureRequest,
    hermetic_environment: Path,
    monkeypatch: pytest.MonkeyPatch,
    pytestconfig: pytest.Config,
) -> None:
    if (
        request.node.get_closest_marker("live") is not None
        and "environment" in pytestconfig.stash[LIVE_KEY]
    ):
        apply_live_environment(monkeypatch)


@pytest.fixture(autouse=True)
def hermetic_environment(tmp_path: Path, pytestconfig: pytest.Config) -> Iterator[Path]:
    """Explicit dependency for resources that must close before sandbox cleanup."""
    with sandbox_context(tmp_path.resolve(), pytestconfig.stash[ARTIFACTS_KEY]) as home:
        yield home
