from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("ci equal-count sharding")
    group.addoption("--ci-splits", type=int, default=None)
    group.addoption("--ci-group", type=int, default=None)


def pytest_configure(config: pytest.Config) -> None:
    splits = config.getoption("ci_splits")
    shard = config.getoption("ci_group")
    if splits is None and shard is None:
        return
    if splits is None or shard is None:
        raise pytest.UsageError("--ci-splits and --ci-group must be supplied together")
    if splits < 1:
        raise pytest.UsageError("--ci-splits must be at least 1")
    if shard < 1 or shard > splits:
        raise pytest.UsageError("--ci-group must be between 1 and --ci-splits")


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    del session
    splits = config.getoption("ci_splits")
    shard = config.getoption("ci_group")
    if splits is None or shard is None:
        return
    # Deal tests out round-robin so each shard gets every Nth collected test.
    # Counts differ by at most one, membership depends only on collection order,
    # and no directory's tests (e.g. the hub suite) concentrate in one shard.
    group = shard - 1
    selected = items[group::splits]
    deselected = [item for index, item in enumerate(items) if index % splits != group]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected
