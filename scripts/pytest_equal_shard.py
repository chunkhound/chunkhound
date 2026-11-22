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
    count = len(items)
    quotient, remainder = divmod(count, splits)
    group = shard - 1
    start = group * quotient + min(group, remainder)
    stop = (group + 1) * quotient + min(group + 1, remainder)
    selected = items[start:stop]
    deselected = items[:start] + items[stop:]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected
