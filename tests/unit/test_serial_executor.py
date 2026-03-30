"""Tests for the _connection_suspended guard in serial_executor."""

import threading

import pytest

from chunkhound.core.exceptions import CompactionError
from chunkhound.providers.database.serial_executor import (
    _executor_local,
    get_thread_local_connection,
)


@pytest.fixture(autouse=True)
def _clean_thread_local():
    """Ensure no stale connection leaks between tests."""
    if hasattr(_executor_local, "connection"):
        delattr(_executor_local, "connection")
    yield
    if hasattr(_executor_local, "connection"):
        delattr(_executor_local, "connection")


class _StubProvider:
    """Minimal provider with _connection_suspended and _create_connection."""

    def __init__(self, *, suspended: bool) -> None:
        self._connection_suspended = threading.Event()
        if suspended:
            self._connection_suspended.set()

    def _create_connection(self):
        return object()  # sentinel


def test_connection_refused_when_suspended() -> None:
    provider = _StubProvider(suspended=True)
    with pytest.raises(CompactionError, match="compaction in progress"):
        get_thread_local_connection(provider)


def test_connection_allowed_when_not_suspended() -> None:
    provider = _StubProvider(suspended=False)
    conn = get_thread_local_connection(provider)
    assert conn is _executor_local.connection


def test_cached_connection_returned_on_second_call() -> None:
    provider = _StubProvider(suspended=False)
    first = get_thread_local_connection(provider)
    second = get_thread_local_connection(provider)
    assert first is second
