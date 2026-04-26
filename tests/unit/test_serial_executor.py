"""Tests for the _connection_allowed guard in serial_executor."""

import threading
import time

import pytest

from chunkhound.core.exceptions import CompactionError
from chunkhound.providers.database.serial_executor import (
    _executor_local,
    get_thread_local_connection,
    get_thread_local_state,
    reset_thread_local_state,
)


@pytest.fixture(autouse=True)
def _clean_thread_local():
    """Ensure no stale connection/state leaks between tests."""
    if hasattr(_executor_local, "connection"):
        delattr(_executor_local, "connection")
    if hasattr(_executor_local, "state"):
        delattr(_executor_local, "state")
    yield
    if hasattr(_executor_local, "connection"):
        delattr(_executor_local, "connection")
    if hasattr(_executor_local, "state"):
        delattr(_executor_local, "state")


class _StubProvider:
    """Minimal provider with _connection_allowed and _create_connection."""

    def __init__(self, *, suspended: bool) -> None:
        self._connection_allowed = threading.Event()
        if not suspended:
            self._connection_allowed.set()

    @property
    def is_accepting_connections(self) -> bool:
        return self._connection_allowed.is_set()

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


def test_reset_thread_local_state_clears_stale_keys() -> None:
    """reset_thread_local_state must restore defaults and remove extras."""
    state = get_thread_local_state()
    state["last_checkpoint_time"] = time.time() - 600
    state["deferred_checkpoint"] = True
    state["transaction_active"] = True

    reset_thread_local_state()

    fresh = get_thread_local_state()
    assert fresh["transaction_active"] is False
    assert "last_activity_time" in fresh
    assert "last_checkpoint_time" not in fresh
    assert "deferred_checkpoint" not in fresh


def test_disconnect_resets_stale_state() -> None:
    """After _executor_disconnect, thread-local state must be fresh defaults."""
    from chunkhound.providers.database.serial_database_provider import (
        SerialDatabaseProvider,
    )

    # Seed stale state simulating a pre-compaction session
    state = get_thread_local_state()
    state["last_checkpoint_time"] = time.time() - 600  # 10 min ago
    state["deferred_checkpoint"] = True
    state["transaction_active"] = True

    # Call the base _executor_disconnect with no real connection
    SerialDatabaseProvider._executor_disconnect(None, conn=None, state=state, skip_checkpoint=False)

    fresh = get_thread_local_state()
    assert fresh["transaction_active"] is False
    assert "last_activity_time" in fresh
    assert "last_checkpoint_time" not in fresh
    assert "deferred_checkpoint" not in fresh
