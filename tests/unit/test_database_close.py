"""Tests for the high-level Database shutdown contract."""

import pytest

from chunkhound.core.exceptions import DatabaseError
from chunkhound.database import Database


class FailingDisconnectProvider:
    """Minimal provider that exposes a disconnect failure to Database."""

    def __init__(self, error: DatabaseError) -> None:
        self.is_connected = True
        self.error = error
        self.disconnect_calls = 0

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        raise self.error


def test_database_close_propagates_provider_disconnect_error() -> None:
    """Database.close must not hide a failed checkpoint or disconnection."""
    error = DatabaseError(
        operation="disconnect", reason="checkpoint failed before disconnect"
    )
    provider = FailingDisconnectProvider(error)
    database = Database(
        ":memory:",
        indexing_coordinator=object(),
        search_service=object(),
        embedding_service=object(),
        provider=provider,
    )

    with pytest.raises(DatabaseError) as raised:
        database.close()

    assert raised.value is error
    assert provider.disconnect_calls == 1
