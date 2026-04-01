"""Tests for CompactionError structured fields."""

from chunkhound.core.exceptions.core import CompactionError


class TestCompactionError:
    def test_operation_and_reason(self) -> None:
        err = CompactionError("lock held", operation="lock")
        assert err.operation == "lock"
        assert err.reason == "lock held"
        assert "operation=lock" in str(err)
        assert "lock held" in str(err)

    def test_reason_kwarg(self) -> None:
        err = CompactionError(reason="detail", operation="preflight")
        assert err.reason == "detail"
        assert "detail" in str(err)

    def test_no_operation(self) -> None:
        err = CompactionError("just a message")
        assert err.operation is None
        assert "Compaction error: just a message" in str(err)

    def test_reason_stored(self) -> None:
        err = CompactionError("some reason")
        assert err.reason == "some reason"

    def test_bare_construction(self) -> None:
        err = CompactionError()
        assert err.operation is None
        assert err.reason is None
        assert "Compaction error" in str(err)
