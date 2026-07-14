"""Unit-test conftest: disable auto-compaction dispatch.

The 1/N random-sampling in serial_executor.py would introduce
non-deterministic compaction during arbitrary test operations.
We disable the sampling unconditionally here so all tests in
this directory are deterministic.
"""

import pytest


@pytest.fixture(autouse=True)
def _disable_auto_compaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent auto-compaction dispatch from firing during tests."""
    monkeypatch.setattr(
        "chunkhound.providers.database.serial_executor.COMPACT_SAMPLE_INTERVAL", 0
    )
