"""Unit-test conftest: disable auto-compaction dispatch.

The 1/N random-sampling in serial_executor.py would introduce
non-deterministic compaction during arbitrary test operations.
We disable the sampling unconditionally here so all tests in
this directory are deterministic.
"""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_llm_capability_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep unit tests isolated from the developer's capability cache."""
    monkeypatch.setenv(
        "CHUNKHOUND_LLM_CAPABILITY_CACHE", str(tmp_path / "llm-capabilities.json")
    )


@pytest.fixture(autouse=True)
def _disable_auto_compaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent auto-compaction dispatch from firing during tests."""
    monkeypatch.setattr(
        "chunkhound.providers.database.serial_executor.COMPACT_SAMPLE_INTERVAL", 0
    )
