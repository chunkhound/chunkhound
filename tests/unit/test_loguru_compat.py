"""Loguru activation contract: mute and restore must survive upgrades."""

from types import SimpleNamespace

import pytest
from loguru import logger

from chunkhound.mcp_server import stdio


def test_round_trip_and_drift_guard():
    logger.disable("compat.quiet")
    saved = stdio._snapshot_loguru_activation(logger)
    with stdio._silence_loguru():
        muted = stdio._snapshot_loguru_activation(logger)
    assert muted == ([("", False)], False)
    assert stdio._snapshot_loguru_activation(logger) == saved
    unknown = SimpleNamespace(_core=SimpleNamespace(activation_list=[{}]))
    with pytest.raises(RuntimeError, match="loguru"):
        stdio._snapshot_loguru_activation(unknown)
