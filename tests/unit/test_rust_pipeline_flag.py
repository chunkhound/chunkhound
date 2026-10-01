"""Tests for the CHUNKHOUND_USE_RUST feature-flag guard."""

from chunkhound.utils.rust_pipeline_flag import _get_use_rust


def test_rust_enabled_by_default(monkeypatch):
    """Without override, _get_use_rust() returns True (opt-out semantics).

    The Rust parse->embed->write pipeline is now the default path.
    Set CHUNKHOUND_USE_RUST=0 to fall back to the Python pipeline.
    """
    monkeypatch.delenv("CHUNKHOUND_USE_RUST", raising=False)

    assert _get_use_rust() is True


def test_rust_disabled_via_env(monkeypatch):
    """CHUNKHOUND_USE_RUST=0 must make _get_use_rust() return False."""
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "0")

    assert _get_use_rust() is False


def test_rust_enabled_via_env(monkeypatch):
    """CHUNKHOUND_USE_RUST=1 must make _get_use_rust() return True."""
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "1")

    assert _get_use_rust() is True
