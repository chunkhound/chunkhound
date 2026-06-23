"""Contract tests for the realtime_indexing_service compatibility shim.

These tests verify the external invariants that downstream code (and
monkeypatch patterns) depend on, NOT implementation details.
"""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType

import pytest

import chunkhound.services.realtime_indexing_service  # noqa: F401

SHIM_PATH = "chunkhound.services.realtime_indexing_service"
REAL_PATH = "chunkhound.services.realtime.service"


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class TestShimDelegationIdentity:
    """from shim import X must be the same object as from real import X."""

    @pytest.mark.parametrize(
        "name",
        [
            "RealtimeIndexingService",
            "SimpleEventHandler",
            "WatchmanRealtimeAdapter",
            "normalize_file_path",
            "RealtimePathFilter",
            "build_watchman_scope_plan",
            "RealtimePathFilterSettings",
            "PollingRealtimeAdapter",
            "WatchdogRealtimeAdapter",
        ],
    )
    def test_import_from_shim_is_same_object_as_real(self, name: str) -> None:
        shim = __import__(SHIM_PATH, fromlist=[name])
        real = __import__(REAL_PATH, fromlist=[name])
        assert getattr(shim, name) is getattr(real, name)


# ---------------------------------------------------------------------------
# Write-through (monkeypatch contract)
# ---------------------------------------------------------------------------


class TestShimWriteThrough:
    """setattr(shim, "X", val) must propagate to the real module."""

    def test_setattr_propagates_to_real_module(self) -> None:
        shim = sys.modules[SHIM_PATH]
        real = import_module(REAL_PATH)
        sentinel = object()
        attr_name = "test_shim_write_through_sentinel"
        try:
            setattr(shim, attr_name, sentinel)
            assert getattr(real, attr_name) is sentinel
        finally:
            if hasattr(real, attr_name):
                delattr(real, attr_name)

    def test_setattr_on_real_is_visible_through_shim(self) -> None:
        shim = sys.modules[SHIM_PATH]
        real = import_module(REAL_PATH)
        sentinel = object()
        attr_name = "test_shim_read_through_sentinel"
        try:
            setattr(real, attr_name, sentinel)
            assert getattr(shim, attr_name) is sentinel
        finally:
            if hasattr(real, attr_name):
                delattr(real, attr_name)


# ---------------------------------------------------------------------------
# Underscore-prefix guard
# ---------------------------------------------------------------------------


class TestShimUnderscoreGuard:
    """setattr(shim, "_X", val) must NOT propagate to the real module."""

    def test_underscore_setattr_stays_on_shim(self) -> None:
        shim = sys.modules[SHIM_PATH]
        real = import_module(REAL_PATH)
        attr_name = "_test_underscore_guard_sentinel"
        try:
            setattr(shim, attr_name, "on_shim")
            assert getattr(shim, attr_name) == "on_shim"
            assert not hasattr(real, attr_name)
        finally:
            if hasattr(shim, attr_name):
                delattr(shim, attr_name)


# ---------------------------------------------------------------------------
# Delete-through (delattr contract)
# ---------------------------------------------------------------------------


class TestShimDeleteThrough:
    """delattr(shim, "X") must propagate to the real module for public names."""

    def test_delattr_on_public_name_removes_from_real(self) -> None:
        shim = sys.modules[SHIM_PATH]
        real = import_module(REAL_PATH)
        sentinel = object()
        attr_name = "test_shim_delete_through_sentinel"
        try:
            setattr(real, attr_name, sentinel)
            assert hasattr(real, attr_name)
            delattr(shim, attr_name)
            assert not hasattr(real, attr_name)
        finally:
            if hasattr(real, attr_name):
                delattr(real, attr_name)

    def test_delattr_on_underscore_name_removes_from_shim(self) -> None:
        shim = sys.modules[SHIM_PATH]
        attr_name = "_test_shim_delete_underscore_sentinel"
        try:
            setattr(shim, attr_name, "on_shim")
            assert hasattr(shim, attr_name)
            delattr(shim, attr_name)
            assert not hasattr(shim, attr_name)
        finally:
            pass


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestShimIdempotency:
    """Repeated imports from the shim must return the same objects."""

    def test_repeated_import_returns_same_objects(self) -> None:
        shim_a = __import__(SHIM_PATH, fromlist=["RealtimeIndexingService"])
        shim_b = __import__(SHIM_PATH, fromlist=["RealtimeIndexingService"])
        assert shim_a.RealtimeIndexingService is shim_b.RealtimeIndexingService


# ---------------------------------------------------------------------------
# AttributeError for missing names
# ---------------------------------------------------------------------------


class TestShimMissingAttribute:
    """Accessing a nonexistent attribute must raise AttributeError."""

    def test_getattr_raises_attribute_error_for_missing(self) -> None:
        shim = sys.modules[SHIM_PATH]
        with pytest.raises(AttributeError):
            _ = shim._nonexistent_symbol_xyz_42

    def test_import_from_shim_raises_for_missing(self) -> None:
        with pytest.raises(ImportError):
            from chunkhound.services.realtime_indexing_service import (  # noqa: F401
                _nonexistent_symbol_xyz_42,
            )


# ---------------------------------------------------------------------------
# dir()
# ---------------------------------------------------------------------------


class TestShimDir:
    """dir(shim) must include the real module's public symbols."""

    def test_dir_includes_real_module_symbols(self) -> None:
        shim = sys.modules[SHIM_PATH]
        real = import_module(REAL_PATH)
        real_names = set(dir(real))
        shim_names = set(dir(shim))
        # Every name from the real module should appear in dir(shim)
        assert real_names <= shim_names


# ---------------------------------------------------------------------------
# Module type
# ---------------------------------------------------------------------------


class TestShimModuleType:
    """The shim must be a ModuleType subclass (not a plain ModuleType)."""

    def test_module_class_is_custom_subclass(self) -> None:
        shim = sys.modules[SHIM_PATH]
        assert type(shim) is not ModuleType
        assert issubclass(type(shim), ModuleType)

    def test_module_name_is_correct(self) -> None:
        shim = sys.modules[SHIM_PATH]
        assert shim.__name__ == SHIM_PATH

    def test_module_file_points_to_shim(self) -> None:
        """__file__ must reference the shim module, not the delegated real module."""
        shim = sys.modules[SHIM_PATH]
        assert shim.__file__ is not None
        assert "realtime_indexing_service" in shim.__file__
        assert "realtime/service" not in shim.__file__
