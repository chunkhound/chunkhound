"""Compatibility module alias for the packaged realtime implementation.

Uses the ``__class__`` hack (``ModuleType`` subclass) to delegate attribute
reads *and* writes to the live realtime service module.  This avoids the
fragile ``sys.modules`` mid-import swap that caused deadlock risk on
Python 3.14 Windows, while still preserving the ability to monkeypatch
runtime symbols through this import path
(e.g. ``import ... as realtime_service_module; monkeypatch.setattr(...)``).

Why not bare ``__getattr__`` / ``__setattr__`` at module scope?
Module-level ``__setattr__`` (PEP 726) was rejected and is dead code on
every released Python version.  The ``__class__`` hack is the only
production-ready mechanism for intercepting module attribute writes.
"""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from typing import Any

_REAL_MODULE_NAME = "chunkhound.services.realtime.service"
_real_module: Any = None  # cached on first access


def _get_real() -> Any:
    """Return the real service module, importing it lazily on first call."""
    global _real_module
    if _real_module is None:
        _real_module = import_module(_REAL_MODULE_NAME)
    return _real_module


class _ShimModule(ModuleType):
    """Module subclass that lazily delegates all attribute access to
    ``chunkhound.services.realtime.service``.

    Reads go through ``__getattr__``; writes go through ``__setattr__``,
    which propagates to the real module so monkeypatches are visible to
    code importing directly from the real path.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(_get_real(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Underscore-prefixed names (including dunders like __name__, __file__)
        # stay on the shim — prevents forwarding cached state like ``_real_module``
        # to the real module, and keeps metadata pointing to the shim file.
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(_get_real(), name, value)

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            super().__delattr__(name)
        else:
            delattr(_get_real(), name)

    def __dir__(self) -> list[str]:
        return dir(_get_real())


# Apply the hack: change this module's type so __setattr__ is honoured.
sys.modules[__name__].__class__ = _ShimModule
