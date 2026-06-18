"""Compatibility module alias for the packaged realtime implementation."""

# Transitional shim: legacy tests and callers patch runtime lookup symbols on
# this module path. Bind the legacy name to the live service module, then graft
# the package re-exports onto it so both import styles keep working.

from importlib import import_module
import sys

_PACKAGE_MODULE = import_module("chunkhound.services.realtime")
_SERVICE_MODULE = import_module("chunkhound.services.realtime.service")

for _name in getattr(_PACKAGE_MODULE, "__all__", ()):
    setattr(_SERVICE_MODULE, _name, getattr(_PACKAGE_MODULE, _name))

sys.modules[__name__] = _SERVICE_MODULE
