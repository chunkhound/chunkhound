"""Compatibility module alias for the packaged realtime implementation."""

# Transitional shim: keep the legacy import path bound to the live realtime
# service module so existing module-level monkeypatches still affect runtime
# symbols.

from importlib import import_module
import sys

_MODULE = import_module("chunkhound.services.realtime.service")
sys.modules[__name__] = _MODULE
