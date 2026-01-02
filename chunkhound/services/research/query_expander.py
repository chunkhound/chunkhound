"""Backwards compatibility shim for QueryExpander.

This module re-exports QueryExpander from shared to maintain backwards compatibility
with code that imports from chunkhound.services.research.query_expander.

New code should import directly from chunkhound.services.research.shared.query_expander.
"""

from chunkhound.services.research.shared.query_expander import QueryExpander

__all__ = ["QueryExpander"]
