"""Contract test: ``__all__`` in ``websearch_core`` stays in sync with the
module's real public API surface.

If a name is added to or removed from ``__all__``, this test forces the
author to acknowledge the change.  The cost is near-zero; the value is
preventing silent documentation drift.
"""

from __future__ import annotations

import chunkhound.utils.websearch_core as ws


def test_all_items_exist_on_module():
    """Every name in ``__all__`` must be resolvable on the module."""
    for name in ws.__all__:
        assert hasattr(ws, name), f"{name!r} in __all__ but not defined on module"


def test_all_items_are_public():
    """No private helpers should leak into ``__all__``."""
    for name in ws.__all__:
        assert not name.startswith("_"), f"Private name {name!r} should not be in __all__"
