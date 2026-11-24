"""Skip a missing local tool, but fail where CI requires it."""

from __future__ import annotations

import os
from typing import NoReturn

import pytest


def skip_or_fail(reason: str, *, require_env: str) -> NoReturn:
    """Skip locally; fail where the named env flag requires the tool."""
    if os.environ.get(require_env) == "1":
        pytest.fail(reason)
    pytest.skip(reason)
