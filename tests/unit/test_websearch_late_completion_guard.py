"""Verify the zendriver Transaction late-completion guard.

The guard prevents InvalidStateError from tearing down zendriver's
Listener.listener_loop when a CDP response arrives for a Transaction whose
awaiter was already cancelled by an outer asyncio.wait_for timeout. See
_install_late_completion_guard in chunkhound/utils/websearch_core.py.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_cancelled_transaction_does_not_raise_on_late_response(
    monkeypatch,
) -> None:
    pytest.importorskip("zendriver")
    from zendriver import cdp
    from zendriver.core.connection import Transaction

    from chunkhound.utils import websearch_core

    # The guard is process-global and idempotent by design — fetch_and_save
    # calls _install_late_completion_guard on every invocation and the latch
    # short-circuits repeats. Reset the latch so this test exercises the
    # install path even when a prior test already ran it; the patch
    # intentionally persists past teardown.
    monkeypatch.setattr(websearch_core, "_late_completion_guard_installed", False)

    websearch_core._install_late_completion_guard()

    # Use a real CDP command so Transaction's constructor (which calls
    # next(cdp_obj) to read method+params) succeeds.
    tx = Transaction(cdp.page.navigate(url="about:blank"))
    tx.cancel()
    assert tx.cancelled()

    # Simulate listener_loop's call shape from connection.py:780.
    # Without the guard, set_result on the cancelled Future raises
    # InvalidStateError; the guard short-circuits before that.
    tx(result={
        "frameId": "F" * 32,
        "loaderId": "L" * 32,
    })  # must not raise

    # Happy path: the guard must still delegate to the original __call__
    # for non-done Transactions, otherwise every CDP send would hang.
    tx2 = Transaction(cdp.page.navigate(url="about:blank"))
    assert not tx2.done()
    tx2(result={
        "frameId": "F" * 32,
        "loaderId": "L" * 32,
    })
    assert tx2.done()
    assert not tx2.cancelled()
    assert tx2.exception() is None
