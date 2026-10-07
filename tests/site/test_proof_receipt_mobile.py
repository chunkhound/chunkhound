"""Proof-receipt mobile stacking contract.

On phones the receipt's figure and its label can't share one line. The shipped
bug: `.proof-grid > div` stayed a single-baseline row, so a wrapped label
pinned to the figure's *first* line while the focal word ("days", "laptop")
dangled below.

happy-dom has no layout, so the guard is structural: below the phone breakpoint
each metric's pair must stack (figure over label) instead of baseline-aligning
a pair that has wrapped.
"""

from __future__ import annotations

from tests.site.css_helpers import bodies


def _flat(body: str) -> str:
    """Whitespace-insensitive view of a shipped declaration block."""
    return body.replace(" ", "")


def test_receipt_row_shares_one_baseline_by_default() -> None:
    """Wide viewports: figure and label sit on one row, one baseline."""
    default = [_flat(body) for body in bodies(".proof-grid>div", media="")]
    assert any(
        "display:flex" in body and "align-items:baseline" in body
        for body in default
    ), f"the receipt row must be a single-baseline flex row: {default}"


def test_receipt_stacks_figure_over_label_on_phones() -> None:
    """Phones: the pair stacks (figure over label), so a wrapped row can never
    baseline-split the label off the figure's first line."""
    stacked = [_flat(body) for body in bodies(".proof-grid>div", media="600px")]
    assert any("flex-direction:column" in body for body in stacked), (
        f"the receipt must stack its figure/label pair on phones: {stacked}"
    )
    assert any("align-items:center" in body for body in stacked), (
        f"the stacked receipt must center the pair: {stacked}"
    )
