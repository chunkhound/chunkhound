"""Built-CSS contracts for the configurator's selection → detail relationship.

The detail is no longer a panel beside the list: every provider row carries its
own reveal and opens it with `:has(input:checked)`, inside a list of fixed
height. Everything that makes that read as one interaction instead of a reflow
lives in the stylesheet and nowhere else, so it can only be contracted against
the shipped CSS — where a broken rule stays invisible to the JS-level tests.
"""

from __future__ import annotations

import re

from tests.site.css_helpers import bodies

# The selector that opens a row's own reveal: the whole connection, in CSS.
_CHECKED = ".provider-option:has(input:checked)"
_REVEAL = ".option-reveal"

_REM_PX = 16
_PANE_TOKEN = re.compile(r"--stage-list-h:\s*([^;}]+)")


def _split_top_level(expression: str) -> tuple[str, str]:
    depth = 0
    for index, char in enumerate(expression):
        depth += char == "("
        depth -= char == ")"
        if char == "," and depth == 0:
            return expression[:index], expression[index + 1 :]
    raise ValueError(f"no top-level comma in {expression!r}")


def _pane_floor_px(declaration: str) -> float:
    """A pane token's minimum height over every viewport.

    Every term is monotonic in viewport height, so evaluating the `min`/`max`
    tree with `svh` at 0 yields the token's floor — the height the pane can
    never drop below.
    """
    match = _PANE_TOKEN.search(declaration)
    assert match, "rule body has no --stage-list-h declaration"

    def value(node: str) -> float:
        node = node.strip()
        if node.startswith(("min(", "max(")):
            left, right = _split_top_level(node[node.index("(") + 1 : -1])
            pair = (value(left), value(right))
            return min(pair) if node.startswith("min(") else max(pair)
        if node.endswith("rem"):
            return float(node[:-3]) * _REM_PX
        if node.endswith("svh"):
            return 0.0
        return float(node)

    return value(match.group(1))


def test_pane_floor_keeps_the_tallest_reranker_row_reachable() -> None:
    """The pane must never shrink below the row it has to show.

    A required reranker forces its form open, so the list must hold a row
    taller than the mobile cap. A token capped purely by viewport height — the
    former `min(40rem, 72svh)` — collapses to a zero floor on a short viewport,
    so no scroll position could show the form's bottom. The wide-layout pane
    therefore carries a fixed rem floor, which is why it only needs checking
    from 640px up; a phone cannot hold a row-sized pane at all, so its pane is
    bounded by the available height instead (see the mobile contract below).
    """
    for media in ("640px", "1024px"):
        declarations = [
            body
            for body in bodies(".setup-builder", media=media)
            if "--stage-list-h" in body
        ]
        assert declarations, f"no --stage-list-h declaration at {media}"
        for declaration in declarations:
            assert _pane_floor_px(declaration) > 0, (
                f"the pane at {media} is capped only by viewport height, so it "
                "collapses below the required-reranker row and strands the "
                "form's bottom"
            )


def test_panel_body_is_height_locked_so_a_pick_never_reflows_the_page() -> None:
    """The body — not the list — carries the fixed `--stage-list-h`.

    Holding the filter and the list under one fixed height keeps a panel with a
    filter the same height as one without, so switching tabs never reflows the
    card; and the checked row's reveal grows the body's scroll content, not the
    card, so a pick never moves the output beside it either.
    """
    bodies_ = bodies(".detail-body", media="")
    assert bodies_, "No .detail-body rule in the built CSS"
    assert any("height:var(--stage-list-h)" in rule for rule in bodies_), (
        "the panel body no longer locks the pane to --stage-list-h"
    )
    assert not any("max-height" in rule for rule in bodies_), (
        "a max-height on the body lets the checked reveal resize the card"
    )


def test_mobile_pane_is_bounded_by_the_available_height() -> None:
    """The base pane is a bounded fraction of the live viewport.

    A phone's pane is smaller than the row it holds, so it can be neither
    unlocked (the card would run to ~1.2k px) nor capped by a fixed viewport
    fraction (the former `min(22rem, 46svh)` split the picked row). `dvh` is the
    only unit that tracks the height actually available, and the `clamp` gives
    it both a floor and the tablet cap, so the pane can neither collapse nor
    grow with its content. `overscroll-behavior: contain` keeps the bounded
    pane's scroll from chaining into the page.
    """
    tokens = [
        body
        for body in bodies(".setup-builder", media="")
        if "--stage-list-h" in body
    ]
    assert tokens, "the base builder declares no pane height"
    assert all("dvh" in body for body in tokens), (
        "the phone's pane is not sized from the available viewport height"
    )
    assert all("clamp(" in body for body in tokens), (
        "the phone's pane has no floor/cap, so it can grow with its content"
    )
    lists = bodies(".provider-list", media="")
    assert lists, "No unconditional .provider-list rule in the built CSS"
    assert any("overscroll-behavior:contain" in rule for rule in lists), (
        "the bounded pane's scroll chains into the page"
    )


def test_collapsed_reveals_are_unreachable_by_keyboard() -> None:
    """`visibility:hidden` is what keeps a closed row's linked chips out of the
    tab order and `0fr` is what keeps them out of the layout; only the checked
    row may restore both, and it must restore them together."""
    closed = bodies(_REVEAL, media="")
    assert closed, "No .option-reveal rule in the built CSS"
    assert all(
        "visibility:hidden" in rule and "grid-template-rows:0fr" in rule
        for rule in closed
    ), "a collapsed reveal still renders its links"

    opened = bodies(f"{_CHECKED} {_REVEAL}")
    assert opened, "No checked-row rule opens the reveal in the built CSS"
    assert all(
        "visibility:visible" in rule and "grid-template-rows:1fr" in rule
        for rule in opened
    ), "the checked row does not restore both reachability and height"


def test_reveal_motion_stops_under_reduced_motion() -> None:
    """Every reveal transition runs on a duration token, and reduced motion
    zeroes those tokens.

    Zeroing tokens — not `transition: none` — is what makes the promise hold:
    the checked-row rules out-specify any reduced-motion selector this stylesheet
    can name, so a direct `transition: none` override silently loses to them.
    Any new reveal element that hardcodes a duration breaks this test.
    """
    for selector in (_REVEAL, ".option-reveal-content"):
        declared = [rule for rule in bodies(selector, media="") if "transition" in rule]
        assert declared, f"{selector} declares no transition in the built CSS"
        assert all("var(--reveal-" in rule for rule in declared), (
            f"{selector} hardcodes a reveal duration instead of a token"
        )

    still = bodies(".stage-detail", media="prefers-reduced-motion")
    assert still, "the stage body lost its prefers-reduced-motion rule"
    for token in ("--reveal-enter-ms:0s", "--reveal-exit-ms:0s"):
        assert any(token in rule for rule in still), (
            f"{token} is not zeroed under prefers-reduced-motion"
        )


def test_recommended_badge_survives_the_checked_rows_highlight() -> None:
    """The badge only ever renders inside a provider row, and a checked row is
    filled with --primary-bg. A transparent fill lets that highlight bleed through
    the badge's own border, so the badge carries the raised-surface token — the
    same tile as the chips — which also keeps it theme-correct."""
    rules = bodies(".option-badge")
    assert rules, "No .option-badge rule in the built CSS"
    assert all("background:var(--bg-surface)" in rule for rule in rules), (
        "the recommended badge is not filled with the raised-surface token"
    )


def test_a_required_reranker_has_no_disclosure_to_collapse() -> None:
    """A required reranker is the pick's own field, not an option to reveal.

    Its form opens itself (`syncRerankerDisclosure` sets `open` for this state)
    and the summary — the only thing that could toggle it — is removed, or the
    form would read as optional and could be collapsed away.
    """
    rules = bodies(".rerank-endpoint[data-rerank-state=required] summary")
    assert rules, "the required reranker state no longer removes its summary"
    assert all("display:none" in rule for rule in rules), (
        "the required reranker form can still be collapsed"
    )


def test_desktop_output_and_list_share_one_height_bound() -> None:
    """Both desktop panes are bounded by the same `--stage-list-h`.

    The list's body locks it and the terminal's code area caps at it, so neither
    card's content can stretch the section, and the code scrolls instead of
    running the page to the length of the generated commands.
    """
    builder = bodies(".configurator-compact .setup-builder", media="1024px")
    assert builder, "No desktop builder rule in the built CSS"
    # The minifier drops the space in `minmax(0,1fr)`, so compare space-free.
    assert any(
        "grid-template-columns:minmax(0,1fr)minmax(0,1fr)"
        in rule.replace(" ", "")
        for rule in builder
    ), "the desktop layout no longer splits into config and output columns"

    code_area = bodies(".configurator-compact .setup-output .code-panel pre")
    assert code_area, "the desktop layout no longer sizes the terminal's code area"
    assert any("max-height:var(--stage-list-h)" in rule for rule in code_area), (
        "the terminal's code area is not bounded by --stage-list-h"
    )


def test_selected_option_out_reads_hover() -> None:
    """Identical hover and selected states leave users unable to tell which row
    owns the open reveal, so the checked row must add emphasis — the product's
    side-neutral inset ring (same idiom as the stage tiles). Emphasising one
    border side instead is the one-off the rest of the configurator avoids, and
    it rests the accent against the list's scrollbar."""
    rules = bodies(_CHECKED)
    assert rules, "No selected-option rule in the built CSS"
    assert any(
        "box-shadow:inset 0 0 0 var(--border-w-default) var(--primary)" in rule
        for rule in rules
    ), "the selected row no longer out-reads hover"
    assert not any(
        "border-right-width" in rule or "border-left-width" in rule for rule in rules
    ), "the selected row emphasises a single border side instead of using the ring"
