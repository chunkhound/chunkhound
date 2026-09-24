"""Built-CSS contracts for the hero's responsive tiers.

The hero is ONE full-width terminal with ONE note layout: every beat's note hangs
inline below its receipt at every card width, railed, connected to the row it
annotates and indented to the copy column, and CSS alone places it — no script
chooses layout, so the no-JS render is the same DOM.

The card's inline width drives exactly two tiers, both container queries and
never a device guess:

- the **narrow tier** (card <= 479px): the run buys back the measure a visible
  role column would cost, and the seam's glyph key carries the mapping instead;
- from a 480px card up the role labels are visible.

The card cannot read its own cqi, so the run/composer measure lives on the
window inside it.

happy-dom has no layout engine, so these assert the shipped rules, not computed
geometry.
"""

from __future__ import annotations

import re

from tests.site.css_helpers import bodies, rules, rules_containing

# The narrow-card threshold, in the form Astro ships it (a range container query).
_NARROW_TIER = "@container hero-terminal (width<=479px)"


def _flat(body: str) -> str:
    """Whitespace-insensitive view of a shipped declaration block."""
    return body.replace(" ", "")


def _conditions(selector: str, declaration: str) -> list[str]:
    """Every enclosing @media/@container condition that sets `declaration`."""
    return [
        rule.condition
        for rule in rules(selector)
        if declaration in _flat(rule.body)
    ]


def test_hero_is_one_surface_with_no_second_register() -> None:
    """The runbook column, its legend and the two-column body are gone, and their
    CSS went with them: a note that lives inside the run cannot drift out of it,
    so a second register would be a second source of truth. Guarding the absence
    keeps a resurrected column from arriving as dead CSS that nothing renders."""
    dead_registers = (
        ".hero-body",
        ".hero-session",
        ".hero-rail",
        ".hero-legend",
        ".hero-masthead",
    )
    for dead in dead_registers:
        assert not rules_containing(dead), (
            f"the hero shipped a second register back: {dead}"
        )


def test_terminal_card_establishes_inline_size_container() -> None:
    """The tiers key to the card's own inline size, so the card must be a
    queryable container — without it every tier query silently never matches."""
    assert any(
        "container:hero-terminal/inline-size" in _flat(body)
        for body in bodies(".terminal-card", media="")
    ), "terminal card must establish the hero-terminal inline-size container"


def test_run_measure_is_capped_without_a_note_track() -> None:
    """The run, the composer and the window's titlebar share one measure, and it
    is declared on the window — never on the card, because a custom property
    resolves its own relative units where it is used and the card IS the
    container: it cannot read its own cqi, but its child can. With the note
    inline there is no note track to subtract and no token to toggle: the measure
    is simply capped so a line never runs the whole window."""
    assert any(
        "--hero-measure:min(100%,68ch)" in _flat(body)
        for body in bodies(".terminal-window", media="")
    ), "the window must declare the run's measure"

    assert not any(
        "--hero-measure" in body for body in bodies(".terminal-card", media="")
    ), "the container cannot read its own cqi — the measure lives on its child"

    assert not any(
        "--hero-note-w" in body or "--hero-note-gutter" in body
        for body in bodies(".terminal-viewport")
    ), "the note track machinery must be gone"


def test_note_names_its_source_class_at_every_tier() -> None:
    """The note's headline states a payoff, not a source class, so the kind label
    is the reader's only referent for what the receipt cites. No tier may hide
    it — three anonymous headlines is exactly the regression the old compact tier
    shipped."""
    hidden = [
        rule
        for selector, rule in rules_containing(".terminal-lines .note-kind")
        if "display:none" in _flat(rule.body)
    ]
    assert not hidden, (
        f"a tier hides the note's source label: {[r.condition for r in hidden]}"
    )


def test_note_is_inline_at_every_width() -> None:
    """One note layout serves every card width: the note never leaves the flow
    and no tier places it, so there is no margin to keep in sync and no band or
    band-ordinal chrome on the terminal rows. The rail is the only live
    highlight."""
    assert not any(
        "position:absolute" in _flat(rule.body)
        for rule in rules(".terminal-lines .note")
    ), "the note must stay in the flow at every width"
    assert not rules(".terminal-lines .line[data-current]"), (
        "the row band must be gone: the note's rail is the only live highlight"
    )
    assert not rules_containing("[data-ordinal]"), (
        "the band ordinal chip must be gone"
    )


def test_note_is_a_railed_offset_annotation() -> None:
    """The note reads as an annotation layer, not a row: a drawn rail that
    brackets it (leading edge plus bottom edge, invisible at rest, accent only
    while its beat is live), an indent that keeps it out of the run's measure
    column, Inter prose against the terminal's monospace, and its own surface
    (--code-note-bg). The run carries no band; there are no leader lines and no
    info glyph."""
    note = _flat("".join(bodies(".terminal-lines .note", media="")))
    assert (
        "linear-gradient(var(--hero-note-rail),var(--hero-note-rail))"
        "leftbottom/0var(--border-w-accent)no-repeat" in note
    ), (
        "the note must draw its bottom bar from --hero-note-rail, so the live "
        "accent reaches it and an off note draws nothing"
    )
    assert "--hero-note-rail:transparent" in note, (
        "the rail must vanish at rest, so an off note carries no bracket"
    )
    current_rail = _flat(
        "".join(bodies(".terminal-lines .note[data-current]", media=""))
    )
    assert "--hero-note-rail:var(--code-accent)" in current_rail, (
        "only the live beat's rail may take the accent — the highlight travels"
    )
    assert "border-inline-end" not in note, (
        "the rail lives on the leading edge alone"
    )
    assert "margin-inline-start:var(--hero-note-indent)" in note, (
        "the note must be offset out of the run's measure column"
    )
    assert (
        "max-inline-size:min(calc(100%-var(--hero-note-indent)),"
        "var(--hero-note-measure))" in note
    ), "the note must be offset from the run's edge and capped below its measure"
    assert "border-radius:var(--radius-none)" in note, (
        "the note's accent edge must be a sharp, straight rail"
    )

    # The note's prose is a different type from the run's monospace: the layer
    # cannot read as another transcript row.
    for part in (".note-headline", ".note-detail"):
        prose = _flat("".join(bodies(f".terminal-lines {part}", media="")))
        assert "font-family:Inter,system-ui,sans-serif" in prose, (
            f"{part} must be set in Inter, not the run's monospace"
        )
    terminal = _flat("".join(bodies(".terminal-window", media="")))
    assert "font-family:JetBrainsMono,monospace" in terminal, (
        "the terminal must keep its monospace — the run inherits it from the "
        "window, so the note's Inter prose is a type contrast, not a variant"
    )

    # The note surface is the annotation's own: the run carries no band at all,
    # so the note never shares a highlight rectangle with the transcript.
    assert "var(--code-note-bg)" in note, (
        "the note's own surface is what marks the annotation"
    )
    assert not rules(".terminal-lines .line[data-current]"), (
        "the run must not flash a band: the highlight lives on the notes alone"
    )

    assert not rules_containing(".note-icon"), (
        "the info glyph must be gone from the layer's grammar"
    )
    assert not rules_containing(".note:after"), (
        "the old pointing leaders must not return"
    )


def test_note_hangs_from_the_row_it_annotates() -> None:
    """The note sits BELOW the call + receipt it explains and its rail runs up
    through the block-start gap as a stroke from the receipt, so the annotation
    reads as one node hanging off the receipt and can never read as applying to
    the beat above it. Drawn with a pseudo-element and a background bar (no
    borders), so the anchor's child contract stays [call, receipt, note] and each
    stroke can animate in its own direction."""
    note = _flat("".join(bodies(".terminal-lines .note", media="")))
    assert "margin-block-start:var(--hero-note-connect)" in note, (
        "the note must open the block-start gap its rail stroke runs through"
    )
    assert "position:relative" in note, (
        "the note must be the rail stroke's positioning context"
    )
    assert "margin-block-end" not in note, (
        "the note hangs below its rows, so it opens the gap, not closes it"
    )

    stroke = rules(".terminal-lines .note:before")
    assert stroke, "the note's rail stroke must exist"
    body = _flat(stroke[0].body)
    assert "background:var(--hero-note-rail)" in body, (
        "the stroke is the rail, so it shares its colour token"
    )
    assert "inset-block:calc(-1*var(--hero-note-connect))0" in body, (
        "the stroke must run from the receipt row down the note's whole leading "
        "edge, so it is one line with the bottom bar"
    )
    assert "transform:scaleY(0)" in body and "transform-origin:top" in body, (
        "the stroke must rest collapsed to the receipt end, to reel down from it"
    )

    tier = _flat("".join(bodies(".terminal-window", media="")))
    assert "--hero-note-connect:var(--space-3)" in tier, (
        "the connector length must be the shared block gap token"
    )


def test_note_rail_strokes_in_then_fills_across_the_beat() -> None:
    """The live note's bracket is drawn in: the leading edge reels down from the
    receipt, then the bottom edge fills left-to-right like a progress bar for the
    beat's own lifetime. The beat span is published by the run (one source with
    its timing), and the stroke rests collapsed, so an off note has no rail."""
    stroke = rules(".terminal-lines .note[data-current]:before", media="")
    assert stroke, "the live note must reel its leading edge down"
    for rule in stroke:
        body = _flat(rule.body)
        assert "hero-note-rail-down" in body, (
            "the leading edge must animate down, not appear at once"
        )
        assert "var(--hero-note-stroke)" in body, (
            "the reel must take the shared stroke token"
        )

    fill = [
        rule
        for rule in rules(".terminal-lines .note[data-current]")
        if "hero-note-progress" in _flat(rule.body)
    ]
    assert fill, "the live note's bottom edge must fill like a progress bar"
    body = _flat(fill[0].body)
    assert "var(--hero-note-beat" in body, (
        "the fill must run for the beat span the run publishes"
    )
    assert "var(--hero-note-stroke)" in body, (
        "the fill must start after the leading edge has reeled down"
    )

    tier = _flat("".join(bodies(".terminal-window", media="")))
    assert "--hero-note-stroke:" in tier, (
        "the reel's lead-in must be one shared token"
    )

    frozen = rules(
        ".terminal-lines .note[data-current]", media="prefers-reduced-motion"
    )
    assert any("animation:none" in _flat(rule.body) for rule in frozen), (
        "reduced motion must not reel the rail in"
    )


def test_note_copy_reveals_as_the_rail_draws_in() -> None:
    """The note's copy arrives with the rail rather than sitting in the tile
    before it draws: a short opacity + rise on the live note, on the same clock
    as the leading-edge reel. The base state is visible, so an off note (and the
    no-JS render) keeps its copy."""
    for part in (".note-head", ".note-detail"):
        reveal = [
            rule
            for rule in rules(f".terminal-lines .note[data-current] {part}")
            if "hero-note-content-in" in _flat(rule.body)
        ]
        assert reveal, f"{part} must reveal with the rail, not appear at once"
        assert "var(--hero-note-stroke)" in _flat(reveal[0].body), (
            f"{part}'s reveal must run on the rail's stroke clock"
        )

    frozen = rules(
        ".terminal-lines .note[data-current] .note-head",
        media="prefers-reduced-motion",
    )
    assert any("animation:none" in _flat(rule.body) for rule in frozen), (
        "reduced motion must not fade the copy in"
    )


def test_note_indent_tracks_the_copy_column() -> None:
    """The note aligns with the copy it annotates, never with the role labels.
    On a phone it takes a small indent (the role column is collapsed); from a
    480px card up it steps past the role + glyph columns. Both the line grid and
    the indent derive from --hero-role-col, so the note cannot drift from the
    copy column."""
    assert any(
        "--hero-role-col:10ch" in _flat(body)
        for body in bodies(".terminal-window", media="")
    ), "the role column must be one token the line grid and the note indent share"
    assert any(
        "grid-template-columns:var(--hero-role-col)auto1fr" in _flat(body)
        for body in bodies(".terminal-lines .line", media="")
    ), "the line grid must use the shared role-column token"

    assert any(
        "--hero-note-indent:var(--space-4)" in _flat(body)
        for body in bodies(".terminal-window", media="")
    ), "on a phone the note takes a small indent from the run edge"

    indent = _flat("".join(bodies(".terminal-window", media="480px")))
    assert (
        "--hero-note-indent:calc(var(--hero-role-col)+2*var(--space-3)+1ch)"
        in indent
    ), "from a 480px card up the note must step past the role + glyph columns"


def test_narrow_tier_collapses_the_role_column_and_buys_measure() -> None:
    """Cards <= 479px drop the visible 10ch role column from the run and the
    composer: the glyphs and the per-role colours carry identity instead, and the
    copy gets the freed measure. The label leaves the grid via `position` (not
    `display:none`) so its screen-reader text stays in the accessibility tree."""
    collapsed = _conditions(".terminal-lines .role-label", "position:absolute")
    assert collapsed == [_NARROW_TIER], (
        "only the narrow tier may collapse the role column"
    )
    assert _conditions(".terminal-input .role-label", "position:absolute") == [
        _NARROW_TIER
    ], "the composer must collapse its role column with the run's"

    assert any(
        "grid-template-columns:auto1fr" in _flat(body)
        for body in bodies(".terminal-lines .line", media="479px")
    ), "the narrow tier must drop the 10ch role column from the line grid"
    assert any(
        "grid-template-columns:var(--hero-role-col)auto1fr" in _flat(body)
        for body in bodies(".terminal-lines .line", media="")
    ), "above the narrow tier the role column is part of every row"


def test_role_labels_are_visible_from_a_480px_card_up() -> None:
    """The Agent / ChunkHound labels are visible from a 480px card up — desktop
    included — while the narrow tier collapses the column to screen-reader text.
    The unclip must not be capped at any upper width, which would hide every
    label on desktop. The user turn never shows a visible label."""
    selector = ".terminal-lines .call .role-label .sr-only"
    assert any(
        "position:static" in _flat(body)
        for body in bodies(selector, media="480px")
    ), "role labels must unclip from a 480px card up"

    conditions = [rule.condition for rule in rules(selector)]
    assert conditions, "the visible-label unclip rule must exist"
    assert all("479px" not in condition for condition in conditions), (
        "the label unclip must not reach into the narrow tier"
    )

    # The narrow tier must un-clip exactly what it clipped: the three labelled
    # roles. The user turn and the composer keep an sr-only label everywhere.
    for row in ("call", "answer", "evidence"):
        unclip = f".terminal-lines .{row} .role-label .sr-only"
        assert any(
            "position:static" in _flat(body) for body in bodies(unclip, media="480px")
        ), f".{row} must show its screen-reader role label from 480px up"
    assert not bodies(".terminal-lines .user-prompt .role-label .sr-only"), (
        "the user turn must not render a visible You label"
    )
    assert not bodies(".terminal-input .role-label .sr-only"), (
        "the composer must not render a visible You label"
    )


def test_narrow_tier_shows_the_glyph_key_and_hides_it_otherwise() -> None:
    """The key exists exactly where the visible role column does not: hidden by
    default, shown only in the narrow tier. It shares the seam row, so it costs
    no vertical space, and it is decorative — assistive tech reads the inline
    roles. It is static: no script reveals it as the run goes."""
    assert any(
        "display:none" in _flat(body) for body in bodies(".terminal-key", media="")
    ), "the glyph key must be hidden where the inline role labels are visible"
    assert _conditions(".terminal-key", "display:flex") == [_NARROW_TIER], (
        "only the narrow tier may show the glyph key"
    )


def test_titlebar_carries_the_session_control_at_the_runs_edge() -> None:
    """One titlebar row holds the decorative dots, the glyph key and the session
    control; the control is pushed to the titlebar's trailing edge. The titlebar
    shares the run's inline gutter and the column that measures the window, so
    that edge IS the run's right edge — chrome for the run, never a button parked
    at the page's far edge. The headline keeps the whole measure above it: there
    is no masthead left to wrap a control under."""
    seam = _flat("".join(bodies(".terminal-seam", media="")))
    assert "display:flex" in seam and "align-items:center" in seam, (
        "the titlebar must be one row of session chrome"
    )
    assert any(
        "margin-inline-start:auto" in _flat(body)
        for body in bodies(".terminal-control", media="")
    ), "the session control must sit at the titlebar's trailing edge"


def test_the_run_is_a_framed_window_at_the_runs_measure() -> None:
    """The titlebar, the scrollback and the composer are ONE framed object, so
    the run reads as a terminal window on the hero's full-bleed surface. The
    frame is a code token (the surface is theme-stable, so the frame must be) and
    flat — one hairline, one radius step above the composer field inside it, no
    shadow. The frame's edges ARE the hero column's edges (see
    `.terminal-card`), so the window ends where its copy does: the titlebar's
    control lands at the run's edge and the terminal holds no dead space beside
    the run."""
    window = _flat("".join(bodies(".terminal-window", media="")))
    assert "border:var(--border-w-default)solidvar(--code-muted)" in window, (
        "the window frame must be one theme-stable hairline"
    )
    assert "border-radius:var(--radius-md)" in window, (
        "the window is one radius step rounder than the field inside it"
    )
    assert "background:var(--code-bg)" in window, (
        "the window owns its surface explicitly, so the hero band can change"
    )
    assert "box-shadow" not in window, (
        "flat design: the frame can never be a shadow"
    )
    for row in (".terminal-seam", ".terminal-viewport", ".terminal-composer"):
        assert any(
            "padding-inline:var(--space-3)" in _flat(body)
            for body in bodies(row, media="")
        ), f"{row} must share the window's one inline gutter"


def test_the_window_column_hugs_the_run_measure() -> None:
    """The window column is the terminal's measure (68 mono ch at the terminal's
    max size) plus the window's inline gutter, so the frame's edges are the
    content edges and the run holds no dead space beside it. It is left-anchored
    with the section grid — the headline is a full-width sibling, not the card's
    cap (see test_hero_headline_spans_the_container). The card IS the container,
    so it cannot resolve that `ch` itself and restates it as a length — this test
    is what keeps the two figures in step."""
    card = _flat("".join(bodies(".terminal-card", media="")))
    assert "max-inline-size:var(--hero-col)" in card, (
        "the window column must cap the card, so the window spans it exactly"
    )
    assert "margin-inline:auto" not in card, (
        "the window column is left-anchored with the section grid, not centered"
    )
    column = re.search(r"--hero-col:min\(100%,(\d+)px\)", card)
    assert column, f"the window column must be one length token: {card[:200]}"

    window = _flat("".join(bodies(".terminal-window", media="")))
    measure = re.search(r"--hero-measure:min\(100%,(\d+)ch\)", window)
    # Only the clamp's ceiling matters: it is the size the desktop column renders
    # at, so it is the size the reading measure is exact at.
    font = re.search(r"--terminal-font-size:clamp\([^,]+,[^,]+,(\d+)px\)", window)
    assert measure and font, "the window must own the run's measure and type size"
    # JetBrains Mono advances 0.6em, so the reading measure in px is exact at the
    # clamp's ceiling — the only size the desktop column ever renders at.
    expected = round(int(measure.group(1)) * 0.6 * int(font.group(1))) + 24
    assert abs(int(column.group(1)) - expected) <= 1, (
        f"the window column ({column.group(1)}px) must be the run's measure plus "
        f"the two inline gutters ({expected}px)"
    )


def test_hero_headline_takes_the_space_beside_the_window_on_desktop() -> None:
    """Desktop only, the hero's container is one row: the window is ordered first
    (left) with an explicit basis, and the headline takes the space beside it
    (ordered second, growing). Below that width the container stays a block, so
    the headline keeps leading over the window."""
    container = _flat("".join(bodies(".hero .container", media="1100px")))
    assert "display:flex" in container, container

    headline = _flat("".join(bodies(".headline-row", media="1100px")))
    assert "order:2" in headline and "flex:110" in headline, headline

    card = _flat("".join(bodies(".terminal-card", media="1100px")))
    assert "order:1" in card, card
    assert "flex:00var(--hero-col)" in card, (
        "the window needs an explicit basis: inline-size containment zeroes auto"
    )

    # The two-column hero is desktop-only: below the tier the container is not a
    # row, so the headline keeps leading over the window.
    assert not bodies(".hero .container", media=""), (
        "the two-column hero must be desktop-only"
    )
    assert "max-width:none" in _flat("".join(bodies(".headline-row"))), (
        "the headline row must keep no measure cap inside its column"
    )


def test_composer_shares_the_run_measure() -> None:
    """The composer and the run are one object's two halves: both take the same
    `--hero-measure`, so the field can never be wider than the output it echoes."""
    for selector in (".terminal-lines", ".terminal-input"):
        assert any(
            "max-inline-size:var(--hero-measure)" in _flat(body)
            for body in bodies(selector, media="")
        ), f"{selector} must track the run's measure"


def test_short_viewport_turns_the_screen_into_scrollback() -> None:
    """A short viewport (e.g. a 667px-tall phone) cannot afford the locked
    transcript height — the composer would fall below the fold. The screen
    becomes a bounded, user-scrollable CLI buffer: the composer stays pinned and
    older rows scroll out above it. Only the block axis scrolls, and only under
    the short-viewport query, so the base clipping contract survives."""
    screen = bodies(".terminal-viewport", media="780px")
    assert screen, "the short-viewport screen rules must exist"
    flat = [_flat(body) for body in screen]
    assert any("max-height:" in body for body in flat), (
        "the screen must be bounded by a max-height, not the full transcript"
    )
    assert any("overflow-y:auto" in body for body in flat), (
        "the bounded screen must be user-scrollable"
    )
    assert any("overscroll-behavior:contain" in body for body in flat), (
        "the read gesture must not chain into page scroll"
    )
    assert any("scrollbar-width:none" in body for body in flat), (
        "a real terminal shows no scrollbar"
    )
    assert any(
        "display:none" in _flat(body)
        for body in bodies(".terminal-viewport::-webkit-scrollbar")
    ), "WebKit must hide the scrollbar too"

    assert any(
        "overflow:hidden" in _flat(body)
        for body in bodies(".terminal-viewport", media="")
    ), "the desktop clipping contract must survive the short-viewport rule"


def test_scrollback_affordances_are_gated_on_js_state() -> None:
    """Both scrollback affordances are CSS-owned and gated on the state JS
    mirrors: the history fade only when rows are hidden above, the jump control
    only while the reader is off the tail (and the base state hides it)."""
    fade = bodies(".terminal-card[data-scrollable=true] .terminal-viewport")
    assert any("mask-image:" in _flat(body) for body in fade), (
        "hidden history above must fade at the top edge"
    )

    jump = bodies(".terminal-jump", media="")
    assert any("display:none" in _flat(body) for body in jump), (
        "the jump control must be hidden while following the tail"
    )
    assert any("position:absolute" in _flat(body) for body in jump), (
        "the jump control must not join the composer's flow"
    )

    shown = bodies(".terminal-card[data-following=false] .terminal-jump")
    assert any("display:inline-flex" in _flat(body) for body in shown), (
        "scrolling off the tail must reveal the jump control"
    )


def test_composer_is_single_line_and_scrolls_right() -> None:
    """The prompt field never wraps: its copy column may shrink to nothing
    (minmax(0, 1fr)) so it fills the available width instead of stretching the
    composer, and the single-line text scrolls right so the caret stays visible.
    The narrow tier must keep the shrinkable column too."""
    composer = bodies(".terminal-input", media="")
    assert any("minmax(0,1fr)" in _flat(body) for body in composer), (
        "the composer's copy column must be allowed to shrink to the available width"
    )
    compact = bodies(".terminal-input", media="479px")
    assert any("minmax(0,1fr)" in _flat(body) for body in compact), (
        "the narrow tier must keep the shrinkable copy column"
    )

    line = bodies(".input-line", media="")
    assert any("overflow-x:auto" in _flat(body) for body in line), (
        "the single-line field must scroll horizontally"
    )
    assert any("white-space:nowrap" in _flat(body) for body in line), (
        "the field must never wrap"
    )
    assert any(
        "white-space:nowrap" in _flat(body)
        for body in bodies(".input-text", media="")
    ), "the typed prompt must never wrap"
