"""The scale section's terms rail: the conditions and interfaces relocated from
the hero.

The rail is the landing's static terms register — the conditions the product runs
on (free & open-source · local-first · nothing leaves your machine · your models,
your bill) and the interfaces you reach it through (CLI · MCP · GitHub). It sits
in the ProvenAtScale section, directly below the engine lede whose "running on
your machine" it makes concrete, and it stays lit: nothing about it tracks the
demo run, so a note's `data-condition` -> `data-promise` mapping is markup, never
animation.

It obeys the section's token contract (marketing-surface tokens only, no
`--code-*`): accent icons, muted labels at the section's body register, no chip
chrome — a term reads as a term, never as a control.

It also has to read as part of the section it sits in: unlabelled, at the lead
size in primary ink, it out-shouted the prose it follows, so its labels moved to
the body tier and its 4+3 seam is carried by a wider gap than the one between
items (no divider, which would need a magic breakpoint to hide once the groups
stack).
"""

from __future__ import annotations

from tests.site.css_helpers import bodies, rules_containing
from tests.site.dom_helpers import browser_dom, dist_body
from tests.site.tsx_runner import run_tsx_json

_HOMEPAGE = "index.html"

_RAIL = """
const doc = window.document;
const scale = doc.getElementById('scale');
const rail = scale.querySelector('.terms');
const fold = scale.querySelector('.engine-fold');
const quote = scale.querySelector('.trust-quote');
const items = [...rail.querySelectorAll('.terms-item')];
console.log(JSON.stringify({
  inScale: rail !== null,
  inHero: doc.querySelector('.terminal-card .terms') !== null,
  // Placement: the rail follows the engine lede and precedes the quote, so the
  // "running on your machine" claim is what names its terms.
  ordered:
    fold.compareDocumentPosition(rail) === 4
    && rail.compareDocumentPosition(quote) === 4,
  labels: items.map((t) => t.textContent.trim()),
  hasIcon: items.map((t) => t.querySelector('.terms-icon') !== null),
  iconHidden: items.map(
    (t) => t.querySelector('.terms-icon')?.getAttribute('aria-hidden'),
  ),
  promise: items.map((t) => t.dataset.promise ?? null),
  runState: items.map(
    (t) => ['data-earned', 'data-ambient'].filter((a) => t.hasAttribute(a)),
  ),
  role: rail.getAttribute('role'),
  ariaLabel: rail.getAttribute('aria-label'),
  githubIsLink: rail.querySelector('a.terms-link[href*="github.com"]') !== null,
  githubHasStar: rail.querySelector('a.terms-link [data-star-count]') !== null,
  groups: [...rail.querySelectorAll('.terms-group')].map(
    (g) => g.querySelectorAll('.terms-item').length,
  ),
  // The seam hairline: decorative, and between the two groups it divides.
  dividerBetween: (() => {
    const [conditions, interfaces] = rail.querySelectorAll('.terms-group');
    const divider = rail.querySelector('.terms-divider');
    return divider !== null
      && conditions.compareDocumentPosition(divider) === 4
      && divider.compareDocumentPosition(interfaces) === 4;
  })(),
  dividerHidden: rail.querySelector('.terms-divider')?.getAttribute('aria-hidden'),
}));
"""


def _rendered() -> dict:
    return run_tsx_json(browser_dom(dist_body(_HOMEPAGE)) + _RAIL)


def test_scale_terms_rail_names_conditions_then_interfaces() -> None:
    """One icon-led rail in the scale section: the four conditions, then the two
    interfaces, then the repo link — every item decorative-icon-safe, the repo a
    real link carrying the live star count, the group labelled for assistive
    tech, and split into two wrap units so a narrow container can never split the
    interface pair across rows."""
    r = _rendered()

    assert r["inScale"] is True, "the terms rail must live in the scale section"
    assert r["inHero"] is False, "the rail must no longer sit inside the hero"
    assert r["ordered"] is True, (
        "the rail must follow the engine lede and precede the trust quote"
    )

    assert r["labels"][:4] == [
        "Free & open-source",
        "Local-first",
        "Nothing leaves your machine",
        "Your models, your bill",
    ]
    assert r["labels"][4] == "CLI"
    assert r["labels"][5] == "MCP"
    assert "GitHub" in r["labels"][6]
    assert all(r["hasIcon"]) and all(h == "true" for h in r["iconHidden"])

    assert r["githubIsLink"] is True
    assert r["githubHasStar"] is True
    assert r["role"] == "group"
    assert r["ariaLabel"] == "Conditions and interfaces"

    # Two wrap units: the four conditions, then the interfaces + the repo.
    assert r["groups"] == [4, 3]
    assert r["dividerBetween"] is True, (
        "the seam hairline must sit between the groups it divides"
    )
    assert r["dividerHidden"] == "true", "the seam hairline is decorative"
    assert {p for p in r["promise"] if p} == {
        "open-source",
        "local-first",
        "nothing-leaves",
        "your-models",
    }, "every condition must name the beat (or lack of one) that proves it"


def test_scale_terms_rail_is_static_chrome_not_a_scoreboard() -> None:
    """The rail is chrome: always lit, nothing about it tracks the run. A rule
    that dims a terms item — or an earned/ambient hook on it — would make the
    claim depend on the animation again, so both absences are asserted."""
    assert all(state == [] for state in _rendered()["runState"]), (
        "the terms rail must not carry run state"
    )
    assert not rules_containing(".terms-item[data-promise]")
    for attr in ("data-earned", "data-ambient"):
        assert not rules_containing(attr), (
            f"the terms rail's {attr} contract is gone: the rail is static"
        )


def test_scale_terms_rail_uses_marketing_surface_tokens() -> None:
    """The rail paints on the section's band, so it may use marketing-surface
    tokens only — a `--code-*` token is a dark-surface colour and would fail on
    the light band (the section's own token contract)."""
    for selector in (
        ".terms",
        ".terms-group",
        ".terms-item",
        ".terms-icon",
        ".terms-star",
        "a.terms-link",
    ):
        for body in bodies(selector, media=""):
            assert "--code-" not in body, f"{selector} leaks a code token: {body}"


def test_scale_terms_rail_wrap_units_share_one_row() -> None:
    """Each wrap unit lays its items out in one wrapping row, so the rail peels
    at the seam between conditions and interfaces rather than breaking mid-group."""
    group = "".join(bodies(".terms-group", media="")).replace(" ", "")
    assert "display:inline-flex" in group and "flex-wrap:wrap" in group, (
        "each wrap unit must lay its items out in one wrapping row"
    )


def test_scale_terms_read_at_the_body_tier() -> None:
    """The rail names the terms behind the lede above it, so it sits below that
    section's lead prose instead of in the claim register. At the lead size in
    primary ink it out-shouted the prose it followed (2.5x that prose's contrast
    at the same size), and its width at the lead size is also what made the
    1600px type-scale step overflow the measure into two rows while 1440px
    stayed on one."""
    item = "".join(bodies(".terms-item", media="")).replace(" ", "")
    assert "font-size:var(--text-base)" in item, item
    assert "color:var(--text-secondary)" in item, item
    assert "font-weight:500" in item, item
    assert "--text-lg" not in item and "--text-primary" not in item, (
        f"the rail is back in the claim register: {item}"
    )

    for selector in (".terms-star", "[data-star-count]"):
        bodies_ = [rule.body for _, rule in rules_containing(selector)]
        assert bodies_, f"{selector} is unstyled"
        for body in bodies_:
            assert "--text-tertiary" in body, f"{selector} reads as a claim: {body}"


def test_scale_terms_seam_separates_the_two_groups() -> None:
    """The seam is the row's only structure, so it must read as structure: the
    section's hairline grammar (a 1px mark at the text's own height) plus a gap
    wider than the one between items — and the hairline must be dropped in the
    tier where the groups stack, or it strands at the start of the second row."""
    rail = "".join(bodies(".terms", media="")).replace(" ", "")
    group = "".join(bodies(".terms-group", media="")).replace(" ", "")
    assert "gap:var(--space-2)var(--space-4)" in rail, rail
    assert "gap:var(--space-2)" in group, group

    hairline = "".join(bodies(".terms-divider", media="")).replace(" ", "")
    assert "--border-color" in hairline, hairline
    assert "border-left" in hairline, hairline
    stacked = "".join(bodies(".terms-divider", media="1250px")).replace(" ", "")
    assert "display:none" in stacked, (
        f"the seam hairline must vanish where the groups stack: {stacked}"
    )
