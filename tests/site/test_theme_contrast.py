"""Site-wide theme contrast contracts.

Guards against the bug class where marketing sections mixed theme-dependent
text tokens with theme-invariant code surfaces (readable in dark mode, broken in
light mode):

1. Every foreground/background token combination the site CSS can produce must
   meet its documented contrast floor in both themes: WCAG AA (4.5:1) unless the
   pair is an explicitly sanctioned sub-AA pairing carrying its own floor + WHY.
   A sanctioned pair is asserted against that specific floor, so a change that
   makes it worse still fails. The matrix is derived from actual token usage in
   site CSS (not an opt-in pair list), so any new token or surface entering the
   CSS joins the check automatically. --text-muted is contracted for page-level
   surfaces only.
2. Components rendered on the homepage may not use code-surface tokens unless
   they genuinely depict code/terminal output (whitelist).
3. The explicit and OS-preference light-theme blocks must stay in sync.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from tests.site.contrast_helpers import (
    GLOBAL_CSS,
    contrast_ratio,
    extract_block,
    extract_tokens,
    theme_tokens,
)

ROOT = Path(__file__).resolve().parents[2]
SITE_SRC = ROOT / "site" / "src"
INDEX = ROOT / "site" / "src" / "pages" / "index.astro"
COMPONENTS = ROOT / "site" / "src" / "components"

# Components that legitimately render code/terminal surfaces on the homepage.
CODE_SURFACE_WHITELIST = {"Hero.astro", "Configurator.astro"}

CODE_TOKENS = ("--code-bg", "--code-text", "--code-accent", "--code-muted")

# Brand tokens are theme-stable — the closing section is always dark — so their
# pairs are pinned once rather than re-resolved per theme. Only the text tokens
# are validated against the brand surface; the CTA label is validated against
# the brand fill only (its sole legal surface).
BRAND_TOKENS = (
    "--brand-surface",
    "--on-brand-surface",
    "--on-brand-muted",
    "--brand-cta",
    "--on-brand-cta",
    "--brand-border",
)
BRAND_TEXT_TOKENS = ("--brand-cta", "--on-brand-surface", "--on-brand-muted")

STYLE_BLOCK = re.compile(r"<style[^>]*>(.*?)</style>", re.DOTALL)
CSS_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
FOREGROUND_USE = re.compile(r"(?<![-\w])color:\s*var\((--[\w-]+)\)")
BACKGROUND_USE = re.compile(r"(?<![-\w])background(?:-color)?:\s*var\((--[\w-]+)\)")

# Tokens intentionally outside the resolved matrix: --bg-band is a color-mix()
# between --bg-page and --bg-surface; the rest are per-block Shiki/terminal
# token colors defined inline in built HTML (code depictions gated by the
# homepage code-token whitelist test). Adding a token here requires a comment.
UNRESOLVED_TOKENS = {
    "--bg-band",
    # Hero note rail/connector: transparent at rest / --code-accent while its
    # beat is live. A decorative 3px mark, not a text surface, so it carries the
    # 3:1 non-text contrast rather than the 4.5:1 text matrix.
    "--hero-note-rail",
    # Hero hound mark's mask fill (Hero-local alias of --code-text): a
    # decorative silhouette beside the hero CTA, never a text surface —
    # carries the 3:1 non-text contract, not the 4.5:1 text matrix.
    "--hero-hound-fill",
    "--shiki-dark",
    "--terminalSuccess",
    "--themeCmd",
    "--themeComment",
    "--themeJsonKey",
    "--themeOp",
    "--themeString",
    "--themeText",
}


def _css_sources() -> Iterator[str]:
    """Yield every CSS block the site ships: stylesheets + astro style blocks."""
    for path in sorted(SITE_SRC.rglob("*.css")):
        yield path.read_text(encoding="utf-8")
    for path in sorted(SITE_SRC.rglob("*.astro")):
        yield from STYLE_BLOCK.findall(path.read_text(encoding="utf-8"))


def _used_tokens() -> tuple[set[str], set[str]]:
    """Tokens actually used as text color vs surface across site CSS."""
    foregrounds: set[str] = set()
    backgrounds: set[str] = set()
    for css in _css_sources():
        for _, body in CSS_RULE.findall(css):
            foregrounds.update(FOREGROUND_USE.findall(body))
            backgrounds.update(BACKGROUND_USE.findall(body))
    return foregrounds, backgrounds


def _is_structural_exclusion(foreground: str, background: str) -> bool:
    """Pairings the CSS cannot legally produce — not contrast debt.

    A filled chip hosts only its own foreground; code tokens may sit only on code
    surfaces; brand tokens are theme-stable and legal only on brand tones.
    Excluding these keeps the matrix honest without loosening the AA floor for
    real text pairings.
    """
    if background == "--primary":
        # filled primary chip hosts only --on-primary
        return foreground != "--on-primary"
    if background == "--primary-bright":
        return True  # bright accent is decorative (icons); never a text surface
    if background == "--green-bg":
        return foreground != "--on-green-bg"  # green chip hosts only --on-green-bg
    if background == "--code-bg":
        # on --code-bg only code tokens are checked
        return foreground not in CODE_TOKENS
    if background == "--code-note-bg":
        # the hero's annotation surface is a code surface too: only code tokens
        # may sit on it (--code-muted is the note's body text, 4.9:1)
        return foreground not in CODE_TOKENS
    if foreground.startswith("--code-"):
        # theme-invariant code tokens pass AA only on --code-bg (checked above)
        return True
    if foreground in ("--on-primary", "--on-green-bg"):
        return True  # inverse chip text; their companion surface is checked above
    if foreground == "--primary-bright":
        return True  # decorative icon color: 3:1 non-text contract, not 4.5:1 text
    if foreground == "--text-muted":
        # contracted for --bg-page only (module docstring)
        return background != "--bg-page"
    if background == "--brand-cta":
        # the brand fill hosts only its own foreground
        return foreground != "--on-brand-cta"
    if background == "--brand-surface":
        # on the brand surface only brand text tokens are checked
        return foreground not in BRAND_TEXT_TOKENS
    if foreground in BRAND_TOKENS:
        # theme-stable brand tokens are only legal on the brand tones
        return True
    return False


# Sanctioned sub-AA floors. Each pair is deliberately below AA today; the floor
# is the WORST ratio measured across themes, so the check still fails when a
# token change worsens the pair and only passes an improvement. Keep the numbers
# in sync with what the test measures — a stale, inflated floor would hide a
# regression.
_DOCUMENTED_FLOORS: dict[tuple[str, str], tuple[float, str]] = {
    ("--link", "--bg-muted"): (
        4.37,
        "link on a muted chip: just under AA (light)",
    ),
    ("--primary", "--bg-muted"): (
        4.37,
        "primary accent on a muted chip: just under AA (light)",
    ),
    ("--danger", "--bg-muted"): (
        4.31,
        "danger accent on a muted chip: just under AA (dark)",
    ),
    ("--text-tertiary", "--bg-muted"): (
        3.52,
        "uppercase micro-label on a muted chip",
    ),
    ("--text-tertiary", "--primary-bg"): (
        3.92,
        "uppercase micro-label on a tinted chip",
    ),
}

# --green is a status accent used as a decorative mark on several surfaces, so it
# carries the 3:1 non-text contract rather than 4.5:1 text; its worst measured
# pairing is 2.69:1 (light theme, on --bg-muted).
_GREEN_FLOOR = (
    2.68,
    "status accent: decorative 3:1 mark, worst case 2.69:1 light on --bg-muted",
)


def _documented_floor(foreground: str, background: str) -> tuple[float, str] | None:
    """Return the (floor, why) for a sanctioned pair, else None (AA applies)."""
    if foreground == "--green":
        return _GREEN_FLOOR
    return _DOCUMENTED_FLOORS.get((foreground, background))


def test_contrast_matrix_stays_within_documented_floors() -> None:
    """Every used token combo clears its documented floor in both themes.

    The default floor is WCAG AA (4.5:1); sanctioned pairs carry an explicit
    lower floor and a WHY, and are asserted against that specific ratio so
    worsening a sanctioned pair still fails.
    """
    tokens_by_theme = theme_tokens()
    defined = tokens_by_theme["dark"]
    foregrounds, backgrounds = _used_tokens()
    unresolved = (foregrounds | backgrounds) - defined.keys() - UNRESOLVED_TOKENS
    assert not unresolved, (
        "Color tokens used in site CSS but not defined in global.css "
        f"(add to UNRESOLVED_TOKENS only if intentionally inline): {sorted(unresolved)}"
    )

    offenders = []
    for foreground in sorted(foregrounds & defined.keys()):
        for background in sorted(backgrounds & defined.keys()):
            if _is_structural_exclusion(foreground, background):
                continue
            sanctioned = _documented_floor(foreground, background)
            floor, why = sanctioned if sanctioned else (4.5, "WCAG AA")
            for theme, tokens in tokens_by_theme.items():
                ratio = contrast_ratio(tokens[foreground], tokens[background])
                if ratio < floor:
                    offenders.append(
                        f"{theme}: {foreground} on {background} = "
                        f"{ratio:.2f}:1 (floor {floor}:1 — {why})"
                    )
    assert not offenders, (
        "Used text/surface token pairs below their documented contrast floor:\n"
        + "\n".join(offenders)
    )


def test_homepage_components_reserve_code_tokens_for_code_depictions() -> None:
    index = INDEX.read_text(encoding="utf-8")
    imported = re.findall(r'import \w+ from "\.\./components/(\w+\.astro)"', index)
    assert imported, "No homepage component imports found in index.astro"

    offenders = []
    for component in imported:
        if component in CODE_SURFACE_WHITELIST:
            continue
        source = (COMPONENTS / component).read_text(encoding="utf-8")
        used = [token for token in CODE_TOKENS if f"var({token})" in source]
        if used:
            offenders.append(f"{component}: {', '.join(used)}")

    assert not offenders, (
        "Homepage components using code-surface tokens outside the "
        f"code-depiction whitelist {sorted(CODE_SURFACE_WHITELIST)}:\n"
        + "\n".join(offenders)
    )


def test_light_theme_blocks_define_identical_tokens() -> None:
    """The explicit toggle and OS-preference light blocks must not drift."""
    css = GLOBAL_CSS.read_text(encoding="utf-8")
    explicit = extract_tokens(extract_block(css, '[data-theme="light"]'))
    media = extract_tokens(extract_block(css, ':root:not([data-theme="dark"])'))

    assert explicit == media


def _rule_bodies(css: str) -> dict[str, str]:
    """Body text of every rule, keyed by each selector in its selector list.

    Comments are stripped first: a comment sits inside the selector group that
    precedes its rule, so it would otherwise be pasted onto the first selector.
    """
    bodies: dict[str, str] = {}
    for selectors, body in CSS_RULE.findall(
        re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    ):
        for selector in selectors.split(","):
            key = selector.strip()
            bodies[key] = bodies.get(key, "") + body
    return bodies


def test_brand_filled_controls_keep_their_foreground_in_link_states() -> None:
    """A control filled with --brand-cta must restate its foreground for :hover
    and :visited.

    The prose-link recipe (`a`, `a:visited`, `a:hover`) has specificity 0,1,1 and
    therefore outranks a lone class (0,1,0). A filled LINK that omits those states
    silently renders its label in --link-hover / --link-visited on its own fill —
    about 1.7:1, i.e. an unreadable button. This is the failure that appears the
    moment a filled control is an <a> instead of a <button>.
    """
    bodies = _rule_bodies(GLOBAL_CSS.read_text(encoding="utf-8"))

    # The prose-link recipe is what makes the state overrides necessary; if it
    # ever stops setting a foreground, this contract can be simplified.
    assert "color: var(--link-hover)" in bodies["a:hover"]

    for state in (".btn--brand", ".btn--brand:hover", ".btn--brand:visited"):
        assert state in bodies, f"{state} missing from global.css"
        assert "color: var(--on-brand-cta)" in bodies[state], (
            f"{state} must restate the fill's foreground; the prose-link recipe "
            "outranks it otherwise"
        )
        assert "text-decoration: none" in bodies[state], (
            f"{state} must cancel the prose-link hover underline"
        )
