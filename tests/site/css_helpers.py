"""Built-CSS parsing for the site's stylesheet contracts.

Astro extracts component styles into content-hashed bundles under
`site/dist/_astro`. Read only bundles linked by the requested page, plus its
inline styles: `index.html` by default. Astro's `[data-astro-cid-*]` scoping attributes
are dropped: they identify the component that authored a rule, not the rule.
"""

from __future__ import annotations

import re
from functools import cache
from typing import NamedTuple

from tests.site.dom_helpers import DIST

_SCOPING = re.compile(r"\[data-astro-cid-[a-z0-9]+\]")
_INLINE_STYLE = re.compile(r"<style(?:\s[^>]*)?>(.*?)</style>", re.DOTALL)

# Existing callers contract homepage components, so the public view defaults there.
DEFAULT_PAGE = "index.html"


class Rule(NamedTuple):
    """One rule as shipped, tagged with the enclosing @media/@container
    condition it sits inside ("" if any)."""

    condition: str
    body: str


def _blocks(css: str) -> list[tuple[str, str]]:
    """(header, body) for each top-level brace group, headers kept verbatim."""
    blocks: list[tuple[str, str]] = []
    depth = start = header = 0
    for index, char in enumerate(css):
        if char == "{":
            if depth == 0:
                header = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                blocks.append((css[start:header].strip(), css[header + 1 : index]))
                start = index + 1
    return blocks


def _selector_list(header: str) -> list[str]:
    """A selector list, split on the commas that sit outside every parenthesis."""
    parts: list[str] = []
    current = ""
    depth = 0
    for char in header:
        depth += char in "(["
        depth -= char in ")]"
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    return [part.strip() for part in [*parts, current] if part.strip()]


def _index(css: str) -> dict[str, list[Rule]]:
    rules: dict[str, list[Rule]] = {}

    def walk(text: str, condition: str) -> None:
        for header, body in _blocks(text):
            if header.startswith(("@media", "@container")):
                walk(body, header)
            elif header.startswith("@"):
                continue  # @keyframes / @font-face carry no selector to match
            else:
                for selector in _selector_list(header):
                    rules.setdefault(selector, []).append(Rule(condition, body))

    walk(css, "")
    return rules


@cache
def shipped_css(page: str = DEFAULT_PAGE) -> str:
    """The stylesheet one built page receives, linked bundles included.

    Rules shipped only to another page are invisible here. Pass a dist-relative
    path, e.g. `"docs/workflows/index.html"`, to read that page instead.
    """
    source = DIST / page
    assert source.is_file(), f"No built page {page!r} — was the site built?"
    # Astro inlines small stylesheets per page instead of emitting a bundle.
    html = source.read_text(encoding="utf-8")
    inline = _INLINE_STYLE.findall(html)
    links = re.findall(r'href="(/_astro/[^"?#]+\.css)"', html)
    bundles = [(DIST / link.lstrip("/")).read_text(encoding="utf-8") for link in links]
    return "".join(bundles + inline)


@cache
def _shipped(page: str = DEFAULT_PAGE) -> dict[str, list[Rule]]:
    return _index(_SCOPING.sub("", shipped_css(page)))


def rules(
    selector: str, *, media: str | None = None, page: str = DEFAULT_PAGE
) -> list[Rule]:
    """Every rule in `page`'s stylesheet whose selector list contains `selector`.

    `media` narrows by enclosing condition: `""` keeps only the unconditional
    rules, any other value matches a substring — `media="1024px"` or
    `media="prefers-reduced-motion"` asks for that condition's rules alone.
    `page` defaults to `index.html`; name another built page to read its view.
    """
    return [
        rule
        for rule in _shipped(page).get(selector, [])
        if media is None
        or (rule.condition == "" if media == "" else media in rule.condition)
    ]


def bodies(
    selector: str, *, media: str | None = None, page: str = DEFAULT_PAGE
) -> list[str]:
    """Declaration blocks behind :func:`rules`, scoped to `page`'s stylesheet."""
    return [rule.body for rule in rules(selector, media=media, page=page)]


def rules_containing(
    fragment: str, *, page: str = DEFAULT_PAGE
) -> list[tuple[str, Rule]]:
    """(selector, rule) for every selector in `page`'s stylesheet with `fragment`.

    The substring seam is for families (`".disclaimer"` then matches the block
    and its label/text children) where only the shape of the rules matters.
    """
    return [
        (selector, rule)
        for selector, selector_rules in _shipped(page).items()
        if fragment in selector
        for rule in selector_rules
    ]


_FONT_SIZE = re.compile(r"font-size:\s*([^;}]+)")


def raw_font_sizes() -> list[tuple[str, str]]:
    """(selector, value) for every non-type-token font size in `index.html`'s CSS.

    The type scale is the site's only source of font sizes; a literal here means
    a component invented a size nobody specified (DESIGN_SYSTEM, Type Scale).
    """
    found = {
        (selector, match.group(1).strip())
        for selector, selector_rules in _shipped().items()
        for rule in selector_rules
        for match in _FONT_SIZE.finditer(rule.body)
        if not match.group(1).strip().startswith("var(--text")
    }
    return sorted(found)
