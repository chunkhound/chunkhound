"""Built-CSS parsing for the site's stylesheet contracts.

Astro extracts component styles into content-hashed bundles under
`site/dist/_astro` and a page ships all the ones it imports concatenated, so a
rule can only be contracted against every bundle at once — never a guessed
filename. Astro's `[data-astro-cid-*]` scoping attributes are dropped: they
identify the component that authored a rule, not the rule itself.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

DIST = Path(__file__).resolve().parents[2] / "site" / "dist"

_SCOPING = re.compile(r"\[data-astro-cid-[a-z0-9]+\]")


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


@lru_cache(maxsize=1)
def _shipped() -> dict[str, list[Rule]]:
    bundles = sorted((DIST / "_astro").glob("*.css"))
    assert bundles, "No CSS bundles in site/dist/_astro — was the site built?"
    raw = _SCOPING.sub(
        "", "".join(bundle.read_text(encoding="utf-8") for bundle in bundles)
    )
    return _index(raw)


def rules(selector: str, *, media: str | None = None) -> list[Rule]:
    """Every shipped rule whose selector list contains `selector` exactly.

    `media` narrows by enclosing condition: `""` keeps only the unconditional
    rules, any other value matches a substring — `media="1024px"` or
    `media="prefers-reduced-motion"` asks for that condition's rules alone.
    """
    found = _shipped().get(selector, [])
    if media is None:
        return found
    return [
        rule
        for rule in found
        if (rule.condition == "" if media == "" else media in rule.condition)
    ]


def bodies(selector: str, *, media: str | None = None) -> list[str]:
    """Declaration blocks behind :func:`rules`."""
    return [rule.body for rule in rules(selector, media=media)]


def rules_containing(fragment: str) -> list[tuple[str, Rule]]:
    """(selector, rule) for every shipped selector containing `fragment`.

    The substring seam is for families (`".disclaimer"` then matches the block
    and its label/text children) where only the shape of the rules matters.
    """
    return [
        (selector, rule)
        for selector, selector_rules in _shipped().items()
        if fragment in selector
        for rule in selector_rules
    ]


_FONT_SIZE = re.compile(r"font-size:\s*([^;}]+)")


def raw_font_sizes() -> list[tuple[str, str]]:
    """(selector, value) for every shipped font size that is not a type token.

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
