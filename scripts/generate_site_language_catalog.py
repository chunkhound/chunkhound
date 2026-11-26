"""Extract the site FAQ's supported-language catalog from the backend SSOT.

WHY: the FAQ must name exactly the languages the backend indexes. Those facts
already exist in the backend ``Language`` enum, so the site file is derived,
never hand-typed. It is committed; run --check after any Language change.

    uv run python scripts/generate_site_language_catalog.py          # write
    uv run python scripts/generate_site_language_catalog.py --check  # verify
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

from chunkhound.core.types.common import Language
from chunkhound.parsers._grammar_availability import SCSS_AVAILABLE

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "site/src/lib/language-catalog.gen.json"
REGENERATE = "uv run python scripts/generate_site_language_catalog.py"


def _catalog() -> dict[str, Any]:
    """The languages a user can index: sentinels and non-language formats
    (text, pdf) are excluded; scss only when its grammar is bundled."""
    non_languages = {Language.UNKNOWN, Language.GIT_DIFF, Language.TEXT, Language.PDF}
    languages = [member for member in Language if member not in non_languages]
    if not SCSS_AVAILABLE:
        languages.remove(Language.SCSS)
    return {
        "languages": sorted(member.value for member in languages),
        "buckets": _buckets(languages),
    }


def _buckets(languages: list[Language]) -> list[dict[str, Any]]:
    """Bucket straight from the backend predicates."""

    def members(predicate) -> list[str]:
        return sorted(member.value for member in languages if predicate(member))

    programming = members(lambda m: m.is_programming_language)
    structured_config = members(lambda m: m.is_structured_config_language)
    assigned = set(programming) | set(structured_config)
    other = sorted(member.value for member in languages if member.value not in assigned)
    return [
        {"label": "Programming languages", "members": programming},
        {"label": "Structured configuration and data", "members": structured_config},
        {"label": "Other formats", "members": other},
    ]


def render_catalog() -> str:
    """Deterministic: sorted ids, fixed field order, LF, no timestamps."""
    return json.dumps(_catalog(), indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the site language catalog."
    )
    parser.add_argument("--check", action="store_true", help="Fail if stale.")
    args = parser.parse_args(argv)

    expected = render_catalog()
    if not args.check:
        OUTPUT_PATH.write_text(expected, encoding="utf-8", newline="\n")
        return 0

    actual = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
    if actual == expected:
        return 0
    sys.stderr.write(
        f"{OUTPUT_PATH} is stale after a backend Language change.\n"
        f"Regenerate and commit it:\n  {REGENERATE}\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
