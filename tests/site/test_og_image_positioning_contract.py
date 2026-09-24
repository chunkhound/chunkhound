"""OG-image headline positioning contract.

site/src/lib/positioning.json is the canonical source for the site's marketing
headline; the social-card SVGs are hand-maintained copies. A headline edit that
updates only one side ships stale social cards, so both og-image SVGs must
contain the exact positioning headline. Reads the sources (not dist) so the
guard fails on the edit, not on a stale build.
"""

from __future__ import annotations

import json

from tests.site.tsx_runner import ROOT

POSITIONING = ROOT / "site" / "src" / "lib" / "positioning.json"
OG_SVGS = (
    ROOT / "site" / "public" / "og-image-dark.svg",
    ROOT / "site" / "public" / "og-image-light.svg",
)


def test_og_images_carry_the_positioning_headline() -> None:
    """Both social-card SVGs render the exact positioning.json headline."""
    headline = json.loads(POSITIONING.read_text(encoding="utf-8"))["headline"]
    for svg in OG_SVGS:
        assert headline in svg.read_text(encoding="utf-8"), (
            f"{svg.name} drifted from positioning.json headline {headline!r}"
        )
