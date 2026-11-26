"""Guard the built FAQ page's functional contracts (DOM probes)."""

import json
import re

from tests.site.dom_helpers import DIST, browser_dom, dist_body
from tests.site.html_helpers import meta_tag_content, visible_text
from tests.site.tsx_runner import ROOT, run_tsx_json

CATALOG = ROOT / "site" / "src" / "lib" / "language-catalog.gen.json"

# Read the visible language chips of the rendered page.
_PROBE = """
const languageCodes = [...document.querySelectorAll('code')]
  .map((c) => c.textContent.trim());
console.log(JSON.stringify({ languageCodes }));
"""


def _dom() -> dict:
    return run_tsx_json(browser_dom(dist_body("docs/faq/index.html")) + _PROBE)


def test_language_figures_equal_catalog(built_site) -> None:
    """The visible language count and chips derive from the catalog."""
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    dom = _dom()
    assert set(catalog["languages"]) <= set(dom["languageCodes"])
    prose = re.search(
        r"ChunkHound can index (\d+) languages:",
        visible_text(dist_body("docs/faq/index.html")),
    )
    assert prose and int(prose.group(1)) == len(catalog["languages"])


def test_retired_root_faq_route_redirects_to_docs_guide() -> None:
    """The old /faq/ marketing route must meta-refresh to /docs/faq/."""
    redirect = DIST / "faq" / "index.html"
    assert redirect.exists()
    content = meta_tag_content(
        redirect.read_text(encoding="utf-8"), "http-equiv", "refresh"
    )
    assert content is not None, "retired FAQ route lost its meta refresh redirect"
    target = re.search(r"url\s*=\s*['\"]?([^'\";]+)", content, re.IGNORECASE)
    assert target is not None, f"meta refresh has no redirect target: {content!r}"
    expected = "/docs/faq/"
    assert target.group(1).rstrip("/") == expected.rstrip("/"), (
        f"retired FAQ route redirects to {target.group(1)!r}, not {expected!r}"
    )
