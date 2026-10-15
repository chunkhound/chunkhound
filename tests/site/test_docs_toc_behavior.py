"""Docs scrollspy follows the rendered TOC while every heading keeps a permalink."""

from tests.site.dom_helpers import browser_dom, dist_body
from tests.site.tsx_runner import run_tsx_json


def test_unlisted_heading_does_not_clear_active_toc_link(built_site) -> None:
    script = (
        browser_dom(dist_body("docs/getting-started/index.html"))
        + """
const doc = window.document;
await import('./site/src/scripts/docs-runtime.ts');
await new Promise((resolve) => setTimeout(resolve, 0));
const toc = doc.querySelector('[data-toc]');
const link = [...toc.querySelectorAll('.toc-link')].find(
  (item) => doc.getElementById(item.getAttribute('href').slice(1)),
);
const listed = doc.getElementById(link.getAttribute('href').slice(1));
const unlisted = doc.getElementById('setup-output-heading');
if (!listed || !unlisted) throw new Error('Expected docs headings are missing');
const observer = FakeIntersectionObserver.instances.find(
  (item) => item.targets.includes(listed),
);
if (!observer) throw new Error('TOC heading was not observed');
observer.emit(listed, true);
const activeBefore = link.classList.contains('active');
// The viewport can only deliver intersection events for observed headings.
if (observer.targets.includes(unlisted)) observer.emit(unlisted, true);
console.log(JSON.stringify({
  activeBefore,
  activeAfter: link.classList.contains('active'),
  activeCount: toc.querySelectorAll('.toc-link.active').length,
  unlistedObserved: observer.targets.includes(unlisted),
  unlistedPermalink: !!unlisted.querySelector('.heading-link'),
}));
"""
    )
    result = run_tsx_json(script)

    assert result == {
        "activeBefore": True,
        "activeAfter": True,
        "activeCount": 1,
        "unlistedObserved": False,
        "unlistedPermalink": True,
    }
