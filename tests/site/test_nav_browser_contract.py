"""Rendered contracts for navigation destinations and fallback-copy focus."""

from tests.site.browser_helpers import homepage_probe


def test_legacy_fragments_land_on_useful_visible_content(built_site):
    result = homepage_probe("""
console.log(JSON.stringify(await homepageProbe(['light'], async page => {
  const band = page.viewportSize().height;
  const destinations = {};
  for (const id of ['install', 'capabilities', 'approach', 'use-cases']) {
    await page.goto(`${new URL(page.url()).origin}/#${id}`);
    destinations[id] = await page.locator(`[id="${id}"]`).evaluate(element => {
      const content = element.closest('section');
      const rect = element.getBoundingClientRect();
      return { section: content.id, text: content.textContent.trim().length,
        top: rect.top, hidden: element.closest('[hidden], .sr-only') !== null };
    });
  }
  return { band, destinations };
})));
""")
    # A fragment must scroll its target into the live viewport band, not a
    # hardcoded pixel height (the default viewport is 720 but may change).
    viewport_band = result["light"]["band"]
    for fragment, section in {
        "install": "configurator",
        "capabilities": "research",
        "approach": "research",
        "use-cases": "questions",
    }.items():
        destination = result["light"]["destinations"][fragment]
        assert destination["section"] == section
        assert destination["text"] > 100
        assert destination["hidden"] is False
        assert 0 <= destination["top"] < viewport_band


def test_fallback_copy_restores_keyboard_focus_in_chromium(built_site):
    result = homepage_probe("""
console.log(JSON.stringify(await homepageProbe(['light'], async page => {
  // Clipboard availability is a browser boundary; execCommand remains real.
  await page.evaluate(() => Object.defineProperty(navigator, 'clipboard', {
    value: undefined,
  }));
  const button = page.locator('#install .copy-btn');
  await button.focus();
  await page.keyboard.press('Enter');
  await page.waitForFunction(() =>
    document.querySelector('#install [data-copy-status]').textContent !== '');
  return await button.evaluate(element => ({
    focused: element === document.activeElement,
    status: element.querySelector('[data-copy-status]').textContent,
    scratch: document.querySelectorAll('textarea').length,
  }));
})));
""")
    assert result["light"] == {"focused": True, "status": "Copied", "scratch": 0}
