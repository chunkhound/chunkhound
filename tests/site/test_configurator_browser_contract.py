"""Copyable setup cannot silently drop invalid reranker overrides."""

from tests.site.browser_helpers import homepage_probe


def test_invalid_reranker_blocks_copy_until_corrected(built_site):
    result = homepage_probe("""
console.log(JSON.stringify(await homepageProbe(['light'], async page => {
  const section = page.locator('#configurator');
  await section.locator('[data-stage-tab="retrieval"]').click();
  await section.locator('[data-retrieval="ollama-embed"]').check();
  const url = section.locator('[data-rerank-url]');
  const format = section.locator('[data-rerank-format]');
  const copy = section.locator('#config-copy-btn');
  await format.selectOption('tei');
  await url.fill('http:/invalid');
  const invalid = { disabled: await copy.isDisabled(),
    payload: await copy.getAttribute('data-copy') };
  await url.fill('HTTPS://example.com/Rerank?Key=Value');
  const valid = { disabled: await copy.isDisabled(),
    payload: await copy.getAttribute('data-copy') };
  return { invalid, valid };
})));
""")
    assert result["light"]["invalid"] == {"disabled": True, "payload": None}
    assert result["light"]["valid"]["disabled"] is False
    assert "https://example.com/Rerank?Key=Value" in result["light"]["valid"]["payload"]
