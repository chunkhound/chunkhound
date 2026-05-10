// site/scripts/export-deck.mjs
// Usage: node scripts/export-deck.mjs [--theme=light|dark] [--route=/deck] [--out=deck.pdf]
//
// Requires the Astro dev server to be running (npm run dev).
// Checks that the dev server is reachable before launching Puppeteer.

import puppeteer from "puppeteer";

function parseFlag(name, fallback) {
  const prefix = `--${name}=`;
  const hit = process.argv.find((arg) => arg.startsWith(prefix));
  return hit ? hit.slice(prefix.length) : fallback;
}

const theme = parseFlag("theme", "dark");
const route = parseFlag("route", "/deck");
const out = parseFlag("out", "deck.pdf");

const url = new URL(`http://localhost:4321${route}`);
url.searchParams.set("theme", theme);

// Pre-flight: check that the Astro dev server is already running before
// launching a full browser — Puppeteer's timeout for a missing server is
// opaque and slow.
try {
  const probe = await fetch(url.origin);
  if (!probe.ok) throw new Error(`HTTP ${probe.status}`);
} catch (_) {
  console.error(
    `Cannot reach ${url.origin} — is the Astro dev server running?\n` +
    `  Run "npm run dev" in the site/ directory first, then retry.`
  );
  process.exit(1);
}

let browser;
try {
  browser = await puppeteer.launch();
  const page = await browser.newPage();

  await page.emulateMediaType("print");
  await page.goto(url.toString(), { waitUntil: "networkidle0" });
  await page.evaluateHandle("document.fonts.ready");
  await page.waitForSelector("body[data-deck-ready]", { timeout: 10000 });

  await page.pdf({
    path: out,
    printBackground: true,
    width: "254mm",
    height: "143mm",
  });

  console.log(`Exported: ${out} (theme=${theme}, route=${route})`);
} catch (err) {
  console.error(`Failed to export deck PDF: ${err.message}`);
  process.exitCode = 1;
} finally {
  if (browser) await browser.close();
}
