import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve, sep } from 'node:path';
import { chromium } from '../../site/node_modules/playwright/index.mjs';

const dist = resolve(fileURLToPath(new URL('../../site/dist/', import.meta.url)));
const types = { '.css': 'text/css', '.js': 'text/javascript', '.html': 'text/html', '.svg': 'image/svg+xml' };

async function serve(request, response) {
  const pathname = decodeURIComponent(new URL(request.url, 'http://local').pathname);
  const path = resolve(dist, `.${pathname}`);
  if (!path.startsWith(dist + sep) && path !== dist) return response.writeHead(403).end();
  // resolve() drops the trailing slash, so the URL decides whether this is a directory.
  const file = pathname.endsWith('/') || path === dist ? resolve(path, 'index.html') : path;
  try {
    const body = await readFile(file);
    const extension = file.slice(file.lastIndexOf('.'));
    response.writeHead(200, { 'Content-Type': types[extension] || 'application/octet-stream' });
    response.end(body);
  } catch (error) {
    response.writeHead(error.code === 'ENOENT' ? 404 : 500).end();
  }
}

async function withServer(probe) {
  const server = createServer(serve);
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  try {
    return await probe(`http://127.0.0.1:${server.address().port}`);
  } finally {
    await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  }
}

// Opt-in production-font mode: Base.astro is the only font authority, so the
// fixture stylesheet is generated from its request. Inter maps to the brand's
// vendored face; other families map to latin-subset WOFF2 fixtures. External
// requests stay aborted.
const FONT_BASE = 'https://fonts.gstatic.com/';
// Families whose local file is not the default latin WOFF2 subset.
const FONT_FILES = {
  inter: { file: '../../brand/fonts/inter-variable.ttf', format: 'truetype', type: 'font/ttf' },
};
let fontFixtures;

async function fixtureFonts() {
  if (fontFixtures) return fontFixtures;
  const href = (await readFile(new URL('../../site/src/layouts/Base.astro', import.meta.url), 'utf8'))
    .match(/fonts\.googleapis\.com\/css2\?[^"\s]+/)?.[0];
  if (!href) throw new Error('Base.astro no longer links the Google Fonts stylesheet the font fixtures mirror');
  const files = new Map();
  const css = new URL('https://' + href).searchParams.getAll('family').flatMap(entry => {
    const [family, wght] = entry.split(':wght@');
    const slug = family.toLowerCase().replaceAll(' ', '-');
    const local = FONT_FILES[slug] ?? { file: `fixtures/fonts/${slug}-latin.woff2`, format: 'woff2', type: 'font/woff2' };
    const name = `${slug}.${local.format === 'truetype' ? 'ttf' : 'woff2'}`;
    files.set(name, local);
    return (wght ? wght.split(';') : ['400']).map(weight =>
      `@font-face { font-family: '${family}'; font-style: normal; font-weight: ${weight}; font-display: swap; src: url(${FONT_BASE}${name}) format('${local.format}'); }`);
  });
  return (fontFixtures = { css: css.join('\n'), files });
}

async function fixtureFont(url) {
  const fixtures = await fixtureFonts();
  if (url.host === 'fonts.googleapis.com') return { contentType: 'text/css', body: fixtures.css };
  const local = url.host === 'fonts.gstatic.com' && fixtures.files.get(url.pathname.slice(1));
  if (!local) return null; // other gstatic URLs fall through to the abort
  return { contentType: local.type, body: await readFile(new URL(local.file, import.meta.url)) };
}

async function withPage(browser, origin, options, probe) {
  const context = await browser.newContext({
    colorScheme: options.dark ? 'dark' : 'light', reducedMotion: 'reduce', serviceWorkers: 'block',
    ...(options.width ? { viewport: { width: options.width, height: 1000 } } : {}),
  });
  try {
    await context.route('**/*', async route => {
      const url = new URL(route.request().url());
      const fixture = options.fonts && await fixtureFont(url);
      if (fixture) return route.fulfill({ status: 200, headers: { 'access-control-allow-origin': '*' }, ...fixture });
      // Return the action so the route is settled before the callback resolves.
      if (url.origin === origin) return route.continue();
      return route.abort();
    });
    if (options.setting) await context.addInitScript(value => localStorage.setItem('theme', value), options.setting);
    const page = await context.newPage();
    await page.goto(origin + (options.path ?? ''), { waitUntil: 'networkidle' });
    if (options.setting) assert.equal(await page.locator('html').getAttribute('data-theme-setting'), options.setting);
    if (options.width) await page.evaluate(() => document.fonts.ready);
    return await probe(page);
  } finally {
    await context.close();
  }
}

async function probePages(keys, fonts, options, probe) {
  return withServer(async origin => {
    const browser = await chromium.launch();
    try {
      const result = {};
      for (const key of keys) result[key] = await withPage(browser, origin, { ...options(key), fonts }, probe);
      return result;
    } finally {
      await browser.close();
    }
  });
}

// Each setting gets real shipped scripts/styles and private storage, never synthetic CSS.
export const homepageProbe = (settings, probe, fonts = false) =>
  probePages(settings, fonts, setting => ({ dark: true, setting }), probe);

// Layout contracts for any built page: one fresh light-theme page per width, so
// geometry is measured against real shipped CSS and never a mocked rule.
export const pageProbe = (pagePath, widths, probe, fonts = false) =>
  probePages(widths, fonts, width => ({ path: pagePath, width }), probe);

export function paintedColors(element) {
  const style = getComputedStyle(element);
  const hex = color => {
    const channels = color.match(/^rgba?\(([^)]+)\)$/)?.[1].split(',').map(Number);
    if (!channels || (channels.length === 4 && channels[3] !== 1)) throw new Error(`Expected opaque sRGB: ${color}`);
    return '#' + channels.slice(0, 3).map(value => value.toString(16).padStart(2, '0')).join('');
  };
  let surface = element;
  while (surface && getComputedStyle(surface).backgroundColor === 'rgba(0, 0, 0, 0)') surface = surface.parentElement;
  if (!surface) throw new Error('No painted background for CTA');
  if (getComputedStyle(surface).backgroundImage !== 'none') throw new Error('Contrast probe needs a flat surface');
  return { color: hex(style.color), background: hex(getComputedStyle(surface).backgroundColor),
    outline: style.outlineStyle === 'none' ? null : hex(style.outlineColor),
    outlineWidth: parseFloat(style.outlineWidth), outlineOffset: parseFloat(style.outlineOffset),
    focusVisible: element.matches(':focus-visible') };
}

export function visitedDeclarations(element) {
  // Privacy hides visited computed colors; inspect only matching stylesheet rules.
  const declarations = [];
  const walk = rules => {
    for (const rule of rules) {
      if (rule.cssRules) walk(rule.cssRules);
      if (!rule.selectorText?.includes(':visited') || !rule.style.color) continue;
      if (element.matches(rule.selectorText.replaceAll(':visited', ':link'))) declarations.push(rule.style.color);
    }
  };
  for (const sheet of document.styleSheets) {
    // External font sheets are blocked, not part of the local color cascade.
    if (sheet.href && new URL(sheet.href).origin !== location.origin) continue;
    walk(sheet.cssRules);
  }
  return declarations;
}

export async function themeSnapshot(page) {
  const toggle = page.getByRole('button', { name: /^Switch to (dark mode|system theme|light mode)$/ });
  const icons = toggle.locator('ph-sun, ph-moon, ph-monitor');
  const visible = [];
  for (const icon of await icons.all()) {
    if (await icon.isVisible()) visible.push(await icon.evaluate(element => element.localName));
  }
  return visible;
}

export async function ctaSnapshot(page) {
  const cta = page.locator('.hero-setup-link');
  const normal = await ctaState(cta);
  await cta.hover();
  const hover = await ctaState(cta);
  await page.mouse.move(0, 0);
  await keyboardFocus(page, cta);
  const focus = await cta.evaluate(paintedColors);
  return { normal, hover, focus };
}

function resolvedVisitedColors(element, declarations) {
  // Computed custom properties include local overrides and resolve nested var() references.
  const style = getComputedStyle(element);
  const colors = document.createElement('canvas').getContext('2d');
  return declarations.map(declaration => {
    const token = declaration.match(/^var\((--[\w-]+)\)$/)?.[1];
    const value = token ? style.getPropertyValue(token).trim() : '';
    if (token && !CSS.supports('color', value)) throw new Error(`Invalid CTA token ${token}: ${value}`);
    if (token) colors.fillStyle = value;
    return { declaration, color: token ? colors.fillStyle : null };
  });
}

async function ctaState(cta) {
  const declarations = await cta.evaluate(visitedDeclarations);
  const visited = await cta.evaluate(resolvedVisitedColors, declarations);
  return { ...await cta.evaluate(paintedColors), visited };
}

export async function keyboardFocus(page, target) {
  await target.scrollIntoViewIfNeeded();
  for (let step = 0; step < 100; step++) {
    await page.keyboard.press('Tab');
    if (await target.evaluate(element => element === document.activeElement)) return;
  }
  throw new Error('Hero CTA is not reachable with Tab');
}
