import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve, sep } from 'node:path';
import { chromium } from '../../site/node_modules/playwright/index.mjs';

const dist = resolve(fileURLToPath(new URL('../../site/dist/', import.meta.url)));
const types = { '.css': 'text/css', '.js': 'text/javascript', '.html': 'text/html', '.svg': 'image/svg+xml' };

async function serve(request, response) {
  const path = resolve(dist, `.${decodeURIComponent(new URL(request.url, 'http://local').pathname)}`);
  if (!path.startsWith(dist + sep) && path !== dist) return response.writeHead(403).end();
  const file = path.endsWith(sep) || path === dist ? resolve(path, 'index.html') : path;
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

async function withContext(browser, origin, setting, probe) {
  const context = await browser.newContext({ colorScheme: 'dark', reducedMotion: 'reduce', serviceWorkers: 'block' });
  try {
    await context.route('**/*', route => new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
    await context.addInitScript(value => localStorage.setItem('theme', value), setting);
    const page = await context.newPage();
    await page.goto(origin, { waitUntil: 'networkidle' });
    assert.equal(await page.locator('html').getAttribute('data-theme-setting'), setting);
    return await probe(page);
  } finally {
    await context.close();
  }
}

// Each setting gets real shipped scripts/styles and private storage, never synthetic CSS.
export async function homepageProbe(settings, probe) {
  return withServer(async origin => {
    const browser = await chromium.launch();
    try {
      const result = {};
      for (const setting of settings) result[setting] = await withContext(browser, origin, setting, probe);
      return result;
    } finally {
      await browser.close();
    }
  });
}

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
