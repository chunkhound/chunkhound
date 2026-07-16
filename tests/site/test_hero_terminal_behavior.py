from __future__ import annotations

import re

from tests.site.tsx_runner import run_tsx_json


NUMERIC_LANGUAGE_CLAIM = re.compile(r"\b\d+\+?\s+languages\b", re.IGNORECASE)


def test_hero_terminal_transcript_renders_stable_commands() -> None:
    script = """
class FakeElement {
  constructor({ id = '', className = '', rect = null, closestMap = new Map(), clientWidth = 320 } = {}) {
    this.id = id;
    this.className = className;
    this.rect = rect || { top: 0, bottom: 200, height: 200 };
    this.closestMap = closestMap;
    this.clientWidth = clientWidth;
    this.children = [];
    this._innerHTML = '';
    this.textContent = '';
    this.style = {};
    this.attributes = new Map();
  }
  appendChild(child) {
    this.children.push(child);
    return child;
  }
  closest(selector) {
    return this.closestMap.get(selector) || null;
  }
  cloneNode() {
    return new FakeMeasureElement();
  }
  getBoundingClientRect() {
    return this.rect;
  }
  remove() {}
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }
  set innerHTML(value) {
    this._innerHTML = value;
    if (value === '') {
      this.children = [];
    }
  }
  get innerHTML() {
    return this._innerHTML;
  }
}

class FakeMeasureElement extends FakeElement {
  constructor() {
    super();
  }
  getBoundingClientRect() {
    return { height: 140, top: 0, bottom: 140 };
  }
}

class FakeBody {
  constructor() {
    this.children = [];
  }
  appendChild(child) {
    this.children.push(child);
    return child;
  }
}

class FakeDocument {
  constructor(container) {
    this.hidden = false;
    this.container = container;
    this.listeners = new Map();
    this.body = new FakeBody();
  }
  getElementById(id) {
    if (id === 'terminal-lines') return this.container;
    return null;
  }
  createElement(tagName) {
    return new FakeElement({ className: tagName });
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
}

class FakeWindow {
  constructor() {
    this.innerHeight = 1000;
    this.listeners = new Map();
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  setTimeout(fn) {
    return setTimeout(fn, 0);
  }
}

class FakeIntersectionObserver {
  constructor(callback) {
    this.callback = callback;
    FakeIntersectionObserver.instance = this;
  }
  observe(target) {
    this.target = target;
  }
}

function textOf(element) {
  return [element.textContent, ...element.children.map(textOf)].join('');
}

const card = new FakeElement({ className: 'terminal-card' });
const container = new FakeElement({ id: 'terminal-lines', closestMap: new Map([['.terminal-card', card]]) });
const document = new FakeDocument(container);
const window = new FakeWindow();

globalThis.window = window;
globalThis.IntersectionObserver = FakeIntersectionObserver;

const { initHeroTerminal } = await import('./site/src/scripts/hero-terminal.ts');
initHeroTerminal(document, {
  charDelay: 0,
  execPause: 0,
  loop: false,
  sleep: async () => {},
});

for (let i = 0; i < 20; i += 1) {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

console.log(JSON.stringify({ transcript: textOf(container) }));
"""
    rendered = run_tsx_json(script)
    transcript = rendered["transcript"]

    assert "chunkhound index ." in transcript
    assert 'chunkhound research "How does authentication work?"' in transcript
    assert 'chunkhound research "Summarize auth changes for reviewers" --commit-range main..HEAD' in transcript
    assert 'chunkhound websearch "OAuth refresh token rotation best practices"' in transcript
    assert "dozens of languages/file types" in transcript
    assert NUMERIC_LANGUAGE_CLAIM.search(transcript) is None


def test_hero_terminal_remeasures_after_font_settling() -> None:
    script = """
class FakeElement {
  constructor({ id = '', className = '', rect = null, closestMap = new Map(), clientWidth = 320 } = {}) {
    this.id = id;
    this.className = className;
    this.rect = rect || { top: 0, bottom: 200, height: 200 };
    this.closestMap = closestMap;
    this.clientWidth = clientWidth;
    this.children = [];
    this._innerHTML = '';
    this.textContent = '';
    this.style = {};
    this.attributes = new Map();
  }
  appendChild(child) {
    this.children.push(child);
    return child;
  }
  closest(selector) {
    return this.closestMap.get(selector) || null;
  }
  cloneNode() {
    return new FakeMeasureElement();
  }
  getBoundingClientRect() {
    return this.rect;
  }
  remove() {}
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }
  set innerHTML(value) {
    this._innerHTML = value;
    if (value === '') {
      this.children = [];
    }
  }
  get innerHTML() {
    return this._innerHTML;
  }
}

class FakeMeasureElement extends FakeElement {
  constructor() {
    super();
  }
  getBoundingClientRect() {
    return { height: globalThis.measureHeight, top: 0, bottom: globalThis.measureHeight };
  }
}

class FakeBody {
  constructor() {
    this.children = [];
  }
  appendChild(child) {
    this.children.push(child);
    return child;
  }
}

class FakeDocument {
  constructor(container, fontsReady) {
    this.hidden = false;
    this.container = container;
    this.listeners = new Map();
    this.body = new FakeBody();
    this.fonts = { ready: fontsReady };
  }
  getElementById(id) {
    if (id === 'terminal-lines') return this.container;
    return null;
  }
  createElement(tagName) {
    return new FakeElement({ className: tagName });
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  dispatchEvent(event) {
    for (const fn of this.listeners.get(event.type) || []) fn(event);
  }
}

class FakeWindow {
  constructor() {
    this.innerHeight = 1000;
    this.listeners = new Map();
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  dispatchEvent(event) {
    for (const fn of this.listeners.get(event.type) || []) fn(event);
  }
  setTimeout(fn) {
    return setTimeout(fn, 0);
  }
}

class FakeIntersectionObserver {
  constructor(callback) {
    this.callback = callback;
    FakeIntersectionObserver.instance = this;
  }
  observe(target) {
    this.target = target;
  }
}

let resolveFonts;
const fontsReady = new Promise((resolve) => {
  resolveFonts = resolve;
});
globalThis.measureHeight = 120;

const card = new FakeElement({ className: 'terminal-card' });
const container = new FakeElement({ id: 'terminal-lines', closestMap: new Map([['.terminal-card', card]]) });
const document = new FakeDocument(container, fontsReady);
const window = new FakeWindow();

globalThis.window = window;
globalThis.IntersectionObserver = FakeIntersectionObserver;

const { initHeroTerminal } = await import('./site/src/scripts/hero-terminal.ts');
initHeroTerminal(document, {
  charDelay: 0,
  execPause: 0,
  loop: false,
  sleep: async () => {},
});

const initialHeight = container.style.height;
globalThis.measureHeight = 180;
resolveFonts();
await fontsReady;
await new Promise((resolve) => setTimeout(resolve, 0));
const afterFontsHeight = container.style.height;

console.log(JSON.stringify({ initialHeight, afterFontsHeight }));
"""
    rendered = run_tsx_json(script)

    assert rendered == {
        "initialHeight": "120px",
        "afterFontsHeight": "180px",
    }


def test_hero_terminal_restarts_after_pagehide_and_pageshow() -> None:
    script = """
class FakeElement {
  constructor({ id = '', className = '', rect = null, closestMap = new Map(), clientWidth = 320 } = {}) {
    this.id = id;
    this.className = className;
    this.rect = rect || { top: 0, bottom: 200, height: 200 };
    this.closestMap = closestMap;
    this.clientWidth = clientWidth;
    this.children = [];
    this._innerHTML = '';
    this.textContent = '';
    this.style = {};
    this.attributes = new Map();
  }
  appendChild(child) {
    this.children.push(child);
    return child;
  }
  closest(selector) {
    return this.closestMap.get(selector) || null;
  }
  cloneNode() {
    return new FakeMeasureElement();
  }
  getBoundingClientRect() {
    return this.rect;
  }
  remove() {}
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }
  set innerHTML(value) {
    this._innerHTML = value;
    if (value === '') {
      this.children = [];
    }
  }
  get innerHTML() {
    return this._innerHTML;
  }
}

class FakeMeasureElement extends FakeElement {
  constructor() {
    super();
  }
  getBoundingClientRect() {
    return { height: 140, top: 0, bottom: 140 };
  }
}

class FakeBody {
  constructor() {
    this.children = [];
  }
  appendChild(child) {
    this.children.push(child);
    return child;
  }
}

class FakeDocument {
  constructor(container) {
    this.hidden = false;
    this.container = container;
    this.listeners = new Map();
    this.body = new FakeBody();
  }
  getElementById(id) {
    if (id === 'terminal-lines') return this.container;
    return null;
  }
  createElement(tagName) {
    return new FakeElement({ className: tagName });
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  dispatchEvent(event) {
    for (const fn of this.listeners.get(event.type) || []) fn(event);
  }
}

class FakeWindow {
  constructor() {
    this.innerHeight = 1000;
    this.listeners = new Map();
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  dispatchEvent(event) {
    for (const fn of this.listeners.get(event.type) || []) fn(event);
  }
  setTimeout(fn) {
    return setTimeout(fn, 0);
  }
}

class FakeIntersectionObserver {
  constructor(callback) {
    this.callback = callback;
    FakeIntersectionObserver.instance = this;
  }
  observe(target) {
    this.target = target;
  }
}

const card = new FakeElement({ className: 'terminal-card' });
const container = new FakeElement({ id: 'terminal-lines', closestMap: new Map([['.terminal-card', card]]) });
const document = new FakeDocument(container);
const window = new FakeWindow();

globalThis.window = window;
globalThis.IntersectionObserver = FakeIntersectionObserver;

const { initHeroTerminal } = await import('./site/src/scripts/hero-terminal.ts');
initHeroTerminal(document, {
  charDelay: 0,
  execPause: 0,
  loop: false,
  sleep: async () => {},
});

FakeIntersectionObserver.instance.callback([{ isIntersecting: true }]);
await new Promise((resolve) => setTimeout(resolve, 0));
const firstRenderCount = container.children.length;

window.dispatchEvent({ type: 'pagehide' });
const afterPagehideCount = container.children.length;

window.dispatchEvent({ type: 'pageshow' });
await new Promise((resolve) => setTimeout(resolve, 0));
await new Promise((resolve) => setTimeout(resolve, 0));
const afterPageshowCount = container.children.length;

console.log(JSON.stringify({
  observedClassName: FakeIntersectionObserver.instance.target.className,
  firstRenderCount,
  afterPagehideCount,
  afterPageshowCount,
}));
"""
    rendered = run_tsx_json(script)

    assert rendered["observedClassName"] == "terminal-card"
    assert rendered["firstRenderCount"] > 0
    assert rendered["afterPagehideCount"] == 0
    assert rendered["afterPageshowCount"] > 0
