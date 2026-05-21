from __future__ import annotations

from tests.site.tsx_runner import run_tsx_json


def test_hero_terminal_restarts_after_pagehide_and_pageshow() -> None:
    script = """
class FakeElement {
  constructor({ id = '', className = '', rect = null, closestMap = new Map() } = {}) {
    this.id = id;
    this.className = className;
    this.rect = rect || { top: 0, bottom: 200, height: 200 };
    this.closestMap = closestMap;
    this.children = [];
    this._innerHTML = '';
    this.textContent = '';
  }
  appendChild(child) {
    this.children.push(child);
    return child;
  }
  closest(selector) {
    return this.closestMap.get(selector) || null;
  }
  getBoundingClientRect() {
    return this.rect;
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

class FakeDocument {
  constructor(container) {
    this.hidden = false;
    this.container = container;
    this.listeners = new Map();
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
