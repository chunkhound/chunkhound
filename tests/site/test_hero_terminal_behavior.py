from __future__ import annotations

from tests.site.tsx_runner import run_tsx_json


_FAKE_TIMERS = r"""
class FakeTimers {
  static queue = [];

  static install() {
    globalThis.setTimeout = (callback, _ms) => {
      FakeTimers.queue.push(callback);
      return FakeTimers.queue.length;
    };
    globalThis.clearTimeout = () => {};
  }

  static async flushUntil(predicate, limit = 2000) {
    for (let index = 0; index < limit; index += 1) {
      await Promise.resolve();
      if (predicate()) return true;
      const callback = FakeTimers.queue.shift();
      if (!callback) continue;
      callback();
      await Promise.resolve();
      if (predicate()) return true;
    }
    return predicate();
  }
}
"""

_FAKE_ELEMENT = r"""
class FakeElement {
  constructor(name, owner) {
    this.name = name;
    this.ownerDocument = owner;
    this.children = [];
    this.parentElement = null;
    this.attributes = new Map();
    this.style = {};
    this.className = '';
    this.clientWidth = 0;
    this.listeners = new Map();
    this._textContent = '';
    this.classList = {
      items: new Set(),
      add: (value) => { this.classList.items.add(value); },
      remove: (value) => { this.classList.items.delete(value); },
      contains: (value) => this.classList.items.has(value),
    };
  }
  get textContent() {
    return this._textContent;
  }
  set textContent(value) {
    this._textContent = String(value);
  }
  get innerHTML() {
    return this.children.length ? '[rendered]' : this._textContent;
  }
  set innerHTML(value) {
    this._textContent = String(value);
    this.children = [];
  }
  appendChild(child) {
    child.parentElement = this;
    this.children.push(child);
    return child;
  }
  cloneNode(_deep) {
    const clone = new FakeElement(`${this.name}-clone`, this.ownerDocument);
    clone.clientWidth = this.clientWidth;
    return clone;
  }
  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }
  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  removeEventListener(type, fn) {
    if (!this.listeners.has(type)) return;
    this.listeners.set(type, this.listeners.get(type).filter((listener) => listener !== fn));
  }
  dispatchEvent(event) {
    for (const fn of this.listeners.get(event.type) || []) fn(event);
  }
  getBoundingClientRect() {
    return { height: this.ownerDocument.measureHeights.shift() ?? 0 };
  }
  remove() {
    if (!this.parentElement) return;
    this.parentElement.children = this.parentElement.children.filter((child) => child !== this);
    this.parentElement = null;
  }
  closest(selector) {
    const cls = selector.replace('.', '');
    let el = this;
    while (el) {
      if (el.name === cls || el.className === cls || el.classList.contains(cls)) return el;
      el = el.parentElement;
    }
    return null;
  }
}
"""

_FAKE_RESIZE_OBSERVER = r"""
class FakeResizeObserver {
  static instances = [];

  constructor(callback) {
    this.callback = callback;
    this.observed = [];
    this.disconnected = false;
    FakeResizeObserver.instances.push(this);
  }
  observe(target) {
    this.observed.push(target);
  }
  disconnect() {
    this.disconnected = true;
  }
}
"""

_FAKE_INTERSECTION_OBSERVER = r"""
class FakeIntersectionObserver {
  static instances = [];
  static lastCallback = null;
  static lastOptions = null;
  static lastObserved = null;

  constructor(callback, options) {
    this.callback = callback;
    this.options = options;
    this.observed = [];
    this.disconnected = false;
    FakeIntersectionObserver.instances.push(this);
    FakeIntersectionObserver.lastCallback = callback;
    FakeIntersectionObserver.lastOptions = options;
  }
  observe(target) {
    this.observed.push(target);
    FakeIntersectionObserver.lastObserved = target;
  }
  disconnect() {
    this.disconnected = true;
  }
}
"""

_FAKE_DOCUMENT = r"""
class FakeDocument {
  constructor() {
    this.measureHeights = [88, 132];
    this.body = new FakeElement('body', this);
    this.hidden = false;
    this.listeners = new Map();
  }
  createElement(tagName) {
    return new FakeElement(tagName, this);
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  removeEventListener(type, fn) {
    if (!this.listeners.has(type)) return;
    this.listeners.set(type, this.listeners.get(type).filter((listener) => listener !== fn));
  }
  dispatch(type) {
    for (const fn of this.listeners.get(type) || []) fn({ type });
  }
}
"""

_SERIALIZERS = r"""
const flattenText = (node) => `${node.textContent}${node.children.map(flattenText).join('')}`;
const hasClass = (node, className) => node.className === className || node.children.some((child) => hasClass(child, className));
const serializeTerminal = (container) => container.children.map((line) => ({
  text: flattenText(line),
  classes: line.children.map((child) => child.className),
  hasCursor: line.children.some((child) => hasClass(child, 'cursor')),
}));
"""


def _render(script_body: str) -> dict:
    script = (
        "import { observeTerminalHeight, lockTerminalHeight, initHeroTerminal } "
        "from './site/src/scripts/hero-terminal.ts';\n\n"
        f"{_FAKE_TIMERS}\n"
        f"{_FAKE_ELEMENT}\n"
        f"{_FAKE_RESIZE_OBSERVER}\n"
        f"{_FAKE_INTERSECTION_OBSERVER}\n"
        f"{_FAKE_DOCUMENT}\n"
        f"{_SERIALIZERS}\n"
        f"{script_body}\n"
    )
    return run_tsx_json(script)


def test_hero_terminal_height_lock_remeasures_on_resize() -> None:
    rendered = _render(r"""
const document = new FakeDocument();
document.measureHeights = [88, 132];
const container = new FakeElement('terminal-lines', document);
container.clientWidth = 420;
const card = new FakeElement('terminal-card', document);

globalThis.document = document;
globalThis.ResizeObserver = FakeResizeObserver;

observeTerminalHeight(container, card);
const initial = {
  height: container.style.height || '',
  measuredNodes: document.body.children.length,
  observed: FakeResizeObserver.instances[0]?.observed.map((target) => target.name) || [],
};

FakeResizeObserver.instances[0].callback([]);
const afterResize = {
  height: container.style.height || '',
  measuredNodes: document.body.children.length,
};

console.log(JSON.stringify({ initial, afterResize }));
""")
    assert rendered["initial"]["height"] == "88px"
    assert rendered["initial"]["measuredNodes"] == 0
    assert rendered["initial"]["observed"] == ["terminal-card"]
    assert rendered["afterResize"]["height"] == "132px"
    assert rendered["afterResize"]["measuredNodes"] == 0


def test_hero_terminal_lock_skips_on_zero_width() -> None:
    rendered = _render(r"""
const document = new FakeDocument();
document.measureHeights = [];
const container = new FakeElement('container', document);
container.clientWidth = 0;

globalThis.document = document;

lockTerminalHeight(container);

console.log(JSON.stringify({ height: container.style.height || '' }));
""")
    assert rendered["height"] == ""


def test_hero_terminal_lock_returns_null_without_resize_observer() -> None:
    rendered = _render(r"""
const document = new FakeDocument();
document.measureHeights = [100];
const container = new FakeElement('container', document);
container.clientWidth = 420;

globalThis.document = document;
globalThis.ResizeObserver = undefined;

const result = observeTerminalHeight(container);

console.log(JSON.stringify({ returned: result === null }));
""")
    assert rendered["returned"] is True


def test_hero_terminal_renders_terminal_contract_and_tears_down() -> None:
    rendered = _render(r"""
FakeTimers.install();
const document = new FakeDocument();
document.measureHeights = [120];
const card = new FakeElement('terminal-card', document);
card.className = 'terminal-card';
const container = new FakeElement('terminal-lines', document);
container.clientWidth = 420;
card.appendChild(container);

globalThis.document = document;
globalThis.ResizeObserver = FakeResizeObserver;
globalThis.IntersectionObserver = FakeIntersectionObserver;

const teardown = initHeroTerminal(container);
FakeIntersectionObserver.lastCallback([{ isIntersecting: true }]);
await FakeTimers.flushUntil(() => serializeTerminal(container).some((line) => line.hasCursor));
const beforeTeardown = serializeTerminal(container);
teardown();

console.log(JSON.stringify({
  beforeTeardown,
  afterTeardownChildren: container.children.length,
  ioThreshold: FakeIntersectionObserver.lastOptions.threshold,
  ioObserved: FakeIntersectionObserver.lastObserved?.name || '',
  roObserved: FakeResizeObserver.instances[0]?.observed.map((target) => target.name) || [],
  ioDisconnected: FakeIntersectionObserver.instances[0]?.disconnected || false,
  roDisconnected: FakeResizeObserver.instances[0]?.disconnected || false,
}));
""")
    assert rendered["ioThreshold"] == 0.5
    assert rendered["ioObserved"] == "terminal-card"
    assert rendered["roObserved"] == ["terminal-card"]
    assert rendered["ioDisconnected"] is True
    assert rendered["roDisconnected"] is True
    assert rendered["afterTeardownChildren"] == 0
    assert [line["text"] for line in rendered["beforeTeardown"]] == [
        "$ chunkhound index .",
        "Indexed 12,847 files · 32 languages · 2.1M LOC",
        "\u00a0",
        '$ chunkhound search "how does authentication work"',
        "Found 42 results via 3-hop semantic traversal",
        "\u00a0",
        '$ chunkhound research "authentication architecture"',
        "Generated report · 14 files · 6 components · 23 citations",
    ]
    assert rendered["beforeTeardown"][4]["classes"] == ["output dim"]
    assert rendered["beforeTeardown"][7]["hasCursor"] is True


def test_hero_terminal_waits_for_visible_tab_and_clears_when_scrolled_away() -> None:
    rendered = _render(r"""
FakeTimers.install();
const document = new FakeDocument();
document.measureHeights = [120];
document.hidden = true;
const container = new FakeElement('terminal-lines', document);
container.clientWidth = 420;

globalThis.document = document;
globalThis.ResizeObserver = FakeResizeObserver;
globalThis.IntersectionObserver = FakeIntersectionObserver;

const teardown = initHeroTerminal(container);
FakeIntersectionObserver.lastCallback([{ isIntersecting: true }]);
await FakeTimers.flushUntil(() => FakeTimers.queue.length === 0, 10);
const whileHidden = serializeTerminal(container);

document.hidden = false;
document.dispatch('visibilitychange');
await FakeTimers.flushUntil(() => serializeTerminal(container).some((line) => line.hasCursor));
const afterResume = serializeTerminal(container);

FakeIntersectionObserver.lastCallback([{ isIntersecting: false }]);
const afterScrollAway = serializeTerminal(container);
teardown();

console.log(JSON.stringify({ whileHidden, afterResume, afterScrollAway }));
""")
    assert rendered["whileHidden"] == []
    assert rendered["afterResume"][0]["text"] == "$ chunkhound index ."
    assert rendered["afterResume"][-1]["hasCursor"] is True
    assert rendered["afterScrollAway"] == []
