from __future__ import annotations

from tests.site.tsx_runner import run_tsx_json


def test_mobile_nav_traps_focus_and_restores_modal_state() -> None:
    script = """
class FakeClassList {
  constructor(initial = []) {
    this.items = new Set(initial);
  }
  add(value) { this.items.add(value); }
  remove(value) { this.items.delete(value); }
  contains(value) { return this.items.has(value); }
}

class FakeElement {
  constructor(name, owner, attrs = {}) {
    this.name = name;
    this.ownerDocument = owner;
    this.attributes = new Map(Object.entries(attrs));
    this.classList = new FakeClassList();
    this.listeners = new Map();
    this.inert = false;
    this.style = {};
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  dispatchEvent(event) {
    for (const fn of this.listeners.get(event.type) || []) fn(event);
  }
  click() {
    this.dispatchEvent({ type: 'click' });
  }
  focus() {
    this.ownerDocument.activeElement = this;
  }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }
  removeAttribute(name) { this.attributes.delete(name); }
  hasAttribute(name) { return this.attributes.has(name); }
}

class FakeSidebar extends FakeElement {
  constructor(owner, input, links) {
    super('sidebar', owner);
    this.input = input;
    this.links = links;
  }
  querySelector(selector) {
    if (selector === 'input, a, button') return this.input;
    return null;
  }
  querySelectorAll(selector) {
    if (selector === 'a') return this.links;
    if (selector.includes('a[href]')) return [this.input, ...this.links];
    return [];
  }
}

class FakeMediaQuery {
  constructor(matches) {
    this.matches = matches;
    this.listeners = [];
  }
  addEventListener(type, fn) {
    if (type === 'change') this.listeners.push(fn);
  }
  setMatches(matches) {
    this.matches = matches;
    for (const fn of this.listeners) fn({ matches });
  }
}

class FakeDocument {
  constructor() {
    this.readyState = 'loading';
    this.listeners = new Map();
    this.activeElement = null;
    this.body = { style: {} };
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  dispatch(type, event) {
    for (const fn of this.listeners.get(type) || []) fn(event);
  }
  querySelector(selector) {
    if (selector === '[data-docs-nav-toggle]') return this.toggle;
    if (selector === '[data-docs-nav-scrim]') return this.scrim;
    return null;
  }
  querySelectorAll(selector) {
    if (selector === '[data-docs-mobile-inert]') return this.inertTargets;
    return [];
  }
  getElementById(id) {
    if (id === 'docs-sidebar') return this.sidebar;
    return null;
  }
}

const document = new FakeDocument();
const mediaQuery = new FakeMediaQuery(true);
const window = { matchMedia: () => mediaQuery };

globalThis.document = document;
globalThis.window = window;

const toggle = new FakeElement('toggle', document);
const scrim = new FakeElement('scrim', document);
const filter = new FakeElement('filter', document);
const firstLink = new FakeElement('first-link', document, { href: '/docs/getting-started/' });
const lastLink = new FakeElement('last-link', document, { href: '/docs/configuration/' });
const sidebar = new FakeSidebar(document, filter, [firstLink, lastLink]);
const inertTargets = [
  new FakeElement('wordmark', document),
  new FakeElement('tabs', document),
  new FakeElement('actions', document),
  new FakeElement('main', document),
  new FakeElement('toc', document),
];

document.toggle = toggle;
document.scrim = scrim;
document.sidebar = sidebar;
document.inertTargets = inertTargets;

const { initMobileNav } = await import('./site/src/scripts/docs-runtime.ts');
initMobileNav();
const initiallyHidden = sidebar.getAttribute('aria-hidden');

toggle.click();
const afterOpen = {
  expanded: toggle.getAttribute('aria-expanded'),
  label: toggle.getAttribute('aria-label'),
  active: document.activeElement?.name,
  bodyOverflow: document.body.style.overflow || '',
  sidebarHidden: sidebar.getAttribute('aria-hidden'),
  inertTargets: inertTargets.map((target) => target.inert),
};

lastLink.focus();
let preventedForward = false;
document.dispatch('keydown', {
  key: 'Tab',
  shiftKey: false,
  preventDefault() { preventedForward = true; },
});
const afterForwardTab = document.activeElement?.name;

filter.focus();
let preventedBackward = false;
document.dispatch('keydown', {
  key: 'Tab',
  shiftKey: true,
  preventDefault() { preventedBackward = true; },
});
const afterBackwardTab = document.activeElement?.name;

document.dispatch('keydown', {
  key: 'Escape',
  shiftKey: false,
  preventDefault() {},
});

const afterEscape = {
  expanded: toggle.getAttribute('aria-expanded'),
  label: toggle.getAttribute('aria-label'),
  active: document.activeElement?.name,
  bodyOverflow: document.body.style.overflow || '',
  sidebarHidden: sidebar.getAttribute('aria-hidden'),
  inertTargets: inertTargets.map((target) => target.inert),
};

console.log(JSON.stringify({
  initiallyHidden,
  afterOpen,
  preventedForward,
  afterForwardTab,
  preventedBackward,
  afterBackwardTab,
  afterEscape,
}));
"""
    rendered = run_tsx_json(script)

    assert rendered["initiallyHidden"] == "true"
    assert rendered["afterOpen"]["expanded"] == "true"
    assert rendered["afterOpen"]["label"] == "Close docs menu"
    assert rendered["afterOpen"]["active"] == "filter"
    assert rendered["afterOpen"]["bodyOverflow"] == "hidden"
    assert rendered["afterOpen"]["sidebarHidden"] is None
    assert rendered["afterOpen"]["inertTargets"] == [True, True, True, True, True]
    assert rendered["preventedForward"] is True
    assert rendered["afterForwardTab"] == "filter"
    assert rendered["preventedBackward"] is True
    assert rendered["afterBackwardTab"] == "last-link"
    assert rendered["afterEscape"]["expanded"] == "false"
    assert rendered["afterEscape"]["label"] == "Open docs menu"
    assert rendered["afterEscape"]["active"] == "toggle"
    assert rendered["afterEscape"]["bodyOverflow"] == ""
    assert rendered["afterEscape"]["sidebarHidden"] == "true"
    assert rendered["afterEscape"]["inertTargets"] == [False, False, False, False, False]


def test_mobile_nav_cleans_up_when_viewport_expands_to_desktop() -> None:
    script = """
class FakeClassList {
  constructor(initial = []) {
    this.items = new Set(initial);
  }
  add(value) { this.items.add(value); }
  remove(value) { this.items.delete(value); }
  contains(value) { return this.items.has(value); }
}

class FakeElement {
  constructor(owner) {
    this.ownerDocument = owner;
    this.attributes = new Map();
    this.classList = new FakeClassList();
    this.listeners = new Map();
    this.inert = false;
    this.style = {};
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  dispatchEvent(event) {
    for (const fn of this.listeners.get(event.type) || []) fn(event);
  }
  click() {
    this.dispatchEvent({ type: 'click' });
  }
  focus() {
    this.ownerDocument.activeElement = this;
  }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }
  removeAttribute(name) { this.attributes.delete(name); }
  hasAttribute(name) { return this.attributes.has(name); }
}

class FakeSidebar extends FakeElement {
  constructor(owner, input, links) {
    super(owner);
    this.input = input;
    this.links = links;
  }
  querySelectorAll(selector) {
    if (selector === 'a') return this.links;
    if (selector.includes('a[href]')) return [this.input, ...this.links];
    return [];
  }
}

class FakeMediaQuery {
  constructor(matches) {
    this.matches = matches;
    this.listeners = [];
  }
  addEventListener(type, fn) {
    if (type === 'change') this.listeners.push(fn);
  }
  setMatches(matches) {
    this.matches = matches;
    for (const fn of this.listeners) fn({ matches });
  }
}

class FakeDocument {
  constructor() {
    this.readyState = 'loading';
    this.listeners = new Map();
    this.activeElement = null;
    this.body = { style: {} };
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  querySelector(selector) {
    if (selector === '[data-docs-nav-toggle]') return this.toggle;
    if (selector === '[data-docs-nav-scrim]') return this.scrim;
    return null;
  }
  querySelectorAll(selector) {
    if (selector === '[data-docs-mobile-inert]') return this.inertTargets;
    return [];
  }
  getElementById(id) {
    if (id === 'docs-sidebar') return this.sidebar;
    return null;
  }
}

const document = new FakeDocument();
const mediaQuery = new FakeMediaQuery(true);
const window = { matchMedia: () => mediaQuery };

globalThis.document = document;
globalThis.window = window;

const toggle = new FakeElement(document);
const scrim = new FakeElement(document);
const filter = new FakeElement(document);
const link = new FakeElement(document);
const sidebar = new FakeSidebar(document, filter, [link]);
const inertTargets = [new FakeElement(document), new FakeElement(document)];

document.toggle = toggle;
document.scrim = scrim;
document.sidebar = sidebar;
document.inertTargets = inertTargets;

const { initMobileNav } = await import('./site/src/scripts/docs-runtime.ts');
initMobileNav();
toggle.click();
mediaQuery.setMatches(false);

console.log(JSON.stringify({
  expanded: toggle.getAttribute('aria-expanded'),
  bodyOverflow: document.body.style.overflow || '',
  sidebarOpen: sidebar.classList.contains('open'),
  sidebarHidden: sidebar.getAttribute('aria-hidden'),
  inertTargets: inertTargets.map((target) => target.inert),
}));
"""
    rendered = run_tsx_json(script)

    assert rendered["expanded"] == "false"
    assert rendered["bodyOverflow"] == ""
    assert rendered["sidebarOpen"] is False
    assert rendered["sidebarHidden"] is None
    assert rendered["inertTargets"] == [False, False]
