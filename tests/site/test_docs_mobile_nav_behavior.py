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

const snapshot = () => ({
  expanded: toggle.getAttribute('aria-expanded'),
  label: toggle.getAttribute('aria-label'),
  active: document.activeElement?.name,
  bodyOverflow: document.body.style.overflow || '',
  sidebarHidden: sidebar.getAttribute('aria-hidden'),
  sidebarRole: sidebar.getAttribute('role'),
  sidebarModal: sidebar.getAttribute('aria-modal'),
  sidebarOpen: sidebar.classList.contains('open'),
  scrimOpen: scrim.classList.contains('open'),
  inertTargets: inertTargets.map((target) => target.inert),
  inertAriaHidden: inertTargets.map((target) => target.getAttribute('aria-hidden')),
});

toggle.click();
const afterOpen = snapshot();

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
const afterEscape = snapshot();

toggle.click();
lastLink.focus();
scrim.click();
const afterScrim = snapshot();

toggle.click();
lastLink.focus();
lastLink.click();
const afterLink = snapshot();

console.log(JSON.stringify({
  initiallyHidden,
  afterOpen,
  preventedForward,
  afterForwardTab,
  preventedBackward,
  afterBackwardTab,
  afterEscape,
  afterScrim,
  afterLink,
}));
"""
    rendered = run_tsx_json(script)

    assert rendered["initiallyHidden"] == "true"
    assert rendered["afterOpen"]["expanded"] == "true"
    assert rendered["afterOpen"]["label"] == "Close docs menu"
    assert rendered["afterOpen"]["active"] == "filter"
    assert rendered["afterOpen"]["bodyOverflow"] == "hidden"
    assert rendered["afterOpen"]["sidebarHidden"] is None
    assert rendered["afterOpen"]["sidebarRole"] == "dialog"
    assert rendered["afterOpen"]["sidebarModal"] == "true"
    assert rendered["afterOpen"]["sidebarOpen"] is True
    assert rendered["afterOpen"]["scrimOpen"] is True
    assert rendered["afterOpen"]["inertTargets"] == [True, True, True, True, True]
    assert rendered["afterOpen"]["inertAriaHidden"] == ["true", "true", "true", "true", "true"]
    assert rendered["preventedForward"] is True
    assert rendered["afterForwardTab"] == "filter"
    assert rendered["preventedBackward"] is True
    assert rendered["afterBackwardTab"] == "last-link"
    assert rendered["afterEscape"]["expanded"] == "false"
    assert rendered["afterEscape"]["label"] == "Open docs menu"
    assert rendered["afterEscape"]["active"] == "toggle"
    assert rendered["afterEscape"]["bodyOverflow"] == ""
    assert rendered["afterEscape"]["sidebarHidden"] == "true"
    assert rendered["afterEscape"]["sidebarRole"] == "dialog"
    assert rendered["afterEscape"]["sidebarModal"] == "true"
    assert rendered["afterEscape"]["sidebarOpen"] is False
    assert rendered["afterEscape"]["scrimOpen"] is False
    assert rendered["afterEscape"]["inertTargets"] == [False, False, False, False, False]
    assert rendered["afterEscape"]["inertAriaHidden"] == [None, None, None, None, None]
    assert rendered["afterScrim"]["expanded"] == "false"
    assert rendered["afterScrim"]["active"] == "toggle"
    assert rendered["afterScrim"]["bodyOverflow"] == ""
    assert rendered["afterScrim"]["sidebarHidden"] == "true"
    assert rendered["afterScrim"]["sidebarRole"] == "dialog"
    assert rendered["afterScrim"]["sidebarModal"] == "true"
    assert rendered["afterScrim"]["sidebarOpen"] is False
    assert rendered["afterScrim"]["scrimOpen"] is False
    assert rendered["afterScrim"]["inertTargets"] == [False, False, False, False, False]
    assert rendered["afterScrim"]["inertAriaHidden"] == [None, None, None, None, None]
    assert rendered["afterLink"]["expanded"] == "false"
    assert rendered["afterLink"]["active"] == "last-link"
    assert rendered["afterLink"]["bodyOverflow"] == ""
    assert rendered["afterLink"]["sidebarHidden"] == "true"
    assert rendered["afterLink"]["sidebarRole"] == "dialog"
    assert rendered["afterLink"]["sidebarModal"] == "true"
    assert rendered["afterLink"]["sidebarOpen"] is False
    assert rendered["afterLink"]["scrimOpen"] is False
    assert rendered["afterLink"]["inertTargets"] == [False, False, False, False, False]
    assert rendered["afterLink"]["inertAriaHidden"] == [None, None, None, None, None]


def test_mobile_nav_preserves_preexisting_aria_hidden_targets() -> None:
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
  constructor(owner, attrs = {}) {
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

const toggle = new FakeElement(document);
const scrim = new FakeElement(document);
const filter = new FakeElement(document);
const link = new FakeElement(document, { href: '/docs/getting-started/' });
const sidebar = new FakeSidebar(document, filter, [link]);
const hiddenTarget = new FakeElement(document, { 'aria-hidden': 'true' });
const visibleTarget = new FakeElement(document);

document.toggle = toggle;
document.scrim = scrim;
document.sidebar = sidebar;
document.inertTargets = [hiddenTarget, visibleTarget];

const { initMobileNav } = await import('./site/src/scripts/docs-runtime.ts');
initMobileNav();
toggle.click();
document.dispatch('keydown', {
  key: 'Escape',
  shiftKey: false,
  preventDefault() {},
});

console.log(JSON.stringify({
  hiddenTargetAriaHidden: hiddenTarget.getAttribute('aria-hidden'),
  visibleTargetAriaHidden: visibleTarget.getAttribute('aria-hidden'),
}));
"""
    rendered = run_tsx_json(script)

    assert rendered["hiddenTargetAriaHidden"] == "true"
    assert rendered["visibleTargetAriaHidden"] is None


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
  sidebarRole: sidebar.getAttribute('role'),
  sidebarModal: sidebar.getAttribute('aria-modal'),
  inertTargets: inertTargets.map((target) => target.inert),
  inertAriaHidden: inertTargets.map((target) => target.getAttribute('aria-hidden')),
}));
"""
    rendered = run_tsx_json(script)

    assert rendered["expanded"] == "false"
    assert rendered["bodyOverflow"] == ""
    assert rendered["sidebarOpen"] is False
    assert rendered["sidebarHidden"] is None
    assert rendered["sidebarRole"] is None
    assert rendered["sidebarModal"] is None
    assert rendered["inertTargets"] == [False, False]
    assert rendered["inertAriaHidden"] == [None, None]
