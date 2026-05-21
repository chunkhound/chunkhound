from __future__ import annotations

from tests.site.tsx_runner import run_tsx_json


def test_docs_mobile_nav_applies_modal_semantics_only_while_open() -> None:
    script = """
class FakeClassList {
  constructor(initial = []) {
    this.items = new Set(initial);
  }
  add(value) { this.items.add(value); }
  remove(value) { this.items.delete(value); }
  contains(value) { return this.items.has(value); }
  toggle(value, force) {
    if (force === undefined) {
      if (this.items.has(value)) {
        this.items.delete(value);
        return false;
      }
      this.items.add(value);
      return true;
    }
    if (force) this.items.add(value);
    else this.items.delete(value);
    return force;
  }
}

class FakeElement {
  constructor({ id = '', hidden = false } = {}) {
    this.id = id;
    this.hidden = hidden;
    this.attributes = new Map();
    this.classList = new FakeClassList();
    this.listeners = new Map();
    this.focusCount = 0;
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  click() {
    for (const fn of this.listeners.get('click') || []) fn({ target: this });
  }
  focus() { this.focusCount += 1; }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }
  removeAttribute(name) { this.attributes.delete(name); }
  hasAttribute(name) { return this.attributes.has(name); }
}

class FakeDocument {
  constructor(toggle, sidebar, scrim, filter) {
    this.readyState = 'loading';
    this.toggle = toggle;
    this.sidebar = sidebar;
    this.scrim = scrim;
    this.filter = filter;
    this.listeners = new Map();
  }
  querySelector(selector) {
    if (selector === '[data-docs-nav-toggle]') return this.toggle;
    if (selector === '[data-docs-nav-scrim]') return this.scrim;
    if (selector === '[data-docs-nav-filter]') return this.filter;
    return null;
  }
  getElementById(id) {
    if (id === 'docs-sidebar') return this.sidebar;
    return null;
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  dispatchEvent(event) {
    for (const fn of this.listeners.get(event.type) || []) fn(event);
  }
}

const toggle = new FakeElement();
const sidebar = new FakeElement({ id: 'docs-sidebar' });
const scrim = new FakeElement({ hidden: true });
const filter = new FakeElement();
const document = new FakeDocument(toggle, sidebar, scrim, filter);

globalThis.document = document;

const { initMobileNav } = await import('./site/src/scripts/docs-runtime.ts');
initMobileNav(document);

const initial = {
  expanded: toggle.getAttribute('aria-expanded'),
  scrimHidden: scrim.hidden,
  role: sidebar.getAttribute('role'),
  ariaModal: sidebar.getAttribute('aria-modal'),
  tabindex: sidebar.getAttribute('tabindex'),
};

toggle.click();
const opened = {
  expanded: toggle.getAttribute('aria-expanded'),
  scrimHidden: scrim.hidden,
  role: sidebar.getAttribute('role'),
  ariaModal: sidebar.getAttribute('aria-modal'),
  tabindex: sidebar.getAttribute('tabindex'),
  filterFocusCount: filter.focusCount,
  openClass: sidebar.classList.contains('open'),
};

document.dispatchEvent({ type: 'keydown', key: 'Escape' });
const closed = {
  expanded: toggle.getAttribute('aria-expanded'),
  scrimHidden: scrim.hidden,
  role: sidebar.getAttribute('role'),
  ariaModal: sidebar.getAttribute('aria-modal'),
  tabindex: sidebar.getAttribute('tabindex'),
  toggleFocusCount: toggle.focusCount,
  openClass: sidebar.classList.contains('open'),
};

console.log(JSON.stringify({ initial, opened, closed }));
"""
    rendered = run_tsx_json(script)

    assert rendered["initial"] == {
        "expanded": "false",
        "scrimHidden": True,
        "role": None,
        "ariaModal": None,
        "tabindex": None,
    }
    assert rendered["opened"]["expanded"] == "true"
    assert rendered["opened"]["scrimHidden"] is False
    assert rendered["opened"]["role"] == "dialog"
    assert rendered["opened"]["ariaModal"] == "true"
    assert rendered["opened"]["tabindex"] == "-1"
    assert rendered["opened"]["filterFocusCount"] == 1
    assert rendered["opened"]["openClass"] is True
    assert rendered["closed"] == {
        "expanded": "false",
        "scrimHidden": True,
        "role": None,
        "ariaModal": None,
        "tabindex": None,
        "toggleFocusCount": 1,
        "openClass": False,
    }
