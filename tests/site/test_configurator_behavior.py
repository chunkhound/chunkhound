from __future__ import annotations

from tests.site.tsx_runner import run_tsx_json


def test_configurator_script_updates_selection_callout_and_copy_contract() -> None:
    script = """
import { initConfigurator } from './site/src/scripts/configurator.ts';

class FakeClassList {
  constructor(initial = []) {
    this.items = new Set(initial);
  }
  add(value) { this.items.add(value); }
  remove(value) { this.items.delete(value); }
  contains(value) { return this.items.has(value); }
}

class FakeElement {
  constructor({ id = '', dataset = {}, attrs = {}, selected = false } = {}) {
    this.id = id;
    this.dataset = { ...dataset };
    this.attributes = new Map(Object.entries(attrs));
    this.classList = new FakeClassList(selected ? ['selected'] : []);
    this.listeners = new Map();
    this.innerHTML = '';
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  click() {
    for (const fn of this.listeners.get('click') || []) fn();
  }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }
  removeAttribute(name) { this.attributes.delete(name); }
  hasAttribute(name) { return this.attributes.has(name); }
}

class FakeCallout extends FakeElement {
  constructor(copyBtn, codeEl) {
    super({ id: 'rerank-callout', attrs: { hidden: '' } });
    this.copyBtn = copyBtn;
    this.codeEl = codeEl;
  }
  querySelector(selector) {
    if (selector === '#rerank-copy-btn') return this.copyBtn;
    if (selector === 'code') return this.codeEl;
    return null;
  }
}

class FakeSection extends FakeElement {
  constructor() {
    super({ attrs: { 'data-mode': 'full' } });
    this.output = new FakeElement({ id: 'config-output' });
    this.copyBtn = new FakeElement({ id: 'config-copy-btn' });
    this.rerankCopyBtn = new FakeElement({ id: 'rerank-copy-btn' });
    this.calloutCode = new FakeElement();
    this.callout = new FakeCallout(this.rerankCopyBtn, this.calloutCode);

    this.editors = [
      new FakeElement({ dataset: { editor: 'cursor' }, selected: true }),
      new FakeElement({ dataset: { editor: 'codex' } }),
    ];
    this.embeddings = [
      new FakeElement({ dataset: { embedding: 'voyageai' }, selected: true }),
      new FakeElement({ dataset: { embedding: 'ollama-embed' } }),
    ];
    this.llms = [
      new FakeElement({ dataset: { llm: 'anthropic' }, selected: true }),
      new FakeElement({ dataset: { llm: 'codex-cli' } }),
      new FakeElement({ dataset: { llm: 'ollama-llm' } }),
    ];
  }

  querySelector(selector) {
    if (selector === '#config-output') return this.output;
    if (selector === '#config-copy-btn') return this.copyBtn;
    if (selector === '#rerank-callout') return this.callout;
    if (selector === '[data-embedding].selected') {
      return this.embeddings.find((el) => el.classList.contains('selected')) || null;
    }
    if (selector === '[data-llm].selected') {
      return this.llms.find((el) => el.classList.contains('selected')) || null;
    }
    if (selector === '[data-editor].selected') {
      return this.editors.find((el) => el.classList.contains('selected')) || null;
    }
    const llmMatch = selector.match(/^\\[data-llm='(.+)'\\]$/);
    if (llmMatch) return this.llms.find((el) => el.dataset.llm === llmMatch[1]) || null;
    return null;
  }

  querySelectorAll(selector) {
    if (selector === '[data-editor]') return this.editors;
    if (selector === '[data-embedding]') return this.embeddings;
    if (selector === '[data-llm]') return this.llms;
    return [];
  }
}

const section = new FakeSection();
initConfigurator(section);

const initialHidden = section.callout.hasAttribute('hidden');

section.embeddings.find((el) => el.dataset.embedding === 'ollama-embed').click();
const afterEmbed = {
  calloutHidden: section.callout.hasAttribute('hidden'),
  calloutCopy: section.rerankCopyBtn.getAttribute('data-copy'),
  configCopy: section.copyBtn.getAttribute('data-copy'),
  embedSelected: section.embeddings
    .find((el) => el.dataset.embedding === 'ollama-embed')
    .getAttribute('aria-checked'),
  voyageSelected: section.embeddings
    .find((el) => el.dataset.embedding === 'voyageai')
    .getAttribute('aria-checked'),
};

section.editors.find((el) => el.dataset.editor === 'codex').click();
const afterEditor = {
  llmSelectedByEditor: section.llms
    .find((el) => el.dataset.llm === 'codex-cli')
    .getAttribute('aria-checked'),
  configCopy: section.copyBtn.getAttribute('data-copy'),
};

section.embeddings.find((el) => el.dataset.embedding === 'voyageai').click();
const afterReset = {
  calloutHidden: section.callout.hasAttribute('hidden'),
  calloutCopy: section.rerankCopyBtn.getAttribute('data-copy'),
};

console.log(JSON.stringify({ initialHidden, afterEmbed, afterEditor, afterReset }));
"""
    rendered = run_tsx_json(script)

    assert rendered["initialHidden"] is True
    assert rendered["afterEmbed"]["calloutHidden"] is False
    assert "ollama pull qwen3-embedding" in rendered["afterEmbed"]["calloutCopy"]
    assert "qwen3-embedding" in rendered["afterEmbed"]["configCopy"]
    assert rendered["afterEmbed"]["embedSelected"] == "true"
    assert rendered["afterEmbed"]["voyageSelected"] == "false"
    assert rendered["afterEditor"]["llmSelectedByEditor"] == "true"
    assert "codex-cli" in rendered["afterEditor"]["configCopy"]
    assert rendered["afterReset"]["calloutHidden"] is True
    assert rendered["afterReset"]["calloutCopy"] is None


def test_configurator_platform_selection_persists_and_updates_code_blocks() -> None:
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
  constructor({ id = '', dataset = {}, attrs = {}, selected = false } = {}) {
    this.id = id;
    this.dataset = { ...dataset };
    this.attributes = new Map(Object.entries(attrs));
    this.classList = new FakeClassList(selected ? ['selected'] : []);
    this.listeners = new Map();
    this.innerHTML = '';
    this.hidden = false;
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  click() {
    for (const fn of this.listeners.get('click') || []) fn();
  }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }
  removeAttribute(name) { this.attributes.delete(name); }
  hasAttribute(name) { return this.attributes.has(name); }
}

class FakeCallout extends FakeElement {
  constructor(copyBtn, codeEl) {
    super({ id: 'rerank-callout', attrs: { hidden: '' } });
    this.copyBtn = copyBtn;
    this.codeEl = codeEl;
  }
  querySelector(selector) {
    if (selector === '#rerank-copy-btn') return this.copyBtn;
    if (selector === 'code') return this.codeEl;
    return null;
  }
}

class FakeSection extends FakeElement {
  constructor(platformButtons) {
    super({ attrs: { 'data-mode': 'full' } });
    this.output = new FakeElement({ id: 'config-output' });
    this.copyBtn = new FakeElement({ id: 'config-copy-btn' });
    this.rerankCopyBtn = new FakeElement({ id: 'rerank-copy-btn' });
    this.calloutCode = new FakeElement();
    this.callout = new FakeCallout(this.rerankCopyBtn, this.calloutCode);
    this.platformButtons = platformButtons;

    this.editors = [
      new FakeElement({ dataset: { editor: 'cursor' }, selected: true }),
    ];
    this.embeddings = [
      new FakeElement({ dataset: { embedding: 'voyageai' }, selected: true }),
    ];
    this.llms = [
      new FakeElement({ dataset: { llm: 'anthropic' }, selected: true }),
    ];
  }

  querySelector(selector) {
    if (selector === '#config-output') return this.output;
    if (selector === '#config-copy-btn') return this.copyBtn;
    if (selector === '#rerank-callout') return this.callout;
    if (selector === '[data-embedding].selected') {
      return this.embeddings.find((el) => el.classList.contains('selected')) || null;
    }
    if (selector === '[data-llm].selected') {
      return this.llms.find((el) => el.classList.contains('selected')) || null;
    }
    if (selector === '[data-editor].selected') {
      return this.editors.find((el) => el.classList.contains('selected')) || null;
    }
    return null;
  }

  querySelectorAll(selector) {
    if (selector === '[data-editor]') return this.editors;
    if (selector === '[data-embedding]') return this.embeddings;
    if (selector === '[data-llm]') return this.llms;
    if (selector === '[data-platform-option]') return this.platformButtons;
    return [];
  }
}

class FakeDocument {
  constructor(section, platformButtons, codeBlocks) {
    this.section = section;
    this.platformButtons = platformButtons;
    this.codeBlocks = codeBlocks;
    this.listeners = new Map();
  }
  querySelectorAll(selector) {
    if (selector === '.configurator') return [this.section];
    if (selector === '[data-platform-option]') return this.platformButtons;
    if (selector === '[data-platform-code]') return this.codeBlocks;
    return [];
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  dispatchEvent(event) {
    for (const fn of this.listeners.get(event.type) || []) fn(event);
  }
}

class FakeStorage {
  constructor() {
    this.values = new Map();
  }
  getItem(key) {
    return this.values.has(key) ? this.values.get(key) : null;
  }
  setItem(key, value) {
    this.values.set(key, String(value));
  }
}

class FakeCustomEvent {
  constructor(type, init = {}) {
    this.type = type;
    this.detail = init.detail;
  }
}

const platformButtons = [
  new FakeElement({ dataset: { platformOption: 'posix' }, selected: true }),
  new FakeElement({ dataset: { platformOption: 'powershell' } }),
];
const codeBlocks = [
  new FakeElement({ dataset: { platformCode: 'posix' } }),
  new FakeElement({ dataset: { platformCode: 'powershell' } }),
];
const section = new FakeSection(platformButtons);
const document = new FakeDocument(section, platformButtons, codeBlocks);
const localStorage = new FakeStorage();
globalThis.document = document;
globalThis.window = { localStorage };
globalThis.CustomEvent = FakeCustomEvent;

await import('./site/src/scripts/configurator.ts');

const beforeClick = {
  posixVisible: codeBlocks[0].hidden,
  powershellVisible: codeBlocks[1].hidden,
  copy: section.copyBtn.getAttribute('data-copy'),
};

platformButtons[1].click();

const afterClick = {
  stored: localStorage.getItem('chunkhound:platform'),
  posixSelected: platformButtons[0].getAttribute('aria-selected'),
  powershellSelected: platformButtons[1].getAttribute('aria-selected'),
  posixHidden: codeBlocks[0].hidden,
  powershellHidden: codeBlocks[1].hidden,
  copy: section.copyBtn.getAttribute('data-copy'),
};

console.log(JSON.stringify({ beforeClick, afterClick }));
"""
    rendered = run_tsx_json(script)

    assert rendered["beforeClick"]["posixVisible"] is False
    assert rendered["beforeClick"]["powershellVisible"] is True
    assert "cat > .chunkhound.json" in rendered["beforeClick"]["copy"]
    assert rendered["afterClick"]["stored"] == "powershell"
    assert rendered["afterClick"]["posixSelected"] == "false"
    assert rendered["afterClick"]["powershellSelected"] == "true"
    assert rendered["afterClick"]["posixHidden"] is True
    assert rendered["afterClick"]["powershellHidden"] is False
    assert "Set-Content -Path '.chunkhound.json'" in rendered["afterClick"]["copy"]
