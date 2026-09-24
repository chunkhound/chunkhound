# ruff: noqa: E501  # Embedded JavaScript keeps browser-like expressions intact.
from __future__ import annotations

from tests.site.dom_helpers import MANUAL_TIMERS, browser_dom
from tests.site.tsx_runner import run_tsx_json


def test_copy_handler_uses_clipboard_then_falls_back_and_resets_status() -> None:
    script = (
        browser_dom(
            '''
<div class="code-block-md">
  <button class="copy-btn" data-copy="clipboard text"></button>
  <span data-copy-status></span>
</div>
'''
        )
        + MANUAL_TIMERS
        + """

let clipboardMode = 'success';
let clipboardWrites = [];
let execResult = true;
let execCalls = 0;
Object.defineProperty(globalThis, 'navigator', {
  configurable: true,
  value: {
    clipboard: {
      writeText: async (text) => {
        clipboardWrites.push(text);
        if (clipboardMode === 'reject') throw new Error('clipboard denied');
      },
    },
  },
});
document.execCommand = () => {
  execCalls += 1;
  return execResult;
};

await import('./site/src/scripts/copy-handler.js');
const button = window.document.querySelector('.copy-btn');
const status = window.document.querySelector('[data-copy-status]');
const click = async (text) => {
  button.setAttribute('data-copy', text);
  button.click();
  await Promise.resolve();
  await Promise.resolve();
  return {
    copied: button.classList.contains('copied'),
    status: status.textContent,
  };
};

const clipboard = await click('clipboard text');
flushTimers();
const reset = {
  copied: button.classList.contains('copied'),
  status: status.textContent,
};

clipboardMode = 'reject';
execResult = true;
const fallback = await click('fallback text');
flushTimers();

execResult = false;
const failed = await click('failed text');
flushTimers();
const failedReset = {
  copied: button.classList.contains('copied'),
  status: status.textContent,
};

console.log(JSON.stringify({
  clipboard,
  reset,
  fallback,
  failed,
  failedReset,
  clipboardWrites,
  execCalls,
  scratchTextareas: window.document.querySelectorAll('textarea').length,
}));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["clipboard"] == {"copied": True, "status": "Copied"}
    assert rendered["reset"] == {"copied": False, "status": ""}
    assert rendered["fallback"] == {"copied": True, "status": "Copied"}
    assert rendered["failed"] == {"copied": False, "status": "Copy failed"}
    assert rendered["failedReset"] == {"copied": False, "status": ""}
    assert rendered["clipboardWrites"] == [
        "clipboard text",
        "fallback text",
        "failed text",
    ]
    assert rendered["execCalls"] == 2
    assert rendered["scratchTextareas"] == 0


def test_copy_handler_announces_failure_on_install_command() -> None:
    """The install CTA (InstallCommand markup) announces copy
    failure through its status live region — a silent copy failure on the
    primary install affordance is an a11y defect."""
    script = (
        browser_dom(
            '''
<button class="install-command copy-btn" data-copy="uv tool install chunkhound">
  <code>uv tool install chunkhound</code>
  <span class="copy-status sr-only" data-copy-status role="status" aria-live="polite"></span>
</button>
'''
        )
        + MANUAL_TIMERS
        + """

Object.defineProperty(globalThis, 'navigator', {
  configurable: true,
  value: { clipboard: { writeText: async () => { throw new Error('denied'); } } },
});
document.execCommand = () => false;

await import('./site/src/scripts/copy-handler.js');
const button = window.document.querySelector('.copy-btn');
const status = window.document.querySelector('[data-copy-status]');
button.click();
await Promise.resolve();
await Promise.resolve();
const failed = {
  liveRegion: status.getAttribute('role') === 'status',
  message: status.textContent,
  copied: button.classList.contains('copied'),
};

console.log(JSON.stringify(failed));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered == {
        "liveRegion": True,
        "message": "Copy failed",
        "copied": False,
    }


def test_install_pill_copies_from_a_click_anywhere_inside_it() -> None:
    """The whole pill is one hit target. An icon-only button made the command
    text click-dead — users click the command they can see, not the glyph
    beside it."""
    script = (
        browser_dom(
            '''
<button class="install-command copy-btn" data-copy="uv tool install chunkhound">
  <code>uv tool install chunkhound</code>
  <span class="copy-status sr-only" data-copy-status role="status" aria-live="polite"></span>
</button>
'''
        )
        + MANUAL_TIMERS
        + """

const writes = [];
Object.defineProperty(globalThis, 'navigator', {
  configurable: true,
  value: { clipboard: { writeText: async (text) => { writes.push(text); } } },
});

await import('./site/src/scripts/copy-handler.js');
// Click the command text itself, not the copy glyph.
window.document.querySelector('.install-command code').click();
await Promise.resolve();
await Promise.resolve();
console.log(JSON.stringify({
  writes,
  status: window.document.querySelector('[data-copy-status]').textContent,
}));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered == {
        "writes": ["uv tool install chunkhound"],
        "status": "Copied",
    }
