# ruff: noqa: E501  # Embedded JavaScript keeps browser-like expressions intact.
from __future__ import annotations

from tests.site.dom_helpers import browser_dom, dist_body
from tests.site.tsx_runner import run_tsx_json

# The full-mode configurator ships on the getting-started page.
_GETTING_STARTED = "docs/getting-started/index.html"

_INIT = "await import('./site/src/scripts/configurator.ts');\n"

# Shared shorthands over the real section markup (auto-init at import time runs
# the production entry point: initConfigurator on every .configurator).
_SELECTORS = """
const section = window.document.querySelector('.configurator');
const q = (selector) => section.querySelector(selector);
const choose = (attribute, id) => q(`[${attribute}="${id}"]`).click();
const rerankUrl = q('[data-rerank-url]');
const rerankFormat = q('[data-rerank-format]');
const rerankModel = q('[data-rerank-model]');

// There is no detail panel: each stage's checked row owns the reveal that
// describes it, so every "what the visitor reads about their pick" assertion
// goes through that row.
const checkedRow = (role) => q(`[data-${role}]:checked`).closest('.provider-option');
const optionName = (role) =>
  checkedRow(role).querySelector('.option-title strong').textContent;
const optionDescription = (role) =>
  checkedRow(role).querySelector('.option-description')?.textContent ?? null;
// The reveal must hang off the row it describes — a reveal that followed a
// different row would talk about a provider the visitor did not pick.
const revealBelongsToTheCheckedRow = (role) =>
  checkedRow(role).querySelector('[data-option-requirements]').dataset
    .optionRequirements === q(`[data-${role}]:checked`).value;
// Link contract of a rendered requirements list (renderRequirements in
// site/src/components/configurator/utils.ts): new-tab external anchors, each
// carrying one aria-hidden logo span and a visible text label.
const requirementLinks = (role) =>
  [...checkedRow(role).querySelectorAll('.prerequisite-link')].map((a) => ({
    href: a.getAttribute('href'),
    text: a.textContent.trim(),
    target: a.getAttribute('target'),
    rel: a.getAttribute('rel'),
    logo: !!a.querySelector('span[aria-hidden="true"]'),
  }));
"""


def test_configurator_script_updates_selection_callout_and_copy_contract(
    built_site,
) -> None:
    script = (
        browser_dom(dist_body(_GETTING_STARTED))
        + _INIT
        + _SELECTORS
        + """
const callout = q('#prerequisite-callout');
const copyBtn = q('#config-copy-btn');
// The single reranker editor must sit inside the checked retrieval row's reveal.
const rerankHost = () =>
  q('[data-rerank-endpoint]').closest('.provider-option')
    .querySelector('.option-title strong').textContent;
const rerankCount = () => section.querySelectorAll('[data-rerank-endpoint]').length;

const initialHidden = callout.hasAttribute('hidden');

choose('data-retrieval', 'ollama-embed');
const afterEmbed = {
  calloutHidden: callout.hasAttribute('hidden'),
  calloutCopy: q('#prerequisite-copy-btn').getAttribute('data-copy'),
  configCopy: copyBtn.getAttribute('data-copy'),
  embedSelected: q('[data-retrieval="ollama-embed"]').checked,
  voyageSelected: q('[data-retrieval="voyageai"]').checked,
  stageName: optionName('retrieval'),
  stageDescription: optionDescription('retrieval'),
  // Editors carry no description: the row renders no empty line for them.
  agentDescription: optionDescription('agent'),
  revealOwned: revealBelongsToTheCheckedRow('retrieval'),
  rerankHost: rerankHost(),
  rerankCount: rerankCount(),
  retrievalLinks: requirementLinks('retrieval'),
  researchLinks: requirementLinks('research'),
  agentLinks: requirementLinks('agent'),
  state: q('[data-rerank-endpoint]').getAttribute('data-rerank-state'),
  label: q('[data-rerank-state-label]').textContent,
  status: q('[data-rerank-status]').textContent,
  summary: q('[data-rerank-details-summary]').textContent,
  detailsOpen: q('[data-rerank-details]').hasAttribute('open'),
};

rerankUrl.value = 'https://tei.example.com/rerank';
rerankFormat.value = 'tei';
rerankFormat.dispatchEvent(new window.Event('change'));
const afterExternalConfig = {
  configCopy: copyBtn.getAttribute('data-copy'),
  errorHidden: q('[data-rerank-url-error]').hasAttribute('hidden'),
};

choose('data-agent', 'codex');
const afterEditor = {
  researchChecked: q('[data-research="codex-cli"]').checked,
  configCopy: copyBtn.getAttribute('data-copy'),
};

rerankUrl.value = '';
rerankFormat.value = '';
rerankModel.value = '';
rerankUrl.dispatchEvent(new window.Event('input'));
choose('data-retrieval', 'vllm-embed');
choose('data-research', 'vllm-llm');
const afterVllm = {
  calloutCopy: q('#prerequisite-copy-btn').getAttribute('data-copy'),
  configCopy: copyBtn.getAttribute('data-copy'),
  rerankHost: rerankHost(),
  rerankCount: rerankCount(),
  state: q('[data-rerank-endpoint]').getAttribute('data-rerank-state'),
  status: q('[data-rerank-status]').textContent,
  summary: q('[data-rerank-details-summary]').textContent,
  detailsOpen: q('[data-rerank-details]').hasAttribute('open'),
};

choose('data-research', 'anthropic');
rerankUrl.value = '/rerank';
rerankUrl.dispatchEvent(new window.Event('input'));
choose('data-retrieval', 'voyageai');
const afterReset = {
  calloutHidden: callout.hasAttribute('hidden'),
  calloutCopy: q('#prerequisite-copy-btn').getAttribute('data-copy'),
  configCopy: copyBtn.getAttribute('data-copy'),
  error: q('[data-rerank-url-error]').textContent,
  errorHidden: q('[data-rerank-url-error]').hasAttribute('hidden'),
};

console.log(JSON.stringify({ initialHidden, afterEmbed, afterExternalConfig, afterEditor, afterVllm, afterReset }));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["initialHidden"] is True
    after_embed = rendered["afterEmbed"]
    assert after_embed["calloutHidden"] is False
    assert "ollama pull qwen3-embedding" in after_embed["calloutCopy"]
    assert "qwen3-embedding" in after_embed["configCopy"]
    assert after_embed["embedSelected"] is True
    assert after_embed["voyageSelected"] is False
    assert '"rerank_url"' not in after_embed["configCopy"]
    assert after_embed["state"] == "required"
    assert after_embed["label"] == "Required dependency"
    assert after_embed["status"] == (
        "Add an external reranker to enable research, web search, and fetch. "
        "Semantic search remains available without one."
    )
    assert after_embed["summary"] == "Add external reranker"
    # A required reranker is the pick's own field, not an option to reveal: its
    # form is open, and its summary (the only toggle) is hidden in CSS.
    assert after_embed["detailsOpen"] is True
    assert after_embed["stageName"] == "Ollama"
    assert after_embed["stageDescription"] == "Local embedding model via Ollama"
    assert after_embed["agentDescription"] is None
    assert after_embed["revealOwned"] is True
    # The reranker editor is the checked row's own override, never a stray
    # stage-level block — and only one instance ever exists.
    assert after_embed["rerankHost"] == "Ollama"
    assert after_embed["rerankCount"] == 1
    # Requirement links, per the configurator requirements map: same hrefs,
    # labels, and new-tab/logo contract as renderRequirements renders.
    def link_contract(href: str, text: str) -> dict:
        return {
            "href": href,
            "text": text,
            "target": "_blank",
            "rel": "noopener noreferrer",
            "logo": True,
        }

    assert after_embed["retrievalLinks"] == [
        link_contract("https://ollama.com/download", "Ollama"),
        link_contract(
            "https://ollama.com/library/qwen3-embedding", "qwen3-embedding model"
        ),
    ]
    # The other two stages keep their server-rendered defaults: the checked
    # OpenRouter and Pi rows carry their own chips.
    assert after_embed["researchLinks"] == [
        link_contract("https://openrouter.ai/keys", "OpenRouter API key")
    ]
    assert after_embed["agentLinks"] == [
        link_contract("https://pi.dev/packages/pi-mcp-adapter", "pi-mcp-adapter")
    ]
    assert (
        '"rerank_url": "https://tei.example.com/rerank"'
        in rendered["afterExternalConfig"]["configCopy"]
    )
    assert '"rerank_format": "tei"' in rendered["afterExternalConfig"]["configCopy"]
    assert rendered["afterExternalConfig"]["errorHidden"] is True
    assert rendered["afterEditor"]["researchChecked"] is False
    assert '"model": "google/gemini-3.5-flash"' in rendered["afterEditor"]["configCopy"]
    vllm_callout = rendered["afterVllm"]["calloutCopy"]
    assert "Qwen/Qwen3-Coder-30B-A3B-Instruct --port 8002" in vllm_callout
    vllm_config = rendered["afterVllm"]["configCopy"]
    assert '"model": "Qwen/Qwen3-Embedding-0.6B"' in vllm_config
    assert '"rerank_model": "Qwen/Qwen3-Reranker-0.6B"' in vllm_config
    assert '"rerank_url": "http://localhost:8001/v1/rerank"' in vllm_config
    assert '"base_url": "http://localhost:8000/v1"' in vllm_config
    assert rendered["afterVllm"]["state"] == "included"
    assert rendered["afterVllm"]["status"] == (
        "Included by this preset — runs as a separate local vLLM service on port 8001."
    )
    # The override followed the pick: still one editor, now hosted by vLLM.
    assert rendered["afterVllm"]["rerankHost"] == "vLLM"
    assert rendered["afterVllm"]["rerankCount"] == 1
    assert rendered["afterVllm"]["summary"] == "Customize reranker"
    # vLLM includes its reranker, so the disclosure is the user's again — and the
    # forced-open state of the previous, required pick must not leak into it.
    assert rendered["afterVllm"]["detailsOpen"] is False
    assert rendered["afterReset"]["calloutHidden"] is True
    assert rendered["afterReset"]["calloutCopy"] is None
    assert '"rerank_url"' not in rendered["afterReset"]["configCopy"]
    assert rendered["afterReset"]["errorHidden"] is False
    assert (
        "requires this retrieval preset to have a base URL"
        in rendered["afterReset"]["error"]
    )


def test_configurator_reranker_errors_flag_only_the_offending_input(
    built_site,
) -> None:
    """aria-invalid marks the exact field to fix: a missing model must not
    flag the URL, and a cleared error clears every flag."""
    script = (
        browser_dom(dist_body(_GETTING_STARTED))
        + _INIT
        + _SELECTORS
        + """
const invalid = () => ['url', 'format', 'model'].map((role) =>
  q(`[data-rerank-${role}]`).getAttribute('aria-invalid'));
const errorText = () => q('[data-rerank-url-error]').textContent;

// ollama-embed requires an external reranker; the details open on selection.
choose('data-retrieval', 'ollama-embed');
// Valid TEI URL: no error, no flags.
rerankUrl.value = 'https://tei.example.com/rerank';
rerankFormat.value = 'tei';
rerankFormat.dispatchEvent(new window.Event('change'));
const validTei = { error: errorText(), invalid: invalid() };

// Cohere without a model: error names the model, only the model is flagged.
rerankFormat.value = 'cohere';
rerankFormat.dispatchEvent(new window.Event('change'));
const modelRequired = { error: errorText(), invalid: invalid() };

// Filling the model clears the error and all flags.
rerankModel.value = 'bge-reranker-v2-m3';
rerankModel.dispatchEvent(new window.Event('input'));
const resolved = { error: errorText(), invalid: invalid() };

// URL error (non-HTTP scheme): only the URL is flagged.
rerankUrl.value = 'ftp://example.com/rerank';
rerankUrl.dispatchEvent(new window.Event('input'));
const badUrl = { error: errorText(), invalid: invalid() };

console.log(JSON.stringify({ validTei, modelRequired, resolved, badUrl }));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["validTei"] == {"error": "", "invalid": ["false"] * 3}
    assert rendered["modelRequired"]["error"] == (
        "A Cohere or Voyage reranker requires a model."
    )
    assert rendered["modelRequired"]["invalid"] == ["false", "false", "true"]
    assert rendered["resolved"] == {"error": "", "invalid": ["false"] * 3}
    assert rendered["badUrl"]["error"] == "Use an absolute HTTP(S) URL or a relative path."
    assert rendered["badUrl"]["invalid"] == ["true", "false", "false"]


def test_configurator_rejects_url_less_tei_when_preset_has_no_base_url(
    built_site,
) -> None:
    """openai-embed has no base_url, so a URL-less TEI reranker cannot be
    derived by the backend (embedding_config.validate_rerank_config raises
    RERANK_BASE_URL_REQUIRED). The UI must reject it on the URL field instead
    of green-lighting a config that fails on first run."""
    script = (
        browser_dom(dist_body(_GETTING_STARTED))
        + _INIT
        + _SELECTORS
        + """
const invalid = () => ['url', 'format', 'model'].map((role) =>
  q(`[data-rerank-${role}]`).getAttribute('aria-invalid'));
const errorText = () => q('[data-rerank-url-error]').textContent;

// openai-embed carries no base_url; selecting it reveals the external editor.
choose('data-retrieval', 'openai-embed');
rerankFormat.value = 'tei';
rerankFormat.dispatchEvent(new window.Event('change'));
const urlLessTei = { error: errorText(), invalid: invalid() };

// Supplying the missing URL clears the error and every flag.
rerankUrl.value = 'https://tei.example.com/rerank';
rerankUrl.dispatchEvent(new window.Event('input'));
const resolved = { error: errorText(), invalid: invalid() };

console.log(JSON.stringify({ urlLessTei, resolved }));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["urlLessTei"]["error"] == (
        "This preset has no base URL — enter the reranker URL."
    )
    assert rendered["urlLessTei"]["invalid"] == ["true", "false", "false"]
    assert rendered["resolved"] == {"error": "", "invalid": ["false"] * 3}


def test_configurator_radio_checked_drives_selection_and_ignores_class_markup(
    built_site,
) -> None:
    script = (
        browser_dom(dist_body(_GETTING_STARTED))
        + """
// Strip the server-rendered `checked` attributes and mark a non-default
// provider with a `.selected` class: selectedId() reads only `:checked`, so
// the class must not influence the initial selection (which falls back to
// the preset defaults) and only a checked radio can move it.
const section = window.document.querySelector('.configurator');
const markClassButNotChecked = () => {
  section.querySelectorAll('[data-retrieval]').forEach((input) => {
    input.removeAttribute('checked');
  });
  section.querySelector('[data-retrieval="vllm-embed"]').className = 'selected';
};
markClassButNotChecked();
"""
        + _INIT
        + """
const q = (selector) => section.querySelector(selector);
const initialCopy = q('#config-copy-btn').getAttribute('data-copy');
const outputBefore = q('#config-output').innerHTML;
// No `:checked` radio at all, so no row is open and no reveal is shown.
const initialOpenReveals = section.querySelectorAll('[data-retrieval]:checked').length;
q('[data-retrieval="ollama-embed"]').click();
const afterChange = {
  defaultChecked: q('[data-retrieval="voyageai"]').checked,
  newChecked: q('[data-retrieval="ollama-embed"]').checked,
  outputHtml: q('#config-output').innerHTML,
  copy: q('#config-copy-btn').getAttribute('data-copy'),
  // The open reveal is the one inside the row that is now checked.
  requirements: q('[data-retrieval]:checked').closest('.provider-option')
    .querySelector('[data-option-requirements]').innerHTML,
  openRow: q('[data-retrieval]:checked').closest('.provider-option').dataset
    .providerOption,
  // Exactly one open reveal per stage: the CSS opens whatever `:checked` marks.
  checkedRowsPerStage: ['retrieval', 'research', 'agent'].map(
    (role) => section.querySelectorAll(`[data-${role}]:checked`).length,
  ),
};
console.log(JSON.stringify({ initialCopy, initialOpenReveals, outputBefore, afterChange }));
"""
    )
    rendered = run_tsx_json(script)

    # The `.selected` class is ignored: with no `:checked` radio the selection
    # falls back to the preset defaults, not to the class-marked provider.
    assert "voyage-4-lite" in rendered["initialCopy"]
    assert rendered["initialOpenReveals"] == 0
    assert rendered["afterChange"]["defaultChecked"] is False
    assert rendered["afterChange"]["newChecked"] is True
    assert "qwen3-embedding" in rendered["afterChange"]["outputHtml"]
    assert "qwen3-embedding" in rendered["afterChange"]["copy"]
    assert rendered["afterChange"]["outputHtml"] != rendered["outputBefore"]
    assert "qwen3-embedding model</a></li>" in rendered["afterChange"]["requirements"]
    assert "VoyageAI API key" not in rendered["afterChange"]["requirements"]
    assert rendered["afterChange"]["openRow"] == "retrieval"
    assert rendered["afterChange"]["checkedRowsPerStage"] == [1, 1, 1]


def test_configurator_provider_search_filters_options(built_site) -> None:
    script = (
        browser_dom(dist_body(_GETTING_STARTED))
        + _INIT
        + """
// Real markup ships provider search for the research role.
const section = window.document.querySelector('.configurator');
const filterInput = section.querySelector('[data-provider-search="research"]');
const options = () => [...section.querySelectorAll('[data-provider-option="research"]')];
const groups = () => [...section.querySelectorAll('[data-provider-group="research"]')];
const status = () => section.querySelector('[data-search-status="research"]').textContent;
const type = (value) => {
  filterInput.value = value;
  filterInput.dispatchEvent(new window.Event('input'));
};

const before = {
  total: options().length,
  hidden: options().filter((option) => option.hidden).length,
};
// The reveal is the row's own tail, so hiding a row hides its reveal with it.
// `openRow` is the row the visitor is currently reading (the checked one), and
// it is deliberately filtered out here: no reveal may outlive its own row.
const openRow = () =>
  section.querySelector('[data-research]:checked').closest('[data-provider-option]');
type('ollama');
const filtered = {
  matchesOllama: options()
    .filter((option) => !option.hidden)
    .every((option) => (option.dataset.searchText || '').includes('ollama')),
  visible: options().filter((option) => !option.hidden).length,
  groupHidden: groups().map((group) => group.hidden),
  status: status(),
  openRowHidden: openRow().hidden,
  revealTravelsWithTheRow: openRow().contains(openRow().querySelector('.option-reveal')),
  revealedRows: options().filter((option) => option.querySelector('.option-reveal')).length,
  strayReveals: [...section.querySelectorAll('[data-provider-list="research"] .option-reveal')]
    .filter((reveal) => !reveal.closest('[data-provider-option="research"]')).length,
};
type('');
const cleared = {
  visible: options().filter((option) => !option.hidden).length,
  groupHidden: groups().map((group) => group.hidden),
  status: status(),
  openRowHidden: openRow().hidden,
};
console.log(JSON.stringify({ before, filtered, cleared }));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["before"] == {"total": 12, "hidden": 0}
    assert rendered["filtered"]["matchesOllama"] is True
    assert rendered["filtered"]["visible"] == 1
    assert rendered["filtered"]["groupHidden"] == [True, False, True]
    assert rendered["filtered"]["status"] == "1 result"
    # OpenRouter stays the selection while filtered out, so the hidden row is
    # the one that carries the only open reveal — filtering must hide both.
    assert rendered["filtered"]["openRowHidden"] is True
    assert rendered["filtered"]["revealTravelsWithTheRow"] is True
    assert rendered["filtered"]["revealedRows"] == rendered["before"]["total"]
    assert rendered["filtered"]["strayReveals"] == 0
    assert rendered["cleared"]["visible"] == 12
    assert rendered["cleared"]["groupHidden"] == [False, False, False]
    assert rendered["cleared"]["status"] == "12 results"
    assert rendered["cleared"]["openRowHidden"] is False


def test_configurator_platform_selection_persists_and_updates_code_blocks(
    built_site,
) -> None:
    script = (
        browser_dom(dist_body(_GETTING_STARTED))
        + _INIT
        + """
const pills = () => [...window.document.querySelectorAll('[data-platform-option]')];
const codeBlocks = [...window.document.querySelectorAll('[data-platform-code]')];
const copy = () =>
  window.document.querySelector('#config-copy-btn').getAttribute('data-copy');

// Tabs with real tabpanels report aria-selected; single-panel switchers
// report aria-pressed (no dangling aria-controls).
const stateAttr = (pill) =>
  pill.closest('[data-platform-mode]').dataset.platformMode === 'tabs'
    ? 'aria-selected'
    : 'aria-pressed';

const beforeClick = {
  posixHidden: codeBlocks[0].hidden,
  powershellHidden: codeBlocks[1].hidden,
  copy: copy(),
  switcherControls: pills()
    .filter((pill) => stateAttr(pill) === 'aria-pressed')
    .map((pill) => pill.getAttribute('aria-controls')),
};

window.document.querySelector('[data-platform-option="powershell"]').click();

const afterClick = {
  stored: window.localStorage.getItem('chunkhound:platform'),
  ariaState: pills().map(
    (pill) => `${pill.dataset.platformOption}:${pill.getAttribute(stateAttr(pill))}`,
  ),
  posixHidden: codeBlocks[0].hidden,
  powershellHidden: codeBlocks[1].hidden,
  copy: copy(),
};

console.log(JSON.stringify({ beforeClick, afterClick }));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["beforeClick"]["posixHidden"] is False
    assert rendered["beforeClick"]["powershellHidden"] is True
    assert "cat > .chunkhound.json" in rendered["beforeClick"]["copy"]
    assert rendered["beforeClick"]["switcherControls"] == [None, None]
    assert rendered["afterClick"]["stored"] == "powershell"
    assert rendered["afterClick"]["ariaState"] == [
        "posix:false",
        "powershell:true",
        "posix:false",
        "powershell:true",
    ]
    assert rendered["afterClick"]["posixHidden"] is True
    assert rendered["afterClick"]["powershellHidden"] is False
    assert "Set-Content -Path '.chunkhound.json'" in rendered["afterClick"]["copy"]


def test_configurator_stage_tabs_enforce_roving_tabindex_and_panel_visibility(
    built_site,
) -> None:
    """Stage tabs behave like ARIA tabs: one selected, roving tabindex, panels toggled."""
    script = (
        browser_dom(dist_body(_GETTING_STARTED))
        + _INIT
        + """
const section = window.document.querySelector('.configurator');
const tabs = [...section.querySelectorAll('[data-stage-tab]')];
const ids = tabs.map((tab) => tab.dataset.stageTab);
const tabById = (id) => section.querySelector(`[data-stage-tab="${id}"]`);
const panel = (id) => section.querySelector(`[data-stage-panel="${id}"]`);
const snapshot = () => ({
  selected: ids.filter((id) =>
    tabById(id).getAttribute('aria-selected') === 'true'
  ),
  tabIndexes: tabs.map((tab) => tab.tabIndex),
  panelHidden: ids.map((id) => panel(id).hidden),
  focused: window.document.activeElement?.dataset?.stageTab ?? null,
});
const press = (key) => {
  const active = tabs.find((tab) =>
    tab.getAttribute('aria-selected') === 'true'
  ) ?? tabs[0];
  active.dispatchEvent(new window.KeyboardEvent('keydown', {
    key, bubbles: true, cancelable: true,
  }));
};

const initial = snapshot();
press('ArrowRight');
const afterArrowRight = snapshot();
press('Home');
const afterHome = snapshot();
press('End');
const afterEnd = snapshot();
press('ArrowLeft');
const afterArrowLeftWrap = snapshot();
press('ArrowRight');
const afterArrowRightBack = snapshot();
press('PageDown');
const afterUnsupportedKey = snapshot();
console.log(JSON.stringify({
  ids, initial, afterArrowRight, afterHome, afterEnd,
  afterArrowLeftWrap, afterArrowRightBack, afterUnsupportedKey,
}));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["ids"] == ["retrieval", "research", "agent"]
    assert rendered["initial"] == {
        "selected": ["retrieval"],
        "tabIndexes": [0, -1, -1],
        "panelHidden": [False, True, True],
        "focused": None,
    }
    # ArrowRight moves selection research → and focuses the newly active tab.
    assert rendered["afterArrowRight"] == {
        "selected": ["research"],
        "tabIndexes": [-1, 0, -1],
        "panelHidden": [True, False, True],
        "focused": "research",
    }
    # Home/End jump to first/last; ArrowLeft wraps agent → research.
    assert rendered["afterHome"] == {
        "selected": ["retrieval"],
        "tabIndexes": [0, -1, -1],
        "panelHidden": [False, True, True],
        "focused": "retrieval",
    }
    assert rendered["afterEnd"] == {
        "selected": ["agent"],
        "tabIndexes": [-1, -1, 0],
        "panelHidden": [True, True, False],
        "focused": "agent",
    }
    assert rendered["afterArrowLeftWrap"] == {
        "selected": ["research"],
        "tabIndexes": [-1, 0, -1],
        "panelHidden": [True, False, True],
        "focused": "research",
    }
    assert rendered["afterArrowRightBack"] == {
        "selected": ["agent"],
        "tabIndexes": [-1, -1, 0],
        "panelHidden": [True, True, False],
        "focused": "agent",
    }
    # Navigation keys only: unsupported keys leave the selection untouched.
    assert rendered["afterUnsupportedKey"] == rendered["afterArrowRightBack"]


def test_a_pick_lands_in_the_terminal_and_the_copy_at_once(
    built_site,
) -> None:
    """One pick, one state update - nothing waits on presentation.

    The reveal is CSS (`:has(input:checked)`), so a pick must land the radio,
    the generated commands and the copy payload in the same tick as `change`,
    with or without reduced motion and before any frame is flushed. A row that
    opens while the terminal still describes the previous provider lets a
    visitor copy the wrong setup.
    """
    script = (
        browser_dom(dist_body(_GETTING_STARTED))
        + _INIT
        + _SELECTORS
        + """
// Per pass: role, picked id, what proves the pick in the commands, and what
// proves the previous pick is gone. Two passes because re-picking an
// already-checked radio fires no `change`.
const picksFor = (retrieval, research, agent) => [
  ['retrieval', retrieval[0], retrieval[1], 'voyage-4-lite'],
  ['research', research[0], research[1], 'google/gemini-3.5-flash'],
  ['agent', agent[0], agent[1], 'pi install npm:pi-mcp-adapter'],
];
const snapshot = (role) => ({
  terminal: q('#config-output').textContent,
  copy: q('#config-copy-btn').getAttribute('data-copy'),
  row: optionName(role),
  reveal: checkedRow(role).querySelector('[data-option-requirements]').dataset
    .optionRequirements,
});
const pickAll = (picks) =>
  picks.map(([role, id, marker, stale]) => {
    choose(`data-${role}`, id);
    const sameTick = snapshot(role);
    // Whatever the reveal still has queued must be able to change nothing:
    // the state was already final when the tick ended.
    globalThis.flushRaf?.();
    const afterFrame = snapshot(role);
    return {
      role,
      id,
      marker,
      inTerminal: sameTick.terminal.includes(marker),
      inCopy: sameTick.copy.includes(marker),
      staleGone: !sameTick.copy.includes(stale),
      revealIsTheCheckedRow: sameTick.reveal === id,
      deferredWorkChangedAnything:
        JSON.stringify(sameTick) !== JSON.stringify(afterFrame),
    };
  });

const motion = pickAll(picksFor(
  ['ollama-embed', 'qwen3-embedding'],
  ['ollama-llm', 'qwen3-coder:30b'],
  ['codex', 'codex mcp add ChunkHound'],
));
setReducedMotion(true);
const reduced = pickAll(picksFor(
  ['openai-embed', 'text-embedding-3-small'],
  ['anthropic', '"provider": "anthropic"'],
  ['cursor', '.cursor/mcp.json'],
));
setReducedMotion(false);
console.log(JSON.stringify({ motion, reduced }));
"""
    )
    rendered = run_tsx_json(script)

    for result in [*rendered["motion"], *rendered["reduced"]]:
        stage = f"{result['role']} -> {result['id']}"
        assert result["inTerminal"], f"{stage}: the terminal describes another pick"
        assert result["inCopy"], f"{stage}: the copy payload describes another pick"
        assert result["staleGone"], f"{stage}: the copied setup carries the previous pick"
        assert result["revealIsTheCheckedRow"], (
            f"{stage}: the open reveal describes a different row"
        )
        assert result["deferredWorkChangedAnything"] is False, (
            f"{stage}: part of the pick was deferred out of the tick"
        )
