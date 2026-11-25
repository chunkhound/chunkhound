// Line-grammar highlighter for configurator output (shell + embedded JSON).
//
// WHY NOT SHIKI: configurator blocks mix posix heredocs (`<<'CHUNKHOUND_EOF'`),
// PowerShell here-strings (`@' ... '@ | Set-Content`), and pretty-printed JSON
// writes in one block. The PowerShell command vocabulary is owned by
// builders.ts/shell-write.ts (the only PowerShell emitters), so detection by
// command sniffing is exact here. Shiki would retokenize these mixed spans
// under a single grammar and break the frozen sh-*/json-* span contract
// pinned by tests/site/test_highlight_golden.py. Keep patterns, render fns,
// class strings, and pattern ordering byte-identical.
import type { ConfiguratorPlatform } from "./types.ts";
import { guardedJsonWriteScaffold, jsonWriteScaffold } from "./shell-write.ts";
import { escapeHtml, escapeHtmlTags } from "./utils.ts";

function wrapJsonKeySpan(key: string): string {
  return `<span class="json-key">"${escapeHtml(key)}"</span>`;
}

function wrapCommaSpan(comma: string): string {
  return comma ? `<span class="json-punct">${comma}</span>` : "";
}

function renderJsonKeyValue(
  indent: string,
  key: string,
  valueHtml: string,
  comma: string,
): string {
  return (
    indent +
    wrapJsonKeySpan(key) +
    `<span class="json-punct">:</span> ` +
    `<span class="json-string">${valueHtml}</span>` +
    wrapCommaSpan(comma)
  );
}

function renderJsonStringElement(
  indent: string,
  value: string,
  comma: string,
): string {
  return (
    indent +
    `<span class="json-string">"${escapeHtml(value)}"</span>` +
    wrapCommaSpan(comma)
  );
}

interface JsonLinePattern {
  pattern: RegExp;
  // render receives the full match; m[0] is the line content, groups follow.
  render: (match: RegExpMatchArray, indent: string) => string;
}

// Ordered line grammar for pretty-printed JSON; first matching pattern wins.
// Single source of truth so adding a shape cannot fork the wrapping logic.
const JSON_LINE_PATTERNS: JsonLinePattern[] = [
  {
    // "key": "value",?
    pattern: /^"([^"]+)":\s*"([^"]*)"(,?)$/,
    render: (m, indent) =>
      renderJsonKeyValue(indent, m[1]!, `"${escapeHtml(m[2]!)}"`, m[3]!),
  },
  {
    // "key": <number/bool/null>,?
    pattern: /^"([^"]+)":\s*([^"{}\[\],]+)(,?)$/,
    render: (m, indent) =>
      renderJsonKeyValue(indent, m[1]!, escapeHtml(m[2]!), m[3]!),
  },
  {
    // "key": {  or  "key": [
    pattern: /^"([^"]+)":\s*([{[])$/,
    render: (m, indent) =>
      indent +
      wrapJsonKeySpan(m[1]!) +
      `<span class="json-punct">: ${escapeHtml(m[2]!)}</span>`,
  },
  {
    // bare string array element: "value",?
    pattern: /^"([^"]*)"(,?)$/,
    render: (m, indent) => renderJsonStringElement(indent, m[1]!, m[2]!),
  },
  {
    // structural punctuation: { } [ ] }, ],
    pattern: /^[{}\[\]](,?)$/,
    render: (m, indent) =>
      indent + `<span class="json-punct">${escapeHtml(m[0]!)}</span>`,
  },
];

export function highlightJsonLine(line: string): string {
  const m = line.match(/^(\s*)(.*)$/);
  if (!m) return escapeHtml(line);
  const [, indent = "", content = ""] = m;
  if (!content) return indent;
  for (const { pattern, render } of JSON_LINE_PATTERNS) {
    const match = content.match(pattern);
    if (match) return render(match, indent);
  }
  return indent + escapeHtml(content);
}

export function prettifyJsonBlock(obj: unknown): { plain: string; html: string } {
  const plain = JSON.stringify(obj, null, 2);
  const html = plain.split("\n").map(highlightJsonLine).join("\n");
  return { plain, html };
}

export function highlightCommentLine(line: string): string {
  return `<span class="sh-comment">${escapeHtml(line)}</span>`;
}

export function highlightShellLine(line: string): string {
  if (line.startsWith("#")) return highlightCommentLine(line);

  // Escape tag delimiters before wrapping tokens in class spans: heredoc
  // lines contain `<<` which would otherwise be emitted raw into markup as a
  // tag opening. Quotes stay raw so the string-token regex below can match.
  const escaped = escapeHtmlTags(line);
  return escaped
    .replace(/^(\w+)/, '<span class="sh-cmd">$1</span>')
    .replace(/'([^']*)'/, "'<span class=\"sh-str\">$1</span>'")
    .replace(/ &gt;&gt; /, ' <span class="sh-op">&gt;&gt;</span> ')
    .replace(/ &gt; /, ' <span class="sh-op">&gt;</span> ');
}

function wrapShellToken(value: string, cls: string): string {
  return `<span class="${cls}">${escapeHtml(value)}</span>`;
}

function highlightPowerShellLine(line: string): string {
  if (!line) return "";
  if (line.startsWith("#")) return highlightCommentLine(line);

  const setContentMatch = line.match(
    /^'@ \| Set-Content -Path (.+) -Encoding (.+)$/,
  );
  if (setContentMatch)
    return renderSetContentLine(setContentMatch[1]!, setContentMatch[2]!);

  return renderPowerShellTokens(line);
}

function renderSetContentLine(path: string, encoding: string): string {
  return [
    wrapShellToken("'@", "sh-op"),
    " ",
    wrapShellToken("|", "sh-op"),
    " ",
    wrapShellToken("Set-Content", "sh-cmd"),
    " ",
    wrapShellToken("-Path", "sh-op"),
    " ",
    wrapShellToken(path, "sh-file"),
    " ",
    wrapShellToken("-Encoding", "sh-op"),
    " ",
    escapeHtml(encoding),
  ].join("");
}

interface PowerShellTokenState {
  expectCommand: boolean;
  expectPath: boolean;
}

function renderQuotedPowerShellToken(token: string): string | undefined {
  if (!/^".*"$/.test(token) && !/^'.*'$/.test(token)) return undefined;
  const pathLike =
    token.includes("/") || token.includes("\\") || token.includes(".json");
  return wrapShellToken(token, pathLike ? "sh-file" : "sh-str");
}

function renderPowerShellOperator(
  token: string,
  state: PowerShellTokenState,
): string | undefined {
  if (/^\s+$/.test(token)) return token;
  if (token === "|") {
    state.expectCommand = true;
    state.expectPath = false;
    return wrapShellToken(token, "sh-op");
  }
  if (token === "@'" || token === "'@") {
    state.expectCommand = false;
    state.expectPath = false;
    return wrapShellToken(token, "sh-op");
  }
  if (token.startsWith("-")) {
    state.expectCommand = false;
    state.expectPath = token === "-Path";
    return wrapShellToken(token, "sh-op");
  }
  return undefined;
}

function renderPowerShellToken(token: string, state: PowerShellTokenState): string {
  const operator = renderPowerShellOperator(token, state);
  if (operator) return operator;
  if (state.expectPath) {
    state.expectPath = false;
    state.expectCommand = false;
    return wrapShellToken(token, "sh-file");
  }
  if (state.expectCommand) {
    state.expectCommand = false;
    return wrapShellToken(token, "sh-cmd");
  }
  return renderQuotedPowerShellToken(token) ?? escapeHtml(token);
}

function renderPowerShellTokens(line: string): string {
  const tokens = line.match(/"[^"]*"|'(?:''|[^'])*'|[|]|[^\s|]+|\s+/g) ?? [
    line,
  ];
  const state: PowerShellTokenState = {
    expectCommand: true,
    expectPath: false,
  };
  return tokens.map((token) => renderPowerShellToken(token, state)).join("");
}

// Flow-control keywords that can end a generated posix line; the trailing
// token on those lines is a keyword, never a path.
const POSIX_FLOW_KEYWORDS = new Set(["then", "elif", "else", "fi", "do", "done"]);

export function highlightInlineShellLine(line: string): string {
  if (!line) return "";
  // PowerShell detection by command sniffing: this vocabulary is owned by
  // builders.ts/shell-write.ts (the only PowerShell emitters), so a posix
  // line mentioning e.g. "New-Item" cannot occur in generated output.
  if (
    line.includes("Set-Content") ||
    line.includes("New-Item") ||
    line.includes("Add-Content") ||
    line.includes("Test-Path") ||
    line.includes("Write-Host") ||
    line === "@'" ||
    line.startsWith("'@")
  ) {
    return highlightPowerShellLine(line);
  }

  return highlightShellLine(line).replace(
    /(^| )([\w.~/-]+)$/g,
    (match, prefix: string, path: string) =>
      // Flow-control keywords end merge-guard lines (`...; then`, `else`,
      // `fi`) — wrapping them as file paths would mislead.
      POSIX_FLOW_KEYWORDS.has(path)
        ? match
        : `${prefix}<span class="sh-file">${escapeHtml(path)}</span>`,
  );
}

export function highlightInlineShellBlock(text: string): string {
  return text.split("\n").map(highlightInlineShellLine).join("\n");
}

export function renderMixedJsonWriteBlock(
  filename: string,
  jsonHtmlLines: string[],
  platform: ConfiguratorPlatform,
): string {
  const scaffold = jsonWriteScaffold(filename, platform);
  const highlight =
    platform === "powershell" ? highlightInlineShellLine : highlightShellLine;
  return [
    scaffold.mkdir && highlight(scaffold.mkdir),
    highlight(scaffold.open),
    ...jsonHtmlLines,
    highlight(scaffold.close),
  ]
    .filter(Boolean)
    .join("\n");
}

// HTML twin of assembleGuardedJsonWrite — consumes the same scaffold fields so
// preview and copy stay byte-identical (pinned by the golden suite).
export function renderMixedGuardedJsonWriteBlock(
  filename: string,
  jsonHtmlLines: string[],
  platform: ConfiguratorPlatform,
): string {
  const scaffold = guardedJsonWriteScaffold(filename, platform);
  const highlight =
    platform === "powershell" ? highlightInlineShellLine : highlightShellLine;
  return [
    highlight(scaffold.guardOpen),
    highlight(scaffold.printOpen),
    // The merge instruction is printed text, not a command — keep it plain.
    escapeHtml(scaffold.message),
    ...jsonHtmlLines,
    highlight(scaffold.printClose),
    highlight(scaffold.elseLine),
    renderMixedJsonWriteBlock(filename, jsonHtmlLines, platform),
    highlight(scaffold.guardClose),
  ].join("\n");
}
