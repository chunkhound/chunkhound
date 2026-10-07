import type { ConfiguratorPlatform } from "./types.ts";

export function getParentDir(filePath: string): string | null {
  const lastSlash = filePath.lastIndexOf("/");
  if (lastSlash <= 0) {
    return null;
  }
  return filePath.slice(0, lastSlash);
}

export function quotePowerShell(value: string): string {
  if (value.startsWith("$HOME/")) {
    // Double quotes expand `$` (backtick/quote escapes need backticks too),
    // so a later `$` must be escaped or a `$VAR` segment would interpolate.
    // The leading `$HOME` itself must keep expanding — that is why this
    // branch double-quotes — so only non-initial `$` are escaped.
    const escaped = value.replace(/`/g, "``").replace(/"/g, '`"').replace(/(?!^)\$/g, '`$');
    return `"${escaped}"`;
  }
  return `'${value.replace(/'/g, "''")}'`;
}

// POSIX quoting. `~` does not expand inside quotes, so home-relative paths
// that need quoting must be manually expanded to "$HOME/…" — mirroring
// quotePowerShell. Shell-safe words (including a safe tilde-lead, which
// expands fine unquoted) stay bare so clean output is byte-identical to the
// unquoted form; only paths containing split/quote characters get quoted.
export function quotePosix(value: string): string {
  if (!/[^A-Za-z0-9_@%+=:,./~-]/.test(value)) return value;
  if (value.startsWith("~/")) {
    // Double quotes expand `$` (and trigger backtick command substitution).
    // `$HOME` is prepended unescaped so it expands; every `$` in the body
    // (already stripped of `~/`) must be escaped or a `$VAR` segment would
    // interpolate. An anchor cannot be used here — the body was sliced.
    // Backslash is escaped FIRST: a later `\\"` would close the quoted
    // string early (the `\\` pair consumes itself, leaving the `"` bare).
    const escaped = value
      .slice(2)
      .replace(/\\/g, "\\\\")
      .replace(/`/g, "\\`")
      .replace(/"/g, '\\"')
      .replace(/\$/g, "\\$");
    return `"$HOME/${escaped}"`;
  }
  return `'${value.replace(/'/g, "'\\''")}'`;
}

// Single source of truth for the heredoc delimiter so a generated script can
// nest a print heredoc and a write heredoc without the markers drifting.
const HEREDOC_MARKER = "CHUNKHOUND_EOF";

// Plain-text scaffolding for a "write JSON file" shell block. Shared by the
// plain command builders and the HTML renderer so both stay byte-identical.
export interface JsonWriteScaffold {
  mkdir?: string;
  open: string;
  close: string;
}

export function jsonWriteScaffold(
  filename: string,
  platform: ConfiguratorPlatform,
): JsonWriteScaffold {
  const parentDir = getParentDir(filename);
  if (platform === "powershell") {
    return {
      ...(parentDir && {
        mkdir: `New-Item -ItemType Directory -Force -Path ${quotePowerShell(parentDir)} | Out-Null`,
      }),
      open: "@'",
      close: `'@ | Set-Content -Path ${quotePowerShell(filename)} -Encoding utf8`,
    };
  }
  return {
    ...(parentDir && { mkdir: `mkdir -p ${quotePosix(parentDir)}` }),
    open: `cat > ${quotePosix(filename)} <<'${HEREDOC_MARKER}'`,
    close: HEREDOC_MARKER,
  };
}

export function assembleJsonWrite(
  scaffold: JsonWriteScaffold,
  content: string,
): string {
  return [scaffold.mkdir, scaffold.open, content, scaffold.close]
    .filter(Boolean)
    .join("\n");
}

// Merge-guarded variant of the write scaffold. Editor MCP files can hold
// servers and settings the configurator does not own, and re-running setup is
// common, so an existing target must never be clobbered: the guard prints the
// merge payload instead of writing. Same byte-identity contract as above —
// the HTML renderer consumes these exact fields.
export interface GuardedJsonWriteScaffold {
  guardOpen: string;
  printOpen: string;
  message: string;
  printClose: string;
  elseLine: string;
  write: JsonWriteScaffold;
  guardClose: string;
}

export function guardedJsonWriteScaffold(
  filename: string,
  platform: ConfiguratorPlatform,
): GuardedJsonWriteScaffold {
  const message = `${filename} already exists — merge this block into it manually:`;
  const write = jsonWriteScaffold(filename, platform);
  if (platform === "powershell") {
    return {
      guardOpen: `if (Test-Path ${quotePowerShell(filename)}) {`,
      printOpen: "Write-Host @'",
      message,
      printClose: "'@",
      elseLine: "} else {",
      write,
      guardClose: "}",
    };
  }
  return {
    guardOpen: `if [ -f ${quotePosix(filename)} ]; then`,
    // A second quoted heredoc (not printf) keeps the payload byte-identical —
    // no escape interpretation, exactly like the write branch.
    printOpen: `cat <<'${HEREDOC_MARKER}'`,
    message,
    printClose: HEREDOC_MARKER,
    elseLine: "else",
    write,
    guardClose: "fi",
  };
}

export function assembleGuardedJsonWrite(
  scaffold: GuardedJsonWriteScaffold,
  content: string,
): string {
  // Heredoc/here-string bodies are literal, so the payload lines (and the
  // delimiters, without `<<-`) must stay at column 0 — no branch indentation.
  return [
    scaffold.guardOpen,
    scaffold.printOpen,
    scaffold.message,
    content,
    scaffold.printClose,
    scaffold.elseLine,
    assembleJsonWrite(scaffold.write, content),
    scaffold.guardClose,
  ].join("\n");
}
