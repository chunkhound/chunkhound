# Design: `index_unknown_files` Flag

**Date:** 2026-05-13
**Issue:** [#277 — Chunkhound no longer indexes unknown files in version 5](https://github.com/chunkhound/chunkhound/issues/277)
**Branch:** `unknown_files`

---

## Problem

In v5, files with unrecognized extensions (`Dockerfile`, `.proto`, `.feature`, etc.) are silently skipped at `batch_processor.py:224` when `language == Language.UNKNOWN`. Before v5 a bug accidentally indexed these as plain text — users found this useful. The fix removed the behavior; this design adds it back as an explicit, opt-in flag.

## Goal

A single flag that, when enabled:
1. Causes unknown-extension files to be **discovered** during directory scan
2. Causes them to be **parsed as plain text** (via `TextMapping`)
3. Skips files that are **binary** (null-byte heuristic) to avoid polluting the index

---

## Customer-Facing Interfaces

All three surfaces must expose the flag:

| Interface | Key / Flag | Default |
|---|---|---|
| CLI | `--index-unknown-files` | off (store_true) |
| `.chunkhound.json` | `"indexing": { "index_unknown_files": true }` | `false` |
| Environment variable | `CHUNKHOUND_INDEXING__INDEX_UNKNOWN_FILES=true` | unset |

---

## Architecture

### 1. `IndexingConfig` — new public field

**File:** `chunkhound/core/config/indexing_config.py`

Add to the public fields section (alongside `detect_embedded_sql`, `per_file_timeout_seconds`, etc.):

```python
index_unknown_files: bool = Field(
    default=False,
    description="Index files with unrecognized extensions as plain text",
)
```

### 2. Discovery — append `**/*` when flag is on

**File:** `chunkhound/core/config/indexing_config.py` — `model_validator(mode="after")`

When `index_unknown_files=True`, append `**/*` to the `include` list (if not already present). This ensures files like `Dockerfile` and `schema.graphql` are picked up during the directory walk. Existing `exclude` patterns (`node_modules`, `.git`, `.env`, etc.) continue to apply and filter them out.

The `**/*` pattern is appended regardless of whether the user has a custom or default `include` list.

### 3. Parsing — binary guard + TEXT fallback

**File:** `chunkhound/services/batch_processor.py`

Replace the current unconditional skip at line 224:

```python
# Current:
if language == Language.UNKNOWN:
    # ... skip with status="skipped"

# New:
if language == Language.UNKNOWN:
    if not config_dict.get("index_unknown_files"):
        # ... skip (unchanged behaviour when flag is off)
    else:
        # Binary guard: read first 8 KB, check for null bytes
        try:
            with open(file_path, "rb") as fh:
                sample = fh.read(8192)
            if b"\x00" in sample:
                # ... skip with status="skipped", error="binary_file"
                continue
        except OSError:
            # ... skip with status="error"
            continue
        language = Language.TEXT  # treat as plain text, fall through to normal parse path
```

No changes needed to `config_dict` propagation — `index_unknown_files` flows automatically via the existing `model_dump()` path.

### 4. CLI flag

**File:** `chunkhound/core/config/indexing_config.py` — `add_cli_arguments()`

```python
parser.add_argument(
    "--index-unknown-files",
    action="store_true",
    default=False,
    dest="index_unknown_files",
    help="Index files with unrecognized extensions as plain text (binary files are still skipped)",
)
```

This is picked up automatically by all three commands that call `add_config_arguments(..., ["indexing", ...])`: `index`, `mcp`, `_daemon`.

### 5. Environment variable

**File:** `chunkhound/core/config/indexing_config.py` — `load_from_env()`

```python
if val := os.getenv("CHUNKHOUND_INDEXING__INDEX_UNKNOWN_FILES"):
    overrides["index_unknown_files"] = val.lower() in ("1", "true", "yes")
```

---

## Data Flow

```
Directory scan
  └─ include patterns  ← **/* appended when flag=on
       └─ exclude patterns (unchanged)
            └─ discovered files
                 └─ batch_processor.py
                      ├─ Language.UNKNOWN + flag=off  → skip (status="skipped")
                      ├─ Language.UNKNOWN + flag=on + binary  → skip (status="skipped", error="binary_file")
                      ├─ Language.UNKNOWN + flag=on + text  → Language.TEXT → TextMapping parse
                      └─ Language.known  → normal parse path (unchanged)
```

---

## Error Handling

- `OSError` reading the 8 KB sample → skip with `status="error"`
- Binary detection is conservative: any null byte in the first 8 KB is treated as binary
- Log message on text path: `logger.debug("Indexing unknown file as text: {}", file_path)`
- Log message on binary skip: `logger.debug("Skipping binary file: {}", file_path)`

---

## Testing

- Unit test in `tests/` asserting:
  - `Dockerfile` (text) is indexed when flag=on, skipped when flag=off
  - A binary file (contains null bytes) is skipped even when flag=on
  - The `**/*` pattern is appended to include list when flag=on
  - CLI `--index-unknown-files` sets the field to `True`
  - Env var `CHUNKHOUND_INDEXING__INDEX_UNKNOWN_FILES=true` sets the field to `True`
- Add to smoke tests: index a directory containing an unknown-extension text file and verify it appears in results

---

## Out of Scope

- Adding new extensions to the `EXTENSION_TO_LANGUAGE` map (separate concern)
- Detecting file encoding (UTF-8 vs Latin-1) — `TextMapping` already handles this gracefully
- MCP tool parameters — indexing config is set at server startup, not per-tool-call
