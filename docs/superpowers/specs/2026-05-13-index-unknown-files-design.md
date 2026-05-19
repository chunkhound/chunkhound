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

Tests are split into two phases: **before fix** (written first, verify current behaviour is preserved) and **after fix** (written first as failing tests that drive the implementation).

### Before fix — regression guard (must pass now and after)

These confirm the default "flag-off" path is untouched by the implementation.

**File:** `tests/test_index_unknown_files.py`

```python
# Verify unknown files are skipped when flag=off (existing behaviour)
def test_unknown_file_skipped_by_default(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12\n")
    config = IndexingConfig()                         # flag defaults to False
    assert "**/*" not in config.include               # no wildcard injected
    results = process_file_batch([dockerfile], config.model_dump())
    assert results[0].status == "skipped"
    assert results[0].error == "Unknown file type"

# Verify known-extension files are unaffected
def test_known_extension_unaffected_when_flag_off(tmp_path):
    pyfile = tmp_path / "main.py"
    pyfile.write_text("x = 1\n")
    config = IndexingConfig()
    results = process_file_batch([pyfile], config.model_dump())
    assert results[0].status != "skipped"

# Verify **/* is NOT added to include when flag=off
def test_wildcard_not_injected_when_flag_off():
    config = IndexingConfig()
    assert "**/*" not in config.include
```

### After fix — new behaviour (failing before implementation, passing after)

**File:** `tests/test_index_unknown_files.py` (same file, separate class)

```python
# Text unknown file is indexed when flag=on
def test_unknown_text_file_indexed_when_flag_on(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12\nRUN pip install uv\n")
    config = IndexingConfig(index_unknown_files=True)
    results = process_file_batch([dockerfile], config.model_dump())
    assert results[0].status == "parsed"
    assert len(results[0].chunks) > 0

# Binary unknown file is still skipped even when flag=on
def test_binary_unknown_file_skipped_when_flag_on(tmp_path):
    binfile = tmp_path / "model.bin"
    binfile.write_bytes(b"\x00\x01\x02" * 100)
    config = IndexingConfig(index_unknown_files=True)
    results = process_file_batch([binfile], config.model_dump())
    assert results[0].status == "skipped"
    assert results[0].error == "binary_file"

# **/* is appended to include when flag=on
def test_wildcard_injected_when_flag_on():
    config = IndexingConfig(index_unknown_files=True)
    assert "**/*" in config.include

# **/* is appended even when user has custom include list
def test_wildcard_appended_to_custom_include():
    config = IndexingConfig(index_unknown_files=True, include=["**/*.py"])
    assert "**/*" in config.include
    assert "**/*.py" in config.include

# CLI flag sets the field
def test_cli_flag_sets_index_unknown_files():
    parser = argparse.ArgumentParser()
    IndexingConfig.add_cli_arguments(parser)
    args = parser.parse_args(["--index-unknown-files"])
    assert args.index_unknown_files is True

# Env var sets the field
def test_env_var_sets_index_unknown_files(monkeypatch):
    monkeypatch.setenv("CHUNKHOUND_INDEXING__INDEX_UNKNOWN_FILES", "true")
    config = IndexingConfig.load_from_env()
    assert config.index_unknown_files is True

# JSON config sets the field
def test_json_config_sets_index_unknown_files():
    config = IndexingConfig.model_validate({"index_unknown_files": True})
    assert config.index_unknown_files is True
```

### Smoke test addition

**File:** `tests/test_smoke.py` — add to the existing `TestIndexing` class (or equivalent):

```python
def test_index_unknown_extension_file(tmp_path):
    """End-to-end: unknown-extension text file appears in results when flag=on."""
    feature_file = tmp_path / "login.feature"
    feature_file.write_text("Feature: Login\n  Scenario: Valid user\n")
    config_file = tmp_path / ".chunkhound.json"
    config_file.write_text('{"indexing": {"index_unknown_files": true}}')
    # Run indexer, query for "Login", assert at least one result references login.feature
```

---

## Out of Scope

- Adding new extensions to the `EXTENSION_TO_LANGUAGE` map (separate concern)
- Detecting file encoding (UTF-8 vs Latin-1) — `TextMapping` already handles this gracefully
- MCP tool parameters — indexing config is set at server startup, not per-tool-call
