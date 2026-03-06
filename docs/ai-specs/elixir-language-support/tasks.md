# Implementation Plan

## Phase 1: Infrastructure ✅

- [x] 1 Add tree-sitter-elixir dependency
  - Add `"tree-sitter-elixir>=0.3.4"` to `pyproject.toml` dependencies
  - Run `uv sync` to install
  - _Requirements: 1.1, 1.2_
  - _Agent: Sonnet_

- [x] 2 Register Elixir in Language enum and parser factory
  - _Requirements: 1.1, 1.2, 1.3_
  - _Agent: Sonnet_

- [x] 2.1 Add ELIXIR to Language enum
  - Add `ELIXIR = "elixir"` to `Language` enum in `common.py` (alphabetical, after DART)
  - Add `.ex` and `.exs` to `extension_map` in `from_file_extension()`
  - Add `Language.ELIXIR` to `is_programming_language` property set
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 2.2 Register in parser factory
  - Add `import tree_sitter_elixir as ts_elixir` (direct import, like Haskell)
  - Add `ElixirMapping` to mapping imports
  - Add `Language.ELIXIR: LanguageConfig(ts_elixir, ElixirMapping, True, "elixir")` to `LANGUAGE_CONFIGS`
  - Add `.ex` and `.exs` to `EXTENSION_TO_LANGUAGE`
  - _Requirements: 1.1, 1.2_

- [x] 2.3 Export from mappings __init__.py
  - Add `from .elixir import ElixirMapping` import
  - Add `"ElixirMapping"` to `__all__`
  - _Requirements: 1.1_

- [x] 3 Checkpoint - Verify import resolves and uv sync succeeds

## Phase 2: Core Parser ✅

- [x] 4 Create ElixirMapping
  - Create `chunkhound/parsers/mappings/elixir.py` with full concept query and extraction logic
  - Use `lua.py` as structural template
  - _Requirements: 2.1-2.4, 3.1-3.5, 4.1-4.3, 5.1-5.2, 6.1-6.2_
  - _Agent: Opus_

- [x] 4.1 Implement DEFINITION concept query
  - Match `call` nodes with `#match?` on target identifiers for: `defmodule`, `defprotocol`, `defimpl`, `def`, `defp`, `defmacro`, `defmacrop`, `defguard`, `defguardp`, `defdelegate`, `defstruct`
  - Match `unary_operator` with `@` for: `@spec`, `@type`, `@typep`, `@opaque`, `@callback`
  - **Property 2: Keyword-to-ChunkType Mapping**
  - **Validates: Requirements 2.1-2.3, 3.1-3.4, 4.1-4.3**

- [x] 4.2 Implement COMMENT concept query
  - Match `(comment)` nodes for `# line comments`
  - Match `unary_operator` with `@doc`/`@moduledoc` targets
  - **Property 4: Attribute Recognition**
  - **Validates: Requirements 6.1, 6.2**

- [x] 4.3 Implement IMPORT concept query
  - Match `call` nodes with `#any-of?` on target for: `use`, `import`, `alias`, `require`
  - **Validates: Requirements 5.1, 5.2**

- [x] 4.4 Implement BLOCK and STRUCTURE concept queries
  - BLOCK: match `(do_block)` nodes
  - STRUCTURE: match top-level `(source)` node

- [x] 4.5 Implement extract_name with nested function name resolution
  - Navigate `arguments → first call child → target identifier` for function names
  - Extract module alias for `defmodule` (e.g., `MyApp.Accounts.User`)
  - Handle zero-arity functions (identifier instead of nested call)
  - **Property 3: Name Extraction Accuracy**
  - **Validates: Requirements 3.5, 2.4**

- [x] 4.6 Implement extract_metadata with kind classification
  - Map keywords to `kind` values: `defmodule→"class"`, `def/defp→"function"`, `defmacro→"macro"`, `defprotocol→"interface"`, `@spec/@type→"type"`
  - **Property 2: Keyword-to-ChunkType Mapping**
  - **Validates: Requirements 2.1-2.3, 3.1-3.4, 4.1-4.3**

- [x] 4.7 Implement remaining BaseMapping methods
  - `get_function_query()`, `get_class_query()`, `get_comment_query()`
  - `extract_function_name()`, `extract_class_name()`
  - `extract_content()`, `clean_comment_text()`, `resolve_import_paths()`

- [x] 5 Checkpoint - Verify ElixirMapping imports and instantiates without error

## Phase 3: Testing ✅

- [x] 6 Create test fixture and parser tests
  - _Requirements: 7.1, 7.2_
  - _Agent: Sonnet_

- [x] 6.1 Create comprehensive Elixir fixture
  - Create `tests/fixtures/elixir/comprehensive.ex`
  - Include: GenServer module, protocol + impl, LiveView module, pipe chains, multi-clause functions, guards, delegates, type specs, doc attributes
  - _Requirements: 7.2_

- [x] 6.2 Create parser unit tests
  - Create `tests/test_elixir_parser.py`
  - Test file detection (`.ex`, `.exs` → `Language.ELIXIR`)
  - Test module parsing (`defmodule`, `defprotocol`, `defimpl`)
  - Test function parsing (`def`, `defp`, `defmacro`, `defguard`, `defdelegate`) with name accuracy
  - Test type/spec parsing (`@spec`, `@type`, `@callback`)
  - Test import parsing (`use`, `import`, `alias`, `require`)
  - Test comment parsing (`# comments`, `@doc`, `@moduledoc`)
  - **Property 1: File Extension Recognition**
  - **Validates: Requirements 1.1-1.3, 2.1-2.4, 3.1-3.5, 4.1-4.3, 5.1-5.2, 6.1-6.2**

- [x] 6.3 Add Elixir to smoke tests
  - Add `Language.ELIXIR` sample to `tests/test_smoke.py`
  - **Validates: Requirements 7.1, 7.2**

- [x] 7 Checkpoint - Full verification suite
  - `uv run pytest tests/test_elixir_parser.py -v` — 20/20 passed
  - `uv run pytest tests/test_smoke.py -v -n auto` — 17/17 passed
  - `uv run ruff check chunkhound/parsers/mappings/elixir.py` — clean
  - `uv run mypy chunkhound/parsers/mappings/elixir.py` — 2 unreachable warnings (same as lua.py, pre-existing pattern)
