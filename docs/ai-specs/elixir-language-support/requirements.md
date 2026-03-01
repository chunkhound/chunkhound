# Requirements Document

## Introduction

Add Elixir language support to ChunkHound's Tree-sitter parsing pipeline, enabling semantic and regex search across `.ex` and `.exs` codebases. Elixir's AST uniquely represents all constructs as `call` nodes — the parser must pattern-match on identifier text (e.g., `defmodule`, `def`, `defmacro`) to classify chunks.

## Glossary

- **Call node**: The universal Tree-sitter AST node type in Elixir — every keyword (`def`, `defmodule`, `use`, etc.) is a `call` with its keyword as the `target` identifier
- **Mapping**: A `BaseMapping` subclass that defines Tree-sitter queries for each concept (DEFINITION, COMMENT, IMPORT, etc.)
- **Chunk**: A semantically meaningful code fragment extracted by the parser, tagged with a `ChunkType`
- **Metadata kind**: A string label (e.g., `"function"`, `"class"`, `"macro"`) attached to chunks for fine-grained classification

## Requirements

### 1. Elixir File Detection

**User Story:** As a developer, I want ChunkHound to recognize `.ex` and `.exs` files as Elixir, so that Elixir codebases are automatically indexed.

#### Acceptance Criteria

- 1.1 WHEN a file with extension `.ex` is encountered THEN the system SHALL identify it as `Language.ELIXIR`
- 1.2 WHEN a file with extension `.exs` is encountered THEN the system SHALL identify it as `Language.ELIXIR`
- 1.3 WHEN `Language.ELIXIR` is queried THEN the system SHALL report `is_programming_language = True`

### 2. Module Parsing

**User Story:** As a developer, I want ChunkHound to extract Elixir modules and protocols as named chunks, so that I can search for module definitions.

#### Acceptance Criteria

- 2.1 WHEN a `defmodule` call is parsed THEN the system SHALL emit a chunk with `ChunkType.MODULE` and `kind="class"`
- 2.2 WHEN a `defprotocol` call is parsed THEN the system SHALL emit a chunk with `ChunkType.INTERFACE` and `kind="interface"`
- 2.3 WHEN a `defimpl` call is parsed THEN the system SHALL emit a chunk with `ChunkType.CLASS` and `kind="class"`
- 2.4 WHEN extracting the module name THEN the system SHALL capture the alias (e.g., `MyApp.Accounts.User`)

### 3. Function and Macro Parsing

**User Story:** As a developer, I want ChunkHound to extract functions, macros, guards, and delegates as named chunks, so that I can search for callable definitions.

#### Acceptance Criteria

- 3.1 WHEN a `def` or `defp` call is parsed THEN the system SHALL emit a chunk with `ChunkType.FUNCTION` and `kind="function"`
- 3.2 WHEN a `defmacro` or `defmacrop` call is parsed THEN the system SHALL emit a chunk with `ChunkType.MACRO` and `kind="macro"`
- 3.3 WHEN a `defguard` or `defguardp` call is parsed THEN the system SHALL emit a chunk with `ChunkType.FUNCTION` and `kind="function"`
- 3.4 WHEN a `defdelegate` call is parsed THEN the system SHALL emit a chunk with `ChunkType.FUNCTION` and `kind="function"`
- 3.5 WHEN extracting function names THEN the system SHALL resolve the name from the nested `arguments > call > identifier` AST structure

### 4. Type and Spec Parsing

**User Story:** As a developer, I want ChunkHound to extract `@spec`, `@type`, `@typep`, `@opaque`, and `@callback` attributes as typed chunks, so that I can search for type definitions.

#### Acceptance Criteria

- 4.1 WHEN a `@spec` attribute is parsed THEN the system SHALL emit a chunk with `ChunkType.TYPE` and `kind="type"`
- 4.2 WHEN a `@type`, `@typep`, or `@opaque` attribute is parsed THEN the system SHALL emit a chunk with `ChunkType.TYPE` and `kind="type"`
- 4.3 WHEN a `@callback` attribute is parsed THEN the system SHALL emit a chunk with `ChunkType.TYPE` and `kind="type"`

### 5. Import and Dependency Parsing

**User Story:** As a developer, I want ChunkHound to extract `use`, `import`, `alias`, and `require` calls as import chunks, so that I can search for module dependencies.

#### Acceptance Criteria

- 5.1 WHEN a `use`, `import`, `alias`, or `require` call is parsed THEN the system SHALL emit a chunk with `ChunkType.IMPORT`
- 5.2 WHEN extracting the import target THEN the system SHALL capture the module alias (e.g., `Ecto.Changeset`)

### 6. Comment and Documentation Parsing

**User Story:** As a developer, I want ChunkHound to extract comments and `@doc`/`@moduledoc` attributes as comment chunks, so that documentation is searchable.

#### Acceptance Criteria

- 6.1 WHEN a `# comment` line is parsed THEN the system SHALL emit a chunk with `ChunkType.COMMENT`
- 6.2 WHEN a `@doc` or `@moduledoc` attribute is parsed THEN the system SHALL emit a chunk with `ChunkType.COMMENT`

### 7. Smoke Test Coverage

**User Story:** As a maintainer, I want Elixir included in the smoke test suite, so that parser regressions are caught before release.

#### Acceptance Criteria

- 7.1 WHEN smoke tests run THEN the system SHALL include a `Language.ELIXIR` sample that parses without error
- 7.2 WHEN the Elixir parser is invoked THEN the system SHALL produce at least one chunk from a valid `.ex` file
