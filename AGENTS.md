# ChunkHound LLM Context

## PROJECT_IDENTITY
ChunkHound: Semantic and regex search tool for codebases with MCP integration
Built: 100% by AI agents - NO human-written code
Purpose: Transform codebases into searchable knowledge bases for AI assistants

## MODIFICATION_RULES
**NEVER:**
- NEVER Use print() in MCP server (stdio.py, http_server.py, tools.py)
- NEVER Make single-row DB inserts in loops
- NEVER Use forward references (quotes) in type annotations unless needed

**ALWAYS:**
- ALWAYS Run smoke tests before committing: `uv run pytest tests/test_smoke.py`
- ALWAYS Batch embeddings (min: 100, max: provider_limit)
- ALWAYS Use uv for all Python operations
- ALWAYS Update version via: `uv run scripts/update_version.py`

## KEY_COMMANDS
```bash
# Development
lint:      uv run ruff check chunkhound
typecheck: uv run mypy chunkhound
test:      uv run pytest
smoke:     uv run pytest tests/test_smoke.py -v -n auto  # MANDATORY before commits
format:    uv run ruff format chunkhound

# Running
index:     uv run chunkhound index [directory]
mcp_stdio: uv run chunkhound mcp
mcp_http:  uv run chunkhound mcp http --port 5173
```

## VERSION_MANAGEMENT
Dynamic versioning via hatch-vcs - version derived from git tags.

```bash
# Create release
uv run scripts/update_version.py 4.1.0

# Create pre-release
uv run scripts/update_version.py 4.1.0b1
uv run scripts/update_version.py 4.1.0rc1

# Bump version
uv run scripts/update_version.py --bump minor      # v4.0.1 → v4.1.0
uv run scripts/update_version.py --bump minor b1   # v4.0.1 → v4.1.0b1
```

NEVER manually edit version strings - ALWAYS create git tags instead.

## PUBLISHING_PROCESS
```bash
# 1. Create version tag
uv run scripts/update_version.py X.Y.Z

# 2. Run smoke tests (MANDATORY)
uv run pytest tests/test_smoke.py -v

# 3. Prepare release
./scripts/prepare_release.sh

# 4. Test local install
pip install dist/chunkhound-X.Y.Z-py3-none-any.whl

# 5. Push tag
git push origin vX.Y.Z

# 6. Publish
uv publish
```

## PROJECT_MAINTENANCE
- Smoke tests are mandatory guardrails
- Run `uv run mypy chunkhound` during reviews to catch Optional/type boundary issues
- All code patterns should be self-documenting

## ARCHITECTURE

### Core Pipeline

```
CLI (api/cli/main.py) / MCP Server (mcp_server/)
  └→ create_services() [database_factory.py] — single factory for all entry points
      └→ ProviderRegistry [registry/] — DI container / composition root
          ├→ DatabaseProvider (DuckDB or LanceDB)
          ├→ EmbeddingProvider (VoyageAI, OpenAI, or Ollama)
          ├→ LazyLanguageParsers (materialize on first use per language)
          └→ DatabaseServices bundle: {provider, indexing_coordinator, search_service, embedding_service}
```

### Indexing (`services/indexing_coordinator.py`)

```
File Discovery (utils/file_patterns.py, respects .gitignore/.chunkhoundignore)
  → Parallel Parsing (ProcessPoolExecutor, 'spawn' mode — NOT fork)
    → Each file: detect_language() → UniversalParser.parse_file()
      → TreeSitterEngine → ConceptExtractor → ChunkSplitter (cAST algorithm)
  → Smart Chunk Diffing (ChunkCacheService — preserves embeddings for unchanged chunks)
  → Batch DB Insert (SerialDatabaseProvider, batch size 5000)
  → Batch Embedding Generation (async, concurrent batches via EmbeddingProvider)
```

### Search (`services/search_service.py`)

- **Semantic**: Query → embed → HNSW vector index → ranked results
  - `SingleHopStrategy` — standard vector similarity
  - `MultiHopStrategy` — iterative expansion with reranking (auto-selected when reranker available)
- **Regex**: DuckDB regex on chunk content (no API keys needed)
- **Deep research** (`services/research/`): BFS exploration using LLM-generated follow-up queries, then synthesis

### MCP Server (two modes)

- **Stdio** (`mcp_server/stdio.py`): One process per client. All logging disabled to protect JSON-RPC.
- **Daemon** (`daemon/server.py`): Multi-client, owns sole DuckDB connection. IPC via Unix sockets (Linux/macOS) or TCP loopback (Windows). `ClientProxy` bridges stdio↔IPC.

### Parser System (`parsers/`)

- `UniversalParser` = `TreeSitterEngine` + `ConceptExtractor` + `ChunkSplitter`
- **cAST algorithm**: Aligns chunks to AST node boundaries, greedily merges siblings up to size limits (1200 non-whitespace chars / 6000 tokens)
- 5 universal concepts: DEFINITION, BLOCK, COMMENT, IMPORT, STRUCTURE
- 30+ language mappings in `parsers/mappings/`
- Embedded SQL detection (`parsers/embedded_sql_detector.py`) extracts SQL from string literals

### Key Interfaces (`interfaces/`)

All use Python `Protocol` (structural typing):
- `DatabaseProvider` — full DB interface (CRUD, search, vector index, transactions)
- `EmbeddingProvider` — async embed/rerank with provider metadata
- `LLMProvider` — ABC for research (utility vs synthesis providers via `LLMManager`)
- `LanguageParser` — parse_file/parse_content

### Database Layer

- **DuckDB** (`providers/database/duckdb_provider.py`): 3 tables (files, chunks, embeddings). HNSW index for vectors. All ops through `SerialDatabaseExecutor` (1-worker ThreadPoolExecutor) enforcing single-writer.
- **LanceDB** (`providers/database/lancedb_provider.py`): Alternative with vectors embedded in chunk rows.
- DB path: `<project>/.chunkhound/db/chunks.db`

### Configuration (`core/config/`)

Precedence: CLI args > env vars (`CHUNKHOUND_` prefix, `__` nested separator) > `.chunkhound.json` > defaults. `Config` is a Pydantic model with sub-configs: database, embedding, llm, mcp, indexing, research.

### Additional Modules

- **code_mapper** (`code_mapper/`): LLM-driven structured code documentation with HyDE embeddings
- **autodoc** (`autodoc/`): Generates Astro static documentation sites from code maps
- **Realtime indexing** (`services/realtime_indexing_service.py`): Watchdog file watcher → async queue → incremental IndexingCoordinator updates
