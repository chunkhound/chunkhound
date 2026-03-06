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

## DB_PATH_GOTCHAS
- Default DB path: `.chunkhound/db/chunks.db` (directory structure, not flat file)
- When using `--db` flag, pass the **directory** path (e.g. `--db .chunkhound/db`), not the full file path — passing `--db .../chunks.db` creates a nested `chunks.db/chunks.db` directory
- Old-style flat `.chunkhound` files (pre-v4) block directory creation — move aside before re-indexing
- Project-local `.chunkhound.json` with relative `"path": ".chunkhound"` resolves to CWD, not the project dir — use `--db` with absolute paths when indexing remote projects
- `--config` does NOT override a project-local `.chunkhound.json` for DB path — always use explicit `--db` when the target project has its own config

## PROJECT_MAINTENANCE
- Smoke tests are mandatory guardrails
- Run `uv run mypy chunkhound` during reviews to catch Optional/type boundary issues
- All code patterns should be self-documenting
