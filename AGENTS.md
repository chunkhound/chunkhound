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
- ALWAYS Drop HNSW indexes for bulk inserts > 50 rows
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

## KNOWN_DEPRECATION_WARNINGS
**HDBSCAN + scikit-learn**: `force_all_finite` parameter warning
- Non-breaking, safe to ignore
- Waiting for upstream HDBSCAN fix
- Will break in sklearn 1.8 if not fixed upstream

## PROJECT_MAINTENANCE
- Smoke tests are mandatory guardrails
- All code patterns should be self-documenting
