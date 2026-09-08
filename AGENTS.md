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
- NEVER Modify `AGENTS.md` without an explicit user request and an explicit approval

**ALWAYS:**
- ALWAYS Run smoke tests before committing: `uv run pytest tests/test_smoke.py -v -n auto`
- ALWAYS Run full test suite before pushing to a PR: `uv run pytest tests/ -v`
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
full:      uv run pytest tests/ -v                     # MANDATORY before pushing to a PR
format:    uv run ruff format chunkhound

# Running
index:     uv run chunkhound index [directory]
mcp_stdio: uv run chunkhound mcp
mcp_http:  uv run chunkhound mcp --transport http --port 5173

# Git diff search
git_search: uv run chunkhound search "<query>" --last-n <N>
git_range:  uv run chunkhound search "<query>" --commit-range <range>
git_hash:   uv run chunkhound search "<query>" --commit-hash <hash>
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

## RUST_RULES
**NEVER:**
- NEVER write `unsafe` code — `#![forbid(unsafe_code)]` is set at the crate root; the compiler will reject it
- NEVER add `#[allow(clippy::...)]` without an inline comment explaining why
- NEVER use `.unwrap()` at the PyO3 boundary — use `?` or `PyErr::new`; `.expect("reason")` is acceptable for truly-unreachable internal invariants
- NEVER borrow `&str` across `py.allow_threads()` — convert to owned `String` before the GIL is released

**ALWAYS:**
- ALWAYS wrap CPU/IO-bound work in `py.allow_threads(|| { ... })` to release the GIL during Rust execution
- ALWAYS run `cargo fmt` and `cargo clippy --all-targets -- -D warnings` before committing Rust changes (`make rust-check`)
- ALWAYS run `cargo test` after Rust changes (`make rust-test`)
- ALWAYS use owned types (`String`, `Vec<T>`) at the `allow_threads` boundary

## PROJECT_MAINTENANCE
- Smoke tests are mandatory guardrails
- Run `uv run mypy chunkhound` during reviews to catch Optional/type boundary issues
- All code patterns should be self-documenting

## TESTING_PHILOSOPHY
- Test external constraints, critical invariants, and user-facing contracts.
- Do NOT write tests for adapters, private helpers, mock behavior, or internal plumbing unless the test is the narrowest way to protect a real external contract.
- If a refactor could change the implementation without changing user-visible behavior, the test is probably too internal and should not exist.
- Prefer contract names like `test_cli_overrides_env` over implementation names like `test_extract_cli_overrides_calls_helper`.
- Use real business logic with fakes only at true external boundaries (network, filesystem, subprocess, third-party APIs).
- For provider integrations, test our contract with the provider: supported/unsupported feature gating, request validity constraints, explicit failures, and stable user-visible semantics. Do NOT test SDK mechanics or mirror every internal request-shaping helper.
- Before adding a test, ask: "Would a user, caller, CI contract, or external system notice if this broke?" If not, do not add the test.
- Prefer one higher-value contract test over many narrow implementation tests.
