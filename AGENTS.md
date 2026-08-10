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

## PUBLISHING_PROCESS
Releases are now fully automated via GitHub Actions (OIDC Trusted Publishing).
See **RELEASING.md** for the authoritative step-by-step guide.

Quick summary:
1. Tag the version: `uv run scripts/update_version.py X.Y.Z`
2. Run smoke tests: `uv run pytest tests/test_smoke.py -v -n auto` (MANDATORY)
3. Create and publish a GitHub Release — `release.yml` handles the PyPI upload automatically.

Pre-releases (alpha/beta/RC) publish to **PyPI** (not TestPyPI) via `release-rc.yml` on tag push.
Do NOT use `uv publish` or `prepare_release.sh` manually — CI owns the publish step.

## TEST RELEASE (alpha to PyPI)

**If version not specified:** fetch latest version from PyPI, increment minor, append `a1`:
```bash
LATEST=$(pip index versions chunkhound 2>/dev/null | grep -oP '[\d.]+' | head -1)
# e.g. 4.0.3 → next minor = 4.1.0 → alpha = 4.1.0a1
```

**If version specified by user** (e.g. `4.2.0`): append `a1` → `4.2.0a1`

**Steps:**
```bash
# 1. Save current remote and switch to chunkhound org remote
ORIGINAL_REMOTE=$(git remote get-url origin)
git remote set-url origin https://github.com/chunkhound/chunkhound.git

# 2. Create the alpha tag
uv run scripts/update_version.py X.Y.Za1

# 3. Push the tag — triggers release-rc.yml → publishes to PyPI as pre-release
git push origin vX.Y.Za1

# 4. Revert remote back to original
git remote set-url origin "$ORIGINAL_REMOTE"

# 5. Update uv.lock — pyproject.toml only pins a floor version, so this must be
#    bumped every release to pick up the version that was just published
uv lock --upgrade-package chunkhound-native
git add uv.lock
git commit -m "chore: bump chunkhound-native in lockfile to vX.Y.Za1"
```

PyPI trusted publisher required for `release-rc.yml` (on the `chunkhound` project):
- Owner: `chunkhound`
- Repository: `chunkhound`
- Workflow: `release-rc.yml`
- Environment: `pypi`

This same tag push also publishes `chunkhound-native` via the `publish-rc-native` job, which needs
its own trusted publisher registered on the **`chunkhound-native`** PyPI project (Workflow:
`release-rc.yml`, Environment: `pypi-native`) — see `RELEASING.md` prerequisites for the full
setup and the environment-scoping gotcha that causes a confusing `403` if it's misconfigured.

## DB_PATH_GOTCHAS
- **Preferred: pass project directory as positional arg** — `chunkhound search "query" /path/to/project` — this reads `.chunkhound.json` and resolves the DB correctly
- **For MCP:** `chunkhound mcp --db /path/to/project/.chunkhound` (the path from `.chunkhound.json`'s `database.path`)
- **`--db` with wrong subpath silently returns 0 results** — no error, just empty. Always verify with a regex search first.
- `--db` accepts either a directory (uses `.../chunks.db` internally) or an explicit file path (`.db` / `.duckdb` extension returned as-is)
- Default DB path: `.chunkhound/db/chunks.db` (directory structure, not flat file)
- Old-style flat `.chunkhound` files (pre-v4) block directory creation — move aside before re-indexing
- Project-local `.chunkhound.json` with relative `"path": ".chunkhound"` resolves to CWD, not the project dir — use `--db` with absolute paths when indexing remote projects
- `--config` does NOT override a project-local `.chunkhound.json` for DB path — always use explicit `--db` when the target project has its own config

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

## RUST_COMMANDS
```bash
rust-check: make rust-check   # cargo fmt --check + clippy -D warnings
rust-test:  make rust-test    # cargo test

# Build the native extension (required before running tests that import chunkhound_native)
#
# DuckDB is linked dynamically on every platform (Linux/macOS/Windows), and the
# extension resolves it relative to its own location -- not via an absolute
# build-machine path or an external environment variable like LD_LIBRARY_PATH.
# Same recipe everywhere:
#
#   1. Download: DUCKDB_DOWNLOAD_LIB=1 fetches the official precompiled shared
#      library from GitHub (unchanged from before).
#   2. Link: RUSTFLAGS bakes in a self-relative RPATH at link time --
#      -Wl,-rpath,$ORIGIN on Linux, -Wl,-rpath,@loader_path on macOS (Windows
#      has no RPATH concept; see chunkhound_native/__init__.py's
#      os.add_dll_directory() guard instead).
#   3. Copy: scripts/copy_duckdb_runtime.py places the downloaded library next
#      to the compiled extension (source tree + site-packages), since RPATH
#      only helps if something is actually there to find.
#
#   Linux:
#     DUCKDB_DOWNLOAD_LIB=1 RUSTFLAGS='-C link-arg=-Wl,-rpath,$ORIGIN' uv run maturin develop
#     uv run python scripts/copy_duckdb_runtime.py
#
#   macOS:
#     DUCKDB_DOWNLOAD_LIB=1 RUSTFLAGS='-C link-arg=-Wl,-rpath,@loader_path' uv run maturin develop
#     uv run python scripts/copy_duckdb_runtime.py
#
#   Windows (no RUSTFLAGS needed):
#     $env:DUCKDB_DOWNLOAD_LIB = "1"; uv run maturin develop
#     uv run python scripts/copy_duckdb_runtime.py
#
# Published wheels get the same bundling via a wheel-repair step in CI
# (auditwheel/delocate/delvewheel — see release.yml/release-rc.yml), since
# those tools only operate on built wheels, not maturin develop's editable
# installs.
#
# chunkhound-native is a hard dependency of the main chunkhound package, so
# `pip install chunkhound` only works on platforms with a published
# chunkhound-native wheel. Currently covered: macOS arm64, Linux x86_64
# (manylinux_2_34), Linux aarch64 (manylinux_2_34, via ubuntu-24.04-arm),
# Windows x86_64. Known, deliberate gaps (not oversights):
#   - Intel macOS (x86_64-apple-darwin): needs either a paid GitHub "larger
#     runner" (macos-13's free tier is retired; Apple/GitHub sunset Intel
#     macOS runners entirely in Fall 2027) or cross-compiling from the arm64
#     runner, which hits an open, unresolved PyO3 issue (framework linking,
#     e.g. CoreFoundation not found, when targeting x86_64-apple-darwin from
#     an arm64 host). Not worth either cost given the shrinking Intel Mac
#     install base and the 2027 sunset — revisit only if that changes.
#   - musl/Alpine and pre-manylinux_2_34 glibc (Ubuntu 20.04, RHEL/CentOS 8,
#     Amazon Linux 2): would need a manylinux-container-based build instead
#     of the current native-runner build; not yet done.
#   - No sdist: chunkhound-native has no source fallback, so any platform
#     without a matching wheel above hard-fails `pip install chunkhound`
#     with no degraded install path.
#
# Air-gapped / no internet (Linux only, untested since this rewrite): the
# download step above needs GitHub access. libduckdb-sys still supports
# linking against a static .a you already have via DUCKDB_LIB_DIR +
# DUCKDB_STATIC=1 instead of the download+dynamic-link recipe above:
#   DUCKDB_LIB_DIR=<dir with libduckdb_static.a> DUCKDB_STATIC=1 \
#     RUSTFLAGS="-C link-arg=-lstdc++" uv run maturin develop --release
#   (-lstdc++ is required: the static .a's C++ exception-handling code lives
#   in libstdc++.so; without this flag the .so builds but fails at Python
#   import with "undefined symbol: __gxx_personality_v0".)
```

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
