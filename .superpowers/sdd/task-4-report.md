# Plan 2 Task 4 report — pull consent metadata and single validated fetch

## Outcome
Implemented and committed as `35d01b97f822b210022d137aead8d8b2608aac5f` (`feat: show sanitized hub pull consent details`). No push performed. Worktree clean after commit.

## Files changed
- `chunkhound/api/cli/commands/pull.py`
  - Plain trusted and `--trust` pull both fetch one `ValidatedHubSource` and pass it through as `source=` to `pull()`, avoiding refetch.
  - Captures and verifies checkout and manifest URL identity after fetch, before granting authorization or selecting/replacing.
  - `--trust` keeps pre-client fatal logging active across fetch and identity verification; it attempts to persist the grant before selection, then switches to client-fatal semantics.
  - Selects manifest before installation and emits INFO consent metadata with `logger.bind(hub_client=True)`, exposing only sanitized origin and selected provider/model/dimensions. URL userinfo/path/query are excluded; IPv6 is bracketed and default ports omitted.
  - Trust continues to be read/granted only through the central trust APIs; declaration selection uses configured embedding identity and introduces no declaration/environment trust.
- `tests/hub/test_activation_cli.py`
  - Added parsed valid manifest fixture; adapted mocked pull tests for validated-source delivery.
  - Added consent metadata/redaction, userinfo-before-request, and IPv6-origin contract coverage.

## TDD evidence
- Wrote/updated tests before production edits.
- RED command: `uv run --no-sync pytest tests/hub/test_activation_cli.py -k "consent or userinfo or trusted_plain or attempted_explicit or already_trusted" -q -p no:cacheprovider` — failed as expected: trusted plain pull did not fetch source, consent output was absent, and IPv6 consent log was absent (8 passed, 3 failed at the point new tests were included).
- Implemented behavior, then corrected legacy mock setups/identity checks found by wider tests.
- Focused recheck: `uv run --no-sync pytest tests/hub/test_activation_cli.py -k "attempted_explicit_pull_cancellation or captured_identity or unknown_dimensions or consent or userinfo or trusted_plain" -q -p no:cacheprovider` — 11 passed, 70 deselected.

## Test commands and results
- Initial required RED subset: failed expected old behavior (9 selected; one existing trusted test failed before the consent tests were added).
- RED subset after adding requested consent tests: 12 selected; 3 failed (fetch missing and two consent behaviors), 9 passed.
- First requested green suite run: `uv run --no-sync pytest tests/hub/test_activation_cli.py tests/hub/test_activation_client.py -q -p no:cacheprovider` — 179 passed, 6 failures; identified missing validated-source expectations in old mocks, cancellation/mock fetch, post-fetch failure semantics, changed checkout mock fixture, and configured-dimensions preflight regression. Fixed these.
- `uv run --no-sync pytest tests/hub/test_activation_cli.py -q -p no:cacheprovider` — initially 78 passed, 3 failures while iterating; all identified regressions corrected.
- Requested green suite final: `uv run --no-sync pytest tests/hub/test_activation_cli.py tests/hub/test_activation_client.py -q -p no:cacheprovider` — 185 passed.
- `uv run ruff format chunkhound/api/cli/commands/pull.py tests/hub/test_activation_cli.py` — passed; reformatted pull.py.
- `uv run ruff check chunkhound/api/cli/commands/pull.py tests/hub/test_activation_cli.py` — passed.
- `git diff --check` — passed.
- `uv run mypy chunkhound` — failed with 999 errors in 90 files, predominantly established unrelated realtime mixin and other modules. No reported failures were in the changed pull command or test file; treated as repository-wide baseline concern, not modified as out of scope.
- **Immediately before commit**, required `uv run pytest tests/test_smoke.py -v -n auto` — passed, 15 passed; 3 third-party `google.genai` deprecation warnings.

## Self-review
- Reviewed complete staged diff and ran `git diff --check` before smoke/commit.
- Scope restricted to the two files specified by the brief; no AGENTS.md changes.
- Confirmed fetch takes place once and validated source is passed to pull; grant precedes selection; consent follows selection; identity validated before authorization grant/replacement; output omits source path/query/credentials.
- Commit created only after successful mandatory smoke suite. No push.

## Concerns
- Repository-wide mypy remains failing with 999 unrelated errors in 90 source files (including realtime mixin typing failures). Smoke tests report three upstream deprecation warnings. Neither blocks this scoped task.
## Review fix section (Important findings)

Changes (`chunkhound/api/cli/commands/pull.py`, `tests/hub/test_activation_cli.py`):
- Added contract tests `test_trusted_plain_pull_fetch_local_state_error_is_logged_once` (RED: message absent) and `test_trusted_plain_pull_fetch_unavailable_emits_retry_guidance` (RED: guidance absent).
- `pull_attempted` is now `not args.trust` before the pre-client fetch (plain pull keeps failure guidance for fetch/selection failures; `--trust` keeps its prior semantics) and `True` immediately before `pull()`.
- `pre_client_fatal` is no longer reset to `bool(args.trust)` before fetch; it stays True until just before selection, so a fetch `HubLocalStateError` is logged exactly once.
- `configured_key(config)` moved to after the trust gate and immediately before fetch: untrusted plain pull still rejects before any hub/embedding call, and unknown dimensions still fail before any hub request (`test_unknown_dimensions_do_not_probe_hub_or_embedding_service` passes).

Commands and results:
- RED: `uv run --no-sync pytest tests/hub/test_activation_cli.py -k "trusted_plain_pull_fetch" -q -p no:cacheprovider` — 2 failed (expected).
- Intermediate: moving `pull_attempted=True` before fetch for `--trust` broke 4 trust-mode tests; fixed by `pull_attempted = not args.trust` pre-fetch.
- GREEN: `uv run --no-sync pytest tests/hub/test_activation_cli.py tests/hub/test_activation_client.py -q -p no:cacheprovider` — 187 passed.
- `uv run ruff format` and `uv run ruff check` on both files — passed.
- `git diff --check` — passed.
- Smoke (immediately before commit): `uv run pytest tests/test_smoke.py -v -n auto` — 15 passed, 3 warnings (google.genai deprecations).

Spec deviation: resolved without conflict. Brief's "configured_key at selection" is kept in order relative to the trust gate; key resolution happens pre-fetch, after the trust gate, which the brief's ordering allows.

## Second review fixes (HEAD `543547a7`)

Findings fixed:
- Wrapped cancellation at the `fetch_validated_manifest` boundary. A trusted plain pull cancelled there now logs exactly the fixed cancellation diagnostic and `HUB_PULL_FAILURE_GUIDANCE`, then exits 1; it does not invoke artifact pull.
- Set `pull_attempted = not args.trust` immediately after successful trust gate and before `configured_key(config)`. Thus plain trusted key-resolution failures retain retry guidance, while `--trust` pre-client semantics remain unchanged. Configured-key resolution remains before manifest fetch, so unknown dimensions issue neither hub nor embedding requests.
- Removed unused `_capture_errors()` helper.

TDD evidence:
- RED: `uv run --no-sync pytest tests/hub/test_activation_cli.py -k 'manifest_fetch_cancellation or trusted_plain_pull_unknown_dimensions' -q -p no:cacheprovider` — 2 failed as expected: fetch-boundary `CancelledError` escaped, and unknown-dimensions output lacked retry guidance.
- GREEN same targeted command — 2 passed.

Verification commands/results:
- `uv run --no-sync pytest tests/hub/test_activation_cli.py tests/hub/test_activation_client.py -q -p no:cacheprovider` — 189 passed.
- `uv run ruff format chunkhound/api/cli/commands/pull.py tests/hub/test_activation_cli.py` — passed (2 files unchanged).
- `uv run ruff check chunkhound/api/cli/commands/pull.py tests/hub/test_activation_cli.py` — passed.
- `git diff --check` — passed.
- Self-review: trust gate still precedes key resolution; configured key remains before network access; only trusted plain requests get the earlier failure-state; `--trust` does not gain retry guidance for pre-client key/fetch failure; fetch cancellation test asserts exact diagnostics, exit 1, and no artifact pull; helper removed.
- Required immediately-before-commit smoke run: `uv run pytest tests/test_smoke.py -v -n auto` — 15 passed, 3 third-party `google.genai` deprecation warnings.
