# Task 6 report — declaration authority, delegated artifacts, and consent UX

## Status
Implemented the Task 6 external contracts in the three requested files. No runtime API or production code changed.

## Implementation
- `tests/remote_config/test_pipeline_e2e.py`
  - Added end-to-end `hub` whole-object `set` and `merge` rule coverage.
  - Proves the resulting global declaration resolves for a real noninteractive `index`, but causes nonzero exit before hub/embedding traffic, creates no trust store, and creates no local DB.
  - Reuses the real hub/embedding HTTP fixtures through local fixture adapters.
- `tests/hub/test_activation_cli.py`
  - Added the same no-authority contract for environment, explicit `--config`, project, and global declarations.
  - Added a trusted-manifest delegation test: manifest origin A delegates to artifact origin B, B performs one normal redirect, the final archive installs, its manifest byte count and SHA-256 match, the DB is inspectable, and only the manifest grant is persisted (no second prompt).
- `site/src/pages/docs/configuration.md`
  - Documented hub URL precedence and null/empty disable semantics.
  - Documented declaration versus authorization, one-time pull, persistent `--trust`, interactive choices, fail-closed noninteractive behavior, offline forget, checkout/URL rebinding, manifest artifact/redirect delegation with integrity safeguards, and no automatic replacement of existing DBs.
  - Treats the trust file as ChunkHound-managed and does not expose its format.

## TDD RED / GREEN
- RED attempt 1: focused declaration test collection failed because imported fixtures were not registered under the requested names (2 setup errors). Fixed the test harness, not production behavior.
- GREEN: remote `set`/`merge` authority tests: `2 passed, 81 deselected`.
- RED for source/delegation assertions: 5 failures. Four were a test-path error (`trust_store_path()` observed the developer HOME rather than the subprocess HOME); one asserted embedding traffic even though the fixture archive already contains no overlay source. Corrected assertions to the actual external contract.
- GREEN: four declaration sources plus delegation: `5 passed, 52 deselected`.
- No production runtime gap was exposed; existing Task 5 behavior already satisfied the Task 6 contracts.

## Verification commands and exact results
All Python commands used:
`UV_PROJECT=/home/e119870/views/g/my_chunkhound/.worktrees/pr442-ci-fixes UV_NO_SYNC=1`.

1. Focused bypass/delegation command from brief:
   - `uv run pytest tests/remote_config/test_pipeline_e2e.py tests/hub/test_activation_cli.py -v -k 'trust or hub or remote or redirect or delegated'`
   - Final: **140 passed in 193.48s**.
2. Full focused F1 suite from brief:
   - `uv run pytest tests/hub/test_config.py tests/hub/test_trust_store.py tests/hub/test_manifest.py tests/hub/test_activation_client.py tests/hub/test_activation_cli.py tests/hub/test_activation_acceptance.py tests/hub/test_download_deadline.py tests/unit/test_cli_hub_consent.py tests/unit/test_cli_model_drift.py tests/unit/test_embedding_model_drift.py tests/remote_config/test_pipeline_e2e.py -v`
   - **486 passed, 8 failed, 12 warnings in 353.51s**.
   - All 8 failures are stale acceptance expectations incompatible with the approved F1 policy: four moved-checkout tests expect a moved checkout to continue without re-trust; four interrupted-auto tests never establish trust and expect old local-state diagnostics. Task 6 explicitly documents moving/recloning requires trust again and preserves Task 5's authorization-before-target-inspection contract. Warnings: one existing Google GenAI deprecation plus 11 existing AsyncMock unawaited warnings.
3. `uv run ruff check chunkhound`
   - Exit 1, **765 existing errors** across unrelated production files (mostly E501; also existing import/order and lint debt). Task 6 changed no `chunkhound/` file.
4. `uv run mypy chunkhound`
   - Exit 1, **975 existing errors in 88 files (400 checked)**, dominated by known realtime mixin/type debt. No Task 6 production file exists.
5. Mandatory smoke:
   - `uv run pytest tests/test_smoke.py -v -n auto`
   - **15 passed, 3 warnings in 7.00s**; warnings are existing Google GenAI deprecation warnings.
6. Changed-test lint:
   - `uv run ruff check tests/remote_config/test_pipeline_e2e.py tests/hub/test_activation_cli.py`
   - **All checks passed**.
7. `git diff --check`
   - Exit 0, no whitespace errors.
8. `git status --short`
   - Only the three requested Task 6 files modified.

## Self-review
- Scope is exactly the requested three files; no production API/runtime redesign and no F2 work.
- Tests exercise real subprocess CLI and real local HTTP boundaries, not private helper/mock plumbing.
- Every untrusted declaration assertion verifies zero hub requests, zero embedding requests, no trust store, and no DB.
- Delegation covers separate origins and one redirect, exact chain, manifest request count, final archive integrity metadata, installed DB inspection, trust persistence, and credential redaction.
- Documentation includes every Task 6 bullet and avoids presenting the trust-store representation as editable configuration.

## Concerns
- The brief's full focused suite includes 8 acceptance tests whose old expectations conflict with the approved F1 behavior and Task 5 interface decision. They should be updated in the Task 7 whole-branch acceptance reconciliation, not bypassed or fixed by weakening authorization.
- Repository-wide Ruff and mypy gates have substantial pre-existing debt (765 and 975 errors respectively); no error is attributable to the three Task 6 files.

---

## Important review-finding fixes

### Root cause and adjudication
- The older Central Indexing Hub prose still described the pre-F1 automatic replacement/fallback policy, contradicting the new authorization section. It now states absent-only authorized automatic installation, no automatic replacement of any existing DB, and no silent noninteractive local fallback after authorized hub failure. Blank/null is explicitly a present higher-priority disable.
- Acceptance setup was stale: moved-checkout cases used one-time pull without persisting trust; interrupted-target cases expected target inspection before authorization. The former now pulls with `--trust`; the latter establishes trust via a real pull, recreates the interrupted marker, then verifies zero hub/embedding requests and a fatal recovery diagnostic.
- `HubConfig` deliberately ignores unknown local/global/explicit keys and environment loading enumerates accepted keys, while remote rules reject unknown model fields. External tests therefore use plausible `hub.trust_store_path`: declaration sources ignore it, remote whole-hub set/merge report `schema_error`, and neither alternate nor managed trust stores are created. No runtime trust-path API was added.
- The interrupted fatal exception was swallowed without its user-facing diagnostic. Runtime now formats the normalized fatal error before exiting; this is real enforcement exercised by acceptance coverage.
- Local HTTP test servers now expose plain context-manager helpers, removing fixture `.__wrapped__()` coupling.

### RED / GREEN evidence
- Baseline RED: `tests/hub/test_activation_acceptance.py -v` => **8 failed, 2 passed** (four missing exact-checkout trust grants; four authorization preempted interrupted-target inspection).
- Trust-path RED: remote whole-hub malicious set/merge correctly rejected the entire declaration, exposing two stale expectations that hub authorization failure would follow; split into the exact schema-rejection contract. Local/env declarations passed because unknown fields are intentionally ignored.
- Intermediate RED after explicit authorization: interrupted-target tests reached runtime enforcement but emitted no fatal diagnostic (**4 failed, 6 passed**); after formatting the normalized fatal error, only stale absent-target assertions remained (**4 failed, 6 deselected**), corrected to preserve the installed target plus interrupted marker.
- GREEN acceptance: `tests/hub/test_activation_acceptance.py -v` => **10 passed in 197.58s**.
- GREEN trust-path/remote focused: declaration/remote path tests => **4 passed, 81 deselected**; local/env/explicit/project/global source tests => **4 passed, 53 deselected**.
- GREEN mandated full focused F1 suite => **496 passed, 12 pre-existing warnings in 487.38s**.
- GREEN changed-file Ruff => **All checks passed**; mandatory smoke => **15 passed, 3 pre-existing warnings in 7.14s**; `git diff --check` => exit 0.

### Changed files
- `chunkhound/api/cli/commands/run.py`
- `site/src/pages/docs/configuration.md`
- `tests/hub/conftest.py`
- `tests/hub/test_activation_acceptance.py`
- `tests/hub/test_activation_cli.py`
- `tests/remote_config/test_pipeline_e2e.py`

### Commit
- Implementation and verification commit: `ad9c0a87` (`fix(hub): close task 6 consent review gaps`).

### Remaining concerns
- The mandated focused suite still reports one Google GenAI deprecation and 11 existing AsyncMock coroutine warnings; no test failures remain.
- No F2 work, push, or publication was performed.
