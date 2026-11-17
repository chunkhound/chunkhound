# Task 3 Report: Manifest Validation and Absent-Only Automatic Installation

## Status

DONE_WITH_CONCERNS. Implementation and required runtime validation are complete. Commit: `93789d08` (`fix(hub): require trust for cold automatic installs`).

## Implementation

- Split strict manifest wire parsing from compatibility selection with `parse_manifest(raw)` and `select_manifest(manifest, key, rel)`.
- Added `ValidatedHubSource` and `fetch_validated_manifest(config)` as a single-fetch capability bound to the canonical checkout digest and exact manifest URL digest.
- Added `source` handoff to public/internal pull interfaces. A supplied source bypasses trust-store reads only when both current binding digests match; it never refetches the manifest.
- Automatic pulls without a source now require persisted exact checkout+URL trust before local or network work.
- Replaced drift-based automatic replacement with ownership-scoped `automatic_target_absent()`: every existing database is ineligible, interrupted markers remain fatal, and the same absent-only predicate is rechecked under exclusive installation ownership.
- Added `HubLocalStateError` to fatal hub categories. Native busy/layout/filesystem errors and client-side target/install filesystem errors no longer permit local-index fallback. Transport, manifest, checksum, and bounded archive failures retain soft behavior.
- Preserved explicit replacement behavior, interrupted-marker recovery, competitor-created-destination handling, and deferred reader ownership contracts applicable to explicit pulls.

## Files Changed

- `chunkhound/hub/manifest.py`
- `chunkhound/hub/client.py`
- `chunkhound/hub/errors.py`
- `chunkhound/hub/install.py`
- `tests/hub/test_manifest.py`
- `tests/hub/test_activation_client.py`
- `tests/hub/test_artifact.py` (mechanical callers required by the split manifest interface)

## TDD Evidence

### RED

Command:

```bash
UV_PROJECT=/home/e119870/views/g/my_chunkhound/.worktrees/pr442-ci-fixes UV_NO_SYNC=1 uv run pytest tests/hub/test_manifest.py tests/hub/test_activation_client.py -v
```

Result: exit 2 during collection, with expected missing-API failures:

- `ImportError: cannot import name 'parse_manifest'`
- `ImportError: cannot import name 'fetch_validated_manifest'`

This established that the new parsing and validated-source contracts did not exist.

During self-review, an additional focused RED was run:

```bash
UV_PROJECT=/home/e119870/views/g/my_chunkhound/.worktrees/pr442-ci-fixes UV_NO_SYNC=1 uv run pytest tests/hub/test_activation_client.py::test_target_resolution_filesystem_failure_is_fatal -v
```

Result: 1 failed because no `HubLocalStateError` was raised, proving target-resolution `OSError` still fell back softly.

### GREEN

The same self-review test passed after splitting `OSError` from `ValueError` in `installation_target()`.

Final focused command:

```bash
UV_PROJECT=/home/e119870/views/g/my_chunkhound/.worktrees/pr442-ci-fixes UV_NO_SYNC=1 uv run pytest tests/hub/test_manifest.py tests/hub/test_activation_client.py -q
```

Result: **103 passed in 66.36s**.

## Exact Validation Commands and Results

1. Focused contracts (final):

```bash
UV_PROJECT=/home/e119870/views/g/my_chunkhound/.worktrees/pr442-ci-fixes UV_NO_SYNC=1 uv run pytest tests/hub/test_manifest.py tests/hub/test_activation_client.py -q
```

Result: **103 passed**.

2. Lint and whitespace validation:

```bash
UV_PROJECT=/home/e119870/views/g/my_chunkhound/.worktrees/pr442-ci-fixes UV_NO_SYNC=1 uv run ruff check chunkhound/hub/manifest.py chunkhound/hub/client.py chunkhound/hub/errors.py chunkhound/hub/install.py tests/hub/test_manifest.py tests/hub/test_activation_client.py tests/hub/test_artifact.py && git diff --check
```

Result: **All checks passed; exit 0**.

3. Required type-check command (expanded to include `install.py`):

```bash
UV_PROJECT=/home/e119870/views/g/my_chunkhound/.worktrees/pr442-ci-fixes UV_NO_SYNC=1 uv run mypy chunkhound/hub/manifest.py chunkhound/hub/client.py chunkhound/hub/errors.py chunkhound/hub/install.py
```

Result: **exit 1, 525 errors in 64 transitively imported files**. Search of the captured output found no diagnostics in the four requested hub files. Examples are established unrelated parser/provider errors such as `chunkhound/parsers/mappings/json.py` assignments and missing pandas/pyarrow stubs.

4. Mandatory smoke after final production change:

```bash
UV_PROJECT=/home/e119870/views/g/my_chunkhound/.worktrees/pr442-ci-fixes UV_NO_SYNC=1 uv run pytest tests/test_smoke.py -v -n auto
```

Result: **15 passed, 3 third-party `google.genai` deprecation warnings, exit 0**.

## Self-Review Findings

- Found that `installation_target()` still converted target-resolution `OSError` into soft `HubUnavailable`, contrary to the fatal local-state requirement. Added a failing contract and corrected it to `HubLocalStateError`; retained `ValueError` as unsupported configured layout.
- Confirmed source mismatch returns before artifact fetching and supplied matching source does not read a malformed store or refetch the manifest.
- Confirmed the absent-only predicate acquires ownership before existence inspection and is reused under exclusive ownership.
- Confirmed exact trust matching is tested for absent grant, wrong checkout, and wrong URL.
- Confirmed no registry initialization was moved and `local_needs_pull()` remains for explicit inspection/retry semantics.
- Updated artifact tests mechanically for the new parsed-manifest interface.

## Concerns

- The requested mypy command cannot currently be green because this repository baseline reports 525 unrelated errors through transitive imports. No changed hub file appears in those diagnostics.
- Automatic variants of tests whose old purpose depended on inspecting/replacing an existing database were removed or narrowed because absent-only policy intentionally makes those paths unreachable. Dedicated marker, race, existing-aligned, and existing-misaligned contracts cover the new automatic behavior.
- Standard `uv` emits the repository's existing deprecated `native-tls` setting warning; smoke also emits three third-party Python 3.14 deprecation warnings.

## Review Fix Wave 1

### Findings Fixed

- `fetch_validated_manifest()` now captures the exact URL plus checkout and URL digests before awaiting transport, and fetches with the captured URL. A concurrency contract mutates both config URL and checkout symlink while transport is suspended and proves the returned capability retains the pre-await bindings.
- `automatic_target_absent()` now uses `Path.lstat()`, returns absent only for `FileNotFoundError`, and maps every other `OSError` to sanitized fatal `HubLocalStateError`. The filesystem-failure contract now injects a non-`FileNotFoundError` through `lstat()` rather than `exists()`; cancellation/race synchronization tests were mechanically moved to that actual lookup boundary.
- `automatic_pull_eligible()` now documents absent-only cold-install eligibility rather than replacement based on local drift.

### RED

Command:

```bash
UV_PROJECT=/home/e119870/views/g/my_chunkhound/.worktrees/pr442-ci-fixes UV_NO_SYNC=1 uv run pytest tests/hub/test_activation_client.py::test_validated_source_binding_is_snapshotted_before_fetch tests/hub/test_activation_client.py::test_automatic_local_filesystem_failure_is_fatal -v
```

Result: **2 failed in 3.27s**. The source used the mutated post-await URL digest instead of the captured digest, and the `lstat()` `OSError` contract did not raise `HubLocalStateError` because production still called `exists()`.

### GREEN

Same command after implementation: **2 passed in 2.67s**.

Focused Task 3 command:

```bash
UV_PROJECT=/home/e119870/views/g/my_chunkhound/.worktrees/pr442-ci-fixes UV_NO_SYNC=1 uv run pytest tests/hub/test_activation_client.py tests/hub/test_manifest.py -v
```

Result: **104 passed in 68.54s**.

Ruff and whitespace command:

```bash
UV_PROJECT=/home/e119870/views/g/my_chunkhound/.worktrees/pr442-ci-fixes UV_NO_SYNC=1 uv run ruff check chunkhound/hub/client.py tests/hub/test_activation_client.py && git diff --check
```

Result: **All checks passed; exit 0**.

Mandatory smoke command:

```bash
UV_PROJECT=/home/e119870/views/g/my_chunkhound/.worktrees/pr442-ci-fixes UV_NO_SYNC=1 uv run pytest tests/test_smoke.py -v -n auto
```

Result: **15 passed, 3 third-party `google.genai` deprecation warnings in 6.50s; exit 0**.

### Files Changed

- `chunkhound/hub/client.py`
- `tests/hub/test_activation_client.py`
- `.superpowers/sdd/task-3-report.md`

### Self-Review

- Verified the URL read used for transport is the same captured string whose digest binds the capability.
- Verified checkout canonicalization and both digests occur before the first await.
- Verified dangling symlinks and ordinary missing paths remain absent via `lstat()` `FileNotFoundError`, while permission/I/O lookup failures are fatal and sanitized.
- Verified automatic race/cancellation tests block the production `lstat()` boundary after the predicate change.
- Diff is limited to the three review findings and their contracts/report evidence; no F2 or unrelated behavior was added.

### Concerns

- No new functional concerns. Existing `uv` deprecated `native-tls` warning and three third-party Python 3.14 `google.genai` deprecation warnings remain.

## Review Fix Wave 2

### Finding Fixed

- Automatic no-source pull now snapshots the exact trusted manifest URL and canonical checkout identity before the absence-worker await, carries those bindings into the single manifest fetch, and refuses if current URL or checkout identity differs before or after transport. The eventual capability therefore describes the same identity that passed stored trust; supplied-source and explicit-pull behavior are unchanged.
- Added a deterministic concurrency contract that suspends the initial absence check, mutates the configured URL, retargets the checkout symlink, then proves no request is made and no database is installed for the untrusted identity.

### RED

Command:

```bash
UV_PROJECT=/home/e119870/views/g/my_chunkhound/.worktrees/pr442-ci-fixes UV_NO_SYNC=1 uv run pytest tests/hub/test_activation_client.py::test_automatic_pull_reauthorizes_identity_after_absence_wait -v
```

Result: **1 failed in 2.72s**. After mutation, production requested `/catalogue?token=secret&changed=true`, proving it reread and fetched the untrusted post-await URL.

### GREEN

Focused regression command after implementation (test subsequently renamed only for accuracy):

```bash
UV_PROJECT=/home/e119870/views/g/my_chunkhound/.worktrees/pr442-ci-fixes UV_NO_SYNC=1 uv run pytest tests/hub/test_activation_client.py::test_automatic_pull_reauthorizes_identity_after_absence_wait -v
```

Result: **1 passed in 2.14s**.

Final Task 3 command:

```bash
UV_PROJECT=/home/e119870/views/g/my_chunkhound/.worktrees/pr442-ci-fixes UV_NO_SYNC=1 uv run pytest tests/hub/test_activation_client.py tests/hub/test_manifest.py -v
```

Result: **105 passed in 68.34s**.

Ruff command:

```bash
UV_PROJECT=/home/e119870/views/g/my_chunkhound/.worktrees/pr442-ci-fixes UV_NO_SYNC=1 uv run ruff check chunkhound/hub/client.py tests/hub/test_activation_client.py
```

Result: **All checks passed; exit 0**.

Whitespace command:

```bash
git diff --check
```

Result: **exit 0, no output**.

Mandatory smoke command:

```bash
UV_PROJECT=/home/e119870/views/g/my_chunkhound/.worktrees/pr442-ci-fixes UV_NO_SYNC=1 uv run pytest tests/test_smoke.py -v -n auto
```

Result: **15 passed, 3 third-party `google.genai` deprecation warnings in 6.52s; exit 0**.

### Files Changed

- `chunkhound/hub/client.py`
- `tests/hub/test_activation_client.py`
- `.superpowers/sdd/task-3-report.md`

### Self-Review

- Confirmed the trust check, manifest transport, and generated `ValidatedHubSource` all use one captured URL plus one captured canonical-checkout digest.
- Confirmed identity drift is rejected both after the absence await and after manifest transport, closing mutation windows without a second fetch.
- Confirmed automatic absent-only inspection and exclusive installation recheck remain unchanged.
- Confirmed supplied-source behavior still bypasses trust-store reads after current binding validation, and explicit pull still uses the existing single-fetch path.
- Diff is limited to this review finding, its user-visible concurrency contract, and report evidence; no F2 or unrelated changes were added.

### Concerns

- No new functional concerns. Existing `uv` deprecated `native-tls` warning and three third-party Python 3.14 `google.genai` deprecation warnings remain.
