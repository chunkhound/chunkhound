# ChunkHound Release Guide

`scripts/prepare_release.sh` is a deprecated local verification helper only; it
does not publish anything and must not replace the GitHub Release workflow
documented below. Do not publish with `uv publish`; CI owns every PyPI upload.

## Release workflow

Run the smoke suite before creating a release tag:

```bash
uv run pytest tests/test_smoke.py -v -n auto
```

A stable release is published when its GitHub Release is published. A
beta/RC tag triggers `release-rc.yml` and publishes the pre-release to PyPI.
Successful upstream `main` push builds publish automatic alphas through the
serialized, tagless alpha path described below, which calls the reusable
`release-packages.yml` workflow directly. Alphas are never created by tags.
Create manual beta/RC/stable tags with `uv run scripts/update_version.py`;
never edit version strings manually.

## Prerequisites (one-time setup)

### 1. OIDC Trusted Publishing — PyPI

Configure Trusted Publishers on **PyPI** for the reusable **`release-packages.yml`**
workflow. PyPI identifies the publishing job through `job_workflow_ref`, which names
the reusable workflow, not the entry workflow (`ci.yml`, `release.yml`, or
`release-rc.yml`). All channels publish to real PyPI, not TestPyPI:

- Project: `chunkhound`
- Owner: `chunkhound`
- Repository: `chunkhound`
- Workflow: `release-packages.yml`, Environment: `pypi`
- Workflow: `release-packages.yml`, Environment: `pypi-alpha`

Separately, configure Trusted Publisher on **PyPI** for the **`chunkhound-native`** project — it
is a distinct PyPI project published by the reusable workflow's `publish-native` job:

- Project: `chunkhound-native`
- Owner: `chunkhound`
- Repository: `chunkhound`
- Workflow: `release-packages.yml`, Environment: `pypi-native`
- Workflow: `release-packages.yml`, Environment: `pypi-alpha-native`

Add these entries before enabling the new workflows. Existing publisher entries for
old workflow filenames do not authorize the reusable publishing jobs.

Both publishing steps disable optional attestations while retaining OIDC Trusted
Publishing. Attestations identify the caller workflow (for example, `ci.yml`),
but PyPI authenticates the reusable `release-packages.yml` workflow. PyPI rejects
that mismatch with `400 Invalid attestations`. Keep attestations disabled until
this reusable-workflow limitation is resolved; do not change the publisher's
workflow filename to the caller to work around it.

**Each project's trusted publisher must only claim the environment name its own jobs actually
declare.** An overly broad "any environment" entry on `chunkhound` will intercept the OIDC token
exchange meant for `chunkhound-native` (or vice versa) — the upload then fails with a `403 Invalid
API Token: OIDC scoped token is not valid for project '...'` error even though both projects look
correctly configured in isolation. If you see that error, check whether the *other* project has an
unscoped or wrongly-scoped entry stealing the match.

### 2. GitHub Environments

Configure these environments in **Settings → Environments**:

| Environment | Purpose | Protection rules |
|---|---|---|
| `pypi` | Production + RC PyPI publish for `chunkhound` | Required reviewers: maintainer team |
| `pypi-native` | Production + RC PyPI publish for `chunkhound-native` | Same trusted-publisher scoping as `pypi`, kept as a separate environment since it's a distinct PyPI project |
| `pypi-alpha` | Automatic alpha publish for `chunkhound` | Restrict to trusted repository runs; no required reviewer |
| `pypi-alpha-native` | Automatic alpha publish for `chunkhound-native` | Same alpha controls; no required reviewer |
| `maintainers` | Deprecation approvals | Required reviewers: maintainer team |

There is no `testpypi` environment — RC releases publish pre-release versions to the real PyPI
index (see `release-rc.yml`), not TestPyPI.

### 3. Tag Protection Rules

In **Settings → Rules → Rulesets**, create a ruleset:

- Protect `v*` tags against updates and deletion.
- Remove any `alpha-build/**` ruleset or tags left from the previous design.
- Do **not** grant the Actions identity permission to create alpha tags or
  `alpha-build/**` tags. Automatic alphas are tagless and need no tag creation.
- Prohibit tags that name alpha versions (`v*a*`). Any such tag is a
  misconfiguration and must not be pushed.
- Keep stable/beta/RC tag creation restricted to maintainers.

Use separate creation and update/deletion rules so the release tag set cannot be
moved. Verify the effective rules with repository admins: a tag-creation rule
applies to `GITHUB_TOKEN` too. The alpha workflow needs no `contents: write`
permission for tags.

### 4. Deprecation secret

Add `PYPI_API_TOKEN` to the `maintainers` environment secrets. This token must have **Owner** role on the `chunkhound` PyPI project (required for yanking releases).

---

## Automatic Alpha Releases

Automatic alphas are **tagless**. Every successful push build on upstream
`chunkhound/chunkhound` main publishes the next alpha selected from PyPI state.
PR, merge-queue, manual, and fork CI runs do not publish.
No tag, GitHub App, PAT, or `contents: write` is used for alphas.

### Allocation

- The latest published `chunkhound` stable `X.Y.Z` selects the next minor core:
  `X.(Y+1).0a1`. If a native-only partial alpha already exists on that new core,
  allocation advances past it. If latest `chunkhound` is alpha `X.Y.ZaN`, the
  core remains exactly `X.Y.Z` and the next serial is `max(aN across both
  published projects for that core) + 1` (for example, `6.2.0a12` becomes
  `6.2.0a13`).
- Under one alpha lock, the workflow queries both PyPI projects (`chunkhound`
  and `chunkhound-native`) and allocates from their published state. The allocator
  accepts no core override.
- Beta, RC, dev, post, and local releases are ignored when choosing the line. Only
  plain stable and plain alpha releases choose it. If a beta or RC already exists on
  the selected core (for example, `6.2.0` is stable and `6.3.0b1` is published, so
  the next core is `6.3.0`), the alpha job stops with a clear error and publishes
  nothing. This case needs a maintainer decision.
- The chosen version, `source_sha`, and `channel=alpha` are passed explicitly to
  the build. The version is never derived from Git tags. The initial serial is
  canonical `a1` (not `a01`).

### Serialization and superseded runs

- One alpha workflow runs at a time. Allocation, builds, validation, and uploads
  share one concurrency group.
- A run that is already stale when it finishes CI is skipped before it enters the
  release queue. Once a tested snapshot enters the release lock, it always finishes
  publication even if `main` moves while its artifacts are building. A later tested
  snapshot waits and receives the next alpha serial.
- Uploads for alphas do not use `skip-existing`. If the selected version already
  exists on PyPI, the run fails instead of mixing artifacts from different commits.

### Gaps and ordering

Partial publication is possible. If native `aN` publishes but main `aN` fails,
PyPI keeps the native file. The next fresh run allocates `aN+1` from the union of
both projects, so `aN` remains a permanent gap. Reusing `aN` is never done.
Installers choose the highest version, so users still reach the newest alpha.

### Testing

```bash
pip install 'chunkhound==6.2.0a1'
```

Alphas sort below beta, RC, and stable releases of the same base version. Alpha
main artifacts require the matching native alpha version exactly. Stable and
beta/RC dependency semantics remain unchanged.

### Failures and reruns

- If the run fails before native publication, fix the cause and use **Re-run all
  jobs** on the current `main` CI run. That performs a fresh allocation. Do not
  rerun an older CI run that was skipped as stale.
- **Re-run failed jobs** keeps the original workflow state and allocated version.
  For a transient native publish failure, use it only if no native file for that
  version reached PyPI. After a partial native upload, use **Re-run all jobs** on
  the current `main` CI run to allocate a fresh version.
- If native publication succeeded but main publication failed **before any main
  file reached PyPI**, rerunning only the failed main publish job is safe: it
  reuses the same validated main artifacts and matching native version.
- If **any main file reached PyPI** before the failure, do not retry that main
  publish job. Alpha uploads do not skip existing files, so a retry would fail
  on those immutable files. Use **Re-run all jobs** on the current `main` CI run
  to allocate a fresh version from both PyPI projects. Leave the incomplete old
  version in place; the fresh run publishes both packages under the new serial.
- Do not create or push alpha tags to "fix" a failure. Tags for alpha versions are
  prohibited.

## RC Release

Use this to validate a build on PyPI (as a pre-release) before cutting the real release.

```bash
# Create and push a pre-release tag — this triggers the RC workflow automatically
uv run scripts/update_version.py 1.2.0rc1
git push origin v1.2.0rc1
```

The `release-rc.yml` wrapper calls `release-packages.yml` to publish to real PyPI.
Automatic alphas use the alpha environments without required reviewers and never
go through this tag path. Beta/RC tags use `pypi` and `pypi-native` and retain
those environments' configured approval rules.

Both release workflows use the same publication order:

1. Build the sdist, packaged runtime wheels, and all native wheels.
2. Install the Linux and Windows main wheels with their matching native wheels
   in clean temporary Python environments and exercise the native API.
3. Validate that every native wheel has the tag's version, then publish
   `chunkhound-native`.
4. Publish `chunkhound` only after native publication succeeds.

The shared native-wheel build smoke test exercises the same API contract on all
native runners, including Linux ARM64 and macOS ARM64, which do not have
matching packaged main runtime-wheel artifacts in this release matrix.

The native wheel version check runs before any native upload. A missing,
malformed, or mismatched wheel fails the workflow and skips both PyPI
publication jobs.

**Validate the RC:**
```bash
pip install chunkhound==1.2.0rc1
```

**Update the lockfile** — `pyproject.toml` only pins a floor version
(`chunkhound-native>=X.Y.Z`), so `uv.lock` must be bumped by hand every release to pick up the
version that was just published:
```bash
uv lock --upgrade-package chunkhound-native
git add uv.lock
git commit -m "chore: bump chunkhound-native in lockfile to v1.2.0rc1"
```
If this fails with "no matching version found," the native wheels haven't finished publishing yet
(or PyPI's index hasn't propagated) — retry in a minute, or force a fresh index fetch with
`uv lock --upgrade-package chunkhound-native --refresh-package chunkhound-native`.

---

## Full Release

1. **Create a GitHub Release draft** (via GitHub UI or CLI):

   ```bash
   gh release create v1.2.0 --draft --title "v1.2.0" --generate-notes
   ```

   `--generate-notes` drafts release notes from PR titles since the last release.

2. **Review and edit** the release notes in the GitHub UI.

3. **Publish the release** (click "Publish release" in the UI, or):

   ```bash
   gh release edit v1.2.0 --draft=false
   ```

   Publishing triggers `release.yml`, which builds and publishes to PyPI via OIDC. The `pypi-native` and `pypi` environments can require maintainer approval before their respective publication jobs run.

4. **If the native build, validation, or native publish fails** — no
   `chunkhound` package is published. The native failure alert creates a
   GitHub issue with the workflow URL, including when a prerequisite failure
   skips the native publish job. Fix the problem and re-run the workflow/jobs;
   native publication is idempotent for artifacts that were already uploaded,
   and the publish steps skip those existing files. The main publish job
   remains available to run after native publication succeeds.

   Failures in the sdist or packaged runtime-wheel builds also prevent
   publication, but do not use the native failure alert because no native
   failure occurred.

   **If the main publish fails** after native publication succeeds, the GitHub
   Release and tag are deliberately left in place. Re-run the failed downstream
   jobs from Actions after addressing the failure; do not create another tag just
   because the main upload needs a retry. A full workflow rerun is supported:
   release artifact uploads use `overwrite: true` for the same stable artifact
   names, while PyPI uploads use `skip-existing: true`. The native job does not
   need to be repeated unless its own upload failed.

   **If a build fails** before the native job starts, nothing is published but
   the GitHub Release and tag remain available for diagnosis and reruns. A full
   workflow rerun replaces the prior GitHub Actions artifacts rather than
   colliding with their names. If the
   source must be changed, delete that release and tag explicitly and create a
   new release from the fixed commit. Common `build-native-wheel` failure modes
   (all fixed as of v5.2.0, documented inline in `release.yml`/`release-rc.yml`'s
   `Build native wheel`/`Smoke test native wheel` steps) include
   `setup-uv@v3` dropping the `python-version` input, `maturin build` having no
   `--set-version` flag, Cargo requiring strict SemVer for PEP 440 prerelease
   tags, maturin misreading the repository `pyproject.toml`, and the repository
   `chunkhound_native/__init__.py` stub shadowing the installed wheel during
   the smoke test.

---

## Deprecating a Release

```bash
gh workflow run deprecate.yml \
  -f version=1.2.0 \
  -f reason="Critical bug in X, upgrade to 1.2.1"
```

This triggers `deprecate.yml`, which:

1. Waits for approval from the `maintainers` environment (approver is notified by GitHub)
2. Adds a deprecation notice to the corresponding GitHub Release

> **Note:** PyPI yanking is not yet automated. The yank API is CSRF-protected and has
> no stable machine-readable endpoint (see TODO in `deprecate.yml`). Until a supported
> API is available, manually yank the release via the PyPI web UI:
> **PyPI → Manage → chunkhound → Release → Yank**.

The full audit trail (who, when, why) is recorded in the Actions run history.

---

## Version Management

Versions are derived from git tags via `hatch-vcs`. Never edit version strings manually.

```bash
# Bump and tag
uv run scripts/update_version.py 1.2.0
```
