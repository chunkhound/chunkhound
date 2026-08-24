# ChunkHound Release Guide

`scripts/prepare_release.sh` is a deprecated local verification helper only; it
does not publish anything and must not replace the GitHub Release workflow
documented below.

## Prerequisites (one-time setup)

### 1. OIDC Trusted Publishing — PyPI

Configure Trusted Publisher on **PyPI** for the **`chunkhound`** project, covering both the
release and RC workflows (RC pre-releases publish to PyPI, not TestPyPI):

- Project: `chunkhound`
- Owner: `chunkhound`
- Repository: `chunkhound`
- Workflow: `release.yml`, Environment: `pypi`
- Workflow: `release-rc.yml`, Environment: `pypi`

Separately, configure Trusted Publisher on **PyPI** for the **`chunkhound-native`** project — it
is a distinct PyPI project published by the `publish-native`/`publish-rc-native` jobs in the same
two workflows:

- Project: `chunkhound-native`
- Owner: `chunkhound`
- Repository: `chunkhound`
- Workflow: `release.yml`, Environment: `pypi-native`
- Workflow: `release-rc.yml`, Environment: `pypi-native`

**Each project's trusted publisher must only claim the environment name its own jobs actually
declare.** An overly broad "any environment" entry on `chunkhound` will intercept the OIDC token
exchange meant for `chunkhound-native` (or vice versa) — the upload then fails with a `403 Invalid
API Token: OIDC scoped token is not valid for project '...'` error even though both projects look
correctly configured in isolation. If you see that error, check whether the *other* project has an
unscoped or wrongly-scoped entry stealing the match.

### 2. GitHub Environments

Create four environments in **Settings → Environments**:

| Environment | Purpose | Protection rules |
|---|---|---|
| `pypi` | Production + RC PyPI publish for `chunkhound` | Required reviewers: maintainer team |
| `pypi-native` | Production + RC PyPI publish for `chunkhound-native` | Same trusted-publisher scoping as `pypi`, kept as a separate environment since it's a distinct PyPI project |
| `maintainers` | Deprecation approvals | Required reviewers: maintainer team |

There is no `testpypi` environment — RC releases publish pre-release versions to the real PyPI
index (see `release-rc.yml`), not TestPyPI.

### 3. Tag Protection Rules

In **Settings → Rules → Rulesets**, create a ruleset:

- Target: tags matching `v*`
- Restrict tag creation/deletion to: maintainer team
- This ensures only maintainers can trigger RC and release workflows

### 4. Deprecation secret

Add `PYPI_API_TOKEN` to the `maintainers` environment secrets. This token must have **Owner** role on the `chunkhound` PyPI project (required for yanking releases).

---

## RC Release

Use this to validate a build on PyPI (as a pre-release) before cutting the real release.

```bash
# Create and push a pre-release tag — this triggers the RC workflow automatically
uv run scripts/update_version.py 1.2.0rc1
git push origin v1.2.0rc1
```

The `release-rc.yml` workflow builds and publishes to the real PyPI index (not TestPyPI) via OIDC, tagged as a pre-release. No manual approval needed — the tag push itself is the human gate (only maintainers can push `v*` tags).

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
   because the main upload needs a retry. A full workflow rerun is also safe:
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
