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

   Publishing triggers `release.yml`, which builds and publishes to PyPI via OIDC. The `pypi` environment requires maintainer approval before the publish step runs.

4. **If the build or publish fails** — the cleanup step runs only if the main `chunkhound` publish fails; it deletes the GitHub Release and its tag so you can retry from step 1. If `publish-native` fails after the main package is already live, the GitHub Release remains published and cleanup does not run. In that case re-trigger `publish-native` manually via Actions (it is idempotent), or upload the native wheel directly with a PyPI token scoped to `chunkhound-native`.

   **If `build-native-wheel` itself fails** (before `publish`/`publish-native` even start), cleanup
   does *not* run either — it's gated on the `publish` job failing, and `publish` never starts if a
   `needs` dependency failed. You'll be left with a published GitHub Release and tag pointing at a
   broken build, with nothing actually shipped to PyPI. Recover by: fixing the issue, deleting the
   release and tag (`gh release delete vX.Y.Z --cleanup-tag --yes`), then retrying from step 1 with
   a new tag on the fixed commit. Common `build-native-wheel` failure modes (all fixed as of v5.2.0,
   documented inline in `release.yml`/`release-rc.yml`'s `Build native wheel`/`Smoke test native
   wheel` steps): `setup-uv@v3` dropping the `python-version` input, `maturin build` having no
   `--set-version` flag (Cargo.toml's version must be patched directly), Cargo requiring strict
   SemVer while git tags use PEP 440 (needs a hyphen before any pre-release suffix), maturin
   misreading the repo's own `pyproject.toml` as authoritative package metadata, and the repo's
   `chunkhound_native/__init__.py` dev-workflow stub shadowing the installed wheel during the smoke
   test.

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
