# ChunkHound Release Guide

## Prerequisites (one-time setup)

### 1. OIDC Trusted Publishing — PyPI

Configure Trusted Publisher on **PyPI** for the release workflow:

- Project: `chunkhound`
- Owner: `chunkhound`
- Repository: `chunkhound`
- Workflow: `release.yml`
- Environment: `pypi`

Repeat for **TestPyPI** for the RC workflow:

- Workflow: `release-rc.yml`
- Environment: `testpypi`

### 2. GitHub Environments

Create three environments in **Settings → Environments**:

| Environment | Purpose | Protection rules |
|---|---|---|
| `pypi` | Production PyPI publish | Required reviewers: maintainer team |
| `testpypi` | TestPyPI RC publish | None required (tag push is the gate) |
| `maintainers` | Deprecation approvals | Required reviewers: maintainer team |

### 3. Tag Protection Rules

In **Settings → Rules → Rulesets**, create a ruleset:

- Target: tags matching `v*`
- Restrict tag creation/deletion to: maintainer team
- This ensures only maintainers can trigger RC and release workflows

### 4. Deprecation secret

Add `PYPI_API_TOKEN` to the `maintainers` environment secrets. This token must have **Owner** role on the `chunkhound` PyPI project (required for yanking releases).

---

## RC Release

Use this to validate a build on TestPyPI before cutting the real release.

```bash
# Create and push a pre-release tag — this triggers the RC workflow automatically
git tag v1.2.0-rc.1
git push origin v1.2.0-rc.1
```

The `release-rc.yml` workflow builds and publishes to TestPyPI via OIDC. No manual approval needed — the tag push itself is the human gate (only maintainers can push `v*` tags).

**Validate the RC:**
```bash
pip install --index-url https://test.pypi.org/simple/ chunkhound==1.2.0rc1
```

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

4. **If the build or publish fails** — the workflow automatically deletes the GitHub Release and its tag (`--cleanup-tag`). No orphaned public state. Simply fix the issue and repeat from step 1.

---

## Deprecating a Release

```bash
gh workflow run deprecate.yml \
  -f version=1.2.0 \
  -f reason="Critical bug in X, upgrade to 1.2.1"
```

This triggers `deprecate.yml`, which:

1. Waits for approval from the `maintainers` environment (approver is notified by GitHub)
2. Yanks the version from PyPI (pip will no longer install it by default)
3. Adds a deprecation notice to the corresponding GitHub Release

The full audit trail (who, when, why) is recorded in the Actions run history.

---

## Version Management

Versions are derived from git tags via `hatch-vcs`. Never edit version strings manually.

```bash
# Bump and tag
uv run scripts/update_version.py 1.2.0
```
