---
argument-hint: <VERSION> (e.g. 4.2.0)
description: Generate release notes for a ChunkHound production release
---

You are preparing release notes for a ChunkHound production release.

## Step 1: Determine the version

If a version was provided in `$ARGUMENTS`, use it as the release version.

If no version was provided, run:
```bash
git fetch --tags origin 2>/dev/null || true
LATEST_TAG=$(git tag --sort=-version:refname | head -1)
LATEST_STABLE=$(git tag --sort=-version:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1)
echo "Latest tag (any):    $LATEST_TAG"
echo "Latest stable tag:   $LATEST_STABLE"
```

Then apply this logic to suggest the next version:
- **If `LATEST_TAG` is a stable tag** (matches `v[0-9]+\.[0-9]+\.[0-9]+$`): suggest bumping
  the minor (e.g. `v4.1.0` → `4.2.0`).
- **If `LATEST_TAG` is a pre-release** (contains `a`, `b`, or `rc` suffix, e.g. `v4.2.0b1`):
  strip the suffix and suggest the corresponding stable release (e.g. `v4.2.0b1` → `4.2.0`).
  Do **not** bump the minor — the stable release that the pre-release was preparing is the
  correct next version.

Ask the user to confirm the suggested version before continuing.

## Step 2: Collect git history since last release

Run these commands to gather raw material:

```bash
# Fetch all tags from origin to ensure local view is complete
git fetch --tags origin 2>/dev/null || true

# Determine the correct baseline tag using the same prerelease-promotion logic as Step 1:
# - If the latest tag is a prerelease being promoted to stable (e.g. v4.2.0b1 → v4.2.0),
#   use that prerelease tag — narrows the range to commits added since the prerelease cut.
# - For a direct stable-to-stable release, use the previous stable tag.
LATEST_ANY=$(git tag --sort=-version:refname | head -1)
if echo "$LATEST_ANY" | grep -qE '^v[0-9]+\.[0-9]+\.[0-9]+[a-z][0-9]+$'; then
    PREV_TAG="$LATEST_ANY"   # prerelease promotion: baseline is the prerelease tag itself
else
    PREV_TAG=$(git tag --sort=-version:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1)
fi
echo "Previous release tag: $PREV_TAG"

# Sanity-check: count commits in range
COMMIT_COUNT=$(git log ${PREV_TAG}..HEAD --oneline | wc -l)
echo "Commits since ${PREV_TAG}: ${COMMIT_COUNT}"
```

**Before continuing**, verify the tag looks correct:
- If `COMMIT_COUNT` is unexpectedly large (e.g. thousands) or `PREV_TAG` looks very old (e.g. `v0.1.0`
  when the project is clearly on v4.x), local tags are likely stale.
- In that case, stop and tell the user:
  > "The most recent stable tag found locally is `PREV_TAG` with `N` commits since then, which
  > looks too far back. Your local tags may be out of date. Run:
  > `git fetch --tags https://github.com/chunkhound/chunkhound.git`
  > then retry `/release-notes`."
- Do **not** proceed with a stale baseline — the resulting release notes would cover years of
  history instead of the actual delta.

Once the tag looks right, collect the commits:

```bash
# Get all commits since last stable release (subject + body for PR descriptions)
git log ${PREV_TAG}..HEAD --format="--- COMMIT ---%nSubject: %s%nBody:%b" --no-merges

# Also get merge commit subjects (PR titles)
git log ${PREV_TAG}..HEAD --merges --format="PR: %s"
```

## Step 3: Think in features, not commits

**The unit of output is a feature or fix, never a commit.**

Multiple commits often build one feature; one commit may span multiple concerns. Your job is to
synthesize the commit log into a list of *capabilities that changed for the user*.

### 3a: Scale check — map-reduce for large ranges

Before summarizing, check `COMMIT_COUNT` from Step 2.

**If `COMMIT_COUNT` ≤ 100:** proceed with single-pass summarization below.

**If `COMMIT_COUNT` > 100:** use map-reduce:

1. Split the commit range into batches of ≤ 50 commits each (by date or hash order).
2. Spawn one agent per batch (use `superpowers:dispatching-parallel-agents` if available).
   Each agent receives its batch and the same grouping rules below, and returns a candidate
   bullet list plus a **coverage ledger** for every commit in its batch.
3. After all agents complete, merge their candidate lists (de-duplicate overlapping features
   across batches) and merge the ledgers into one combined ledger.
4. Apply Step 4b de-duplication against CHANGELOG.md to the merged candidate list.

**Coverage ledger format** — every commit/PR in the range must appear in exactly one row:

| Commit / PR | Disposition | Note |
|---|---|---|
| `abc1234` feat(llm): add Gemini | included | → "Gemini LLM provider" bullet |
| `def5678` fix(ci): flaky test | skipped | CI-only, no user impact |
| `ghi9012` fix: follow-up to Gemini | folded | → into "Gemini LLM provider" |

Present the coverage ledger to the user alongside the candidate bullet list and wait for
confirmation before proceeding to Step 5.

**How to group:**
- Cluster all commits that contribute to the same feature (e.g. initial impl + fixes + tests +
  follow-ups) into a single bullet.
- If several commits collectively add "Gemini LLM provider", that is one bullet — not three.
- If a feature was added and later had its bugs fixed in the same release, fold the fixes into
  the feature entry (e.g. "Gemini LLM provider with extended thinking support").
- Only list a fix separately when it repairs a regression in a *previously released* version.

**What to include** (user-facing changes):
- New features, commands, providers, parsers, languages → **Added**
- Improvements to existing features → **Enhanced**
- Speed / memory / size improvements → **Performance**
- Bug fixes for regressions from prior releases → **Fixed**
- Breaking changes / removed features → **Breaking Changes** / **Removed**
- Security fixes → **Security**
- Dependency upgrades that affect runtime behavior → **Enhanced** or **Fixed**

**What to exclude** (internal noise):
- CI/CD, build, test, docs, style, chore commits with no user-facing effect
- Version bump and release preparation commits
- Merge commits, fixup/squash commits
- Refactors that don't change observable behavior

**When in doubt:** include — it is easier to delete than to miss a real change.

## Step 4: Rewrite for users

For each feature/fix group, write one benefit-oriented bullet in plain language.

Rules:
- Lead with the **outcome for the user**, not the implementation detail
- One bullet = one logical feature or fix (not one commit)
- **Bold** the feature name or area at the start of each bullet
- Keep bullets to 1–2 sentences max
- Never mention commit hashes, PR numbers, or internal file names in the output

Examples:
- Several commits adding and stabilising a Gemini provider → **Gemini LLM provider** — Google
  Gemini models are now supported for deep code research, including extended thinking mode.
- `fix: replace Unix 'which' with shutil.which` (standalone fix for prior release) →
  **Windows compatibility** — Git binary is now located correctly on Windows.
- `feat(ci): add merge queue gate` → skip entirely (CI-only, no user impact)
- Multiple commits across `feat(embedding)`, fix, and pin → **Matryoshka embeddings** —
  OpenAI-compatible providers now support Matryoshka truncation for flexible vector dimensions;
  default model upgraded to `text-embedding-3-large`.

## Step 4b: De-duplicate against CHANGELOG.md

Before writing any bullet, read the current `CHANGELOG.md` and cross-check every candidate entry
against **all existing versioned sections** (not just `[Unreleased]`).

```bash
cat CHANGELOG.md
```

For each candidate entry, ask: *does an equivalent entry already appear in a prior version section?*

**Rules:**
- If the feature or fix is already documented in any existing section (`[Unreleased]`, a pre-release
  like `[4.1.0b1]`, or a stable release like `[4.0.0]`), **skip it entirely** — do not re-add it.
- If the existing section is `[Unreleased]` or a pre-release (`a1`, `b1`, `rc1`) that is being
  folded into this release, **include its entries** in the new versioned section (they belong here),
  but do not double-count — each entry appears only once.
- A git commit adding feature X that was already shipped in v4.0.0 but was mis-tagged is **not new**
  for this release. Trust the CHANGELOG, not the raw commit range.

After filtering, present a two-column table to the user:

| Entry | Status |
|---|---|
| TwinCAT parser | NEW — not in CHANGELOG |
| Svelte SFC support | SKIP — already in [4.0.0] |
| OpenAI Responses API | FROM [Unreleased] — will fold in |

Wait for the user to confirm the table looks right before proceeding to write the CHANGELOG entry.

## Step 5: Produce the CHANGELOG entry

Format as Keep a Changelog (https://keepachangelog.com):

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Breaking Changes
- **Feature name** — Description of what changed and how to migrate.

### Added
- **Feature name** — What it does and why it matters.

### Enhanced
- **Area** — What improved and the benefit.

### Performance
- **Area** — What is faster/smaller and by how much (if known).

### Fixed
- **Component** — What was broken and what the symptom was.

### Removed
- **Feature** — What was removed and the migration path.

### Security
- **Issue** — What was fixed without disclosing exploitable details.
```

Omit any section that has no entries.

## Step 6: Produce the GitHub Release body

Write a short GitHub Release body:

```markdown
## ChunkHound vX.Y.Z

<1–2 sentence summary of the release theme — what kind of release is this? e.g. "This release focuses on X and Y, adding Z.">

### Highlights
<3–5 bullet points for the most impactful changes — use the same phrasing as CHANGELOG>

---

<full CHANGELOG content from Step 5, starting at ### Breaking Changes>

**Full changelog:** https://github.com/chunkhound/chunkhound/blob/main/CHANGELOG.md
```

## Step 7: Offer to apply

After presenting both outputs, ask the user:

1. **Update CHANGELOG.md?** — Apply these changes atomically:

   a. Insert the new `## [X.Y.Z] - YYYY-MM-DD` section at the top (after the file header).

   b. Replace the existing `## [Unreleased]` section with an empty template.

   c. **Fold pre-release sections:** For every pre-release section whose content was folded into
      this stable release (e.g. `## [4.1.0b1]`), **remove the entire section** from the file body.
      Then, in the link reference block at the bottom of the file, **also remove the corresponding
      line** (e.g. `[4.1.0b1]: https://...`). Leaving a section with no matching reference, or a
      reference with no matching section, creates broken Markdown link integrity.

   d. Add the new stable comparison link and update `[Unreleased]` to point to the new tag:
      ```
      [Unreleased]: https://github.com/chunkhound/chunkhound/compare/vX.Y.Z...HEAD
      [X.Y.Z]: https://github.com/chunkhound/chunkhound/compare/vPREV...vX.Y.Z
      ```
      where `vPREV` is the last stable tag before this release (same `PREV_TAG` from Step 2).

   Before writing, show the user a diff preview of the link reference block so they can verify
   no dangling references remain.

2. **Create GitHub Release?** — Ask the user whether to create the release as a **draft** (safe, review in UI first) or **publish immediately**.

   Write the release body to a temp file, then run:
   ```bash
   # Save release notes to temp file
   cat > /tmp/release_notes_X.Y.Z.md << 'EOF'
   <GitHub Release body from Step 6>
   EOF

   # Create as draft (recommended)
   gh release create vX.Y.Z --draft --title "ChunkHound vX.Y.Z" --notes-file /tmp/release_notes_X.Y.Z.md

   # OR publish immediately (only if user explicitly requested)
   gh release create vX.Y.Z --title "ChunkHound vX.Y.Z" --notes-file /tmp/release_notes_X.Y.Z.md
   ```

   After running, print the URL returned by `gh release create` so the user can open it directly.

   If the user chose draft, remind them to publish when ready:
   ```bash
   gh release edit vX.Y.Z --draft=false
   ```

---

$ARGUMENTS