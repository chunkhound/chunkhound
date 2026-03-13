# Fork PR Guide — ChunkHound

How to manage pull requests from a fork (`johan--/chunkhound`) against upstream (`chunkhound/chunkhound`).

## Remotes

```
origin    → johan--/chunkhound (your fork)
upstream  → chunkhound/chunkhound (upstream, where PRs target)
```

Set up once:
```bash
git remote add upstream git@github.com:chunkhound/chunkhound.git
```

## Creating a PR

1. Create a feature branch from your fork's main (or upstream/main):
   ```bash
   git fetch upstream main
   git checkout -b feature-name upstream/main
   ```

2. Develop, commit, push to your fork:
   ```bash
   git push -u origin feature-name
   ```

3. Create PR targeting upstream main:
   ```bash
   gh pr create --repo chunkhound/chunkhound --base main \
     --title "feat: description" --body "..."
   ```

## Before Pushing (Every Time)

```bash
# Mandatory — smoke tests
uv run pytest tests/test_smoke.py -v -n auto

# Mandatory before pushing to a PR branch
uv run pytest tests/ -v
```

Tests first, push second. Always.

## Resolving Merge Conflicts

The PR targets `upstream/main`, not `origin/main`. Your fork's main can drift behind upstream. Always rebase onto the right remote.

```bash
# 1. Fetch upstream
git fetch upstream main

# 2. Rebase onto upstream/main (NOT origin/main)
git rebase upstream/main

# 3. Resolve any conflicts
#    Edit files, then: git add <file> && git rebase --continue

# 4. Run tests after rebase
uv run pytest tests/ -v

# 5. Force-push (required after rebase)
git push --force-with-lease origin feature-name
```

### Common mistake

Rebasing onto `origin/main` instead of `upstream/main`. If your fork's main is behind upstream, the rebase won't incorporate all of upstream's changes, and GitHub will still show conflicts.

### Checking for conflicts without modifying anything

```bash
# Dry-run merge to see conflicts
git merge-tree $(git merge-base HEAD upstream/main) HEAD upstream/main | grep 'CONFLICT'

# Check GitHub's view via API
gh api repos/chunkhound/chunkhound/pulls/NNN --jq '{mergeable, mergeable_state}'
```

## CI on Fork PRs

The workflow (`.github/workflows/smoke-tests.yml`) runs on `pull_request` — all PRs, including forks.

**Maintainer approval required:** GitHub requires a maintainer to approve workflow runs for first-time contributors from forks. This is a repo-level security setting, not related to what files changed.

- Every force-push resets CI approval — a maintainer must re-approve
- Ask a reviewer to approve the workflow run in the PR's Checks tab
- CI runs on 3 platforms: Ubuntu, macOS, Windows (Python 3.11)

## Review Blockers

GitHub requires all "Changes requested" reviews to be resolved before merging. A second reviewer's "Approve" does **not** override another reviewer's "Changes requested."

To unblock:
- The original reviewer updates their review to "Approved"
- Or a repo admin dismisses the stale review

**GitHub hides the conflict banner** when a review blocker is active. If merging is blocked but you don't see a conflict warning, check:
```bash
gh api repos/chunkhound/chunkhound/pulls/NNN --jq '{mergeable, mergeable_state}'
```

## Syncing Your Fork

Not required for PRs (they target upstream directly), but keeps your fork clean:

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

## Merge Checklist

Before asking reviewers for final approval:

- [ ] Branch rebased onto `upstream/main` (not `origin/main`)
- [ ] No merge conflicts (`mergeable_state: "clean"`)
- [ ] Smoke tests pass locally
- [ ] Full test suite passes locally
- [ ] All reviews show "Approved" (no stale "Changes requested")
- [ ] CI green (workflow run approved by maintainer)
