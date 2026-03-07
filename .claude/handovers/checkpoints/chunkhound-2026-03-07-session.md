# Session: chunkhound - 2026-03-07

## Checkpoint 1 — 06:35
- **Focus:** Clone and index 6 Elixir stress-test repos for parser validation
- **Committed:** uncommitted — configs only (`.chunkhound.json` in each repo, not in chunkhound repo)
- **Decisions:**
  - Used absolute DB paths in all `.chunkhound.json` configs to avoid CWD-relative path gotcha
  - Ecto's first index run (with embeddings) wrote to chunkhound's own `.chunkhound` DB due to relative path — needs re-index with fixed config
  - GPU overheating (70°C, fan 100%) — must index one repo at a time with cooldown between
- **Blockers:** GPU temp too high for batch embedding — wait for cooldown before continuing
- **Next:** Index all 6 repos with embeddings, one at a time, smallest first:
  1. Finch (820 chunks)
  2. Swoosh (1,635 chunks)
  3. Oban (2,424 chunks) — also needs `.chunkhound.json` updated (still has old config without absolute path or database section)
  4. Ecto (5,492 chunks) — re-index with correct DB path
  5. Absinthe (5,785 chunks)
  6. Livebook (10,001 chunks)

## Repos Cloned
All at `/home/johan/workbench/elixir/`:
| Repo | GitHub | Files | Chunks (no-embed) |
|------|--------|-------|-------------------|
| ecto | elixir-ecto/ecto | 147 | 5,492 |
| oban | sorentwo/oban | 159 | 2,424 |
| absinthe | absinthe-graphql/absinthe | 622 | 5,785 |
| livebook | livebook-dev/livebook | 646 | 10,001 |
| swoosh | swoosh/swoosh | 135 | 1,635 |
| finch | sneako/finch | 48 | 820 |

## Config Status
- ✅ ecto — absolute path config written
- ❌ oban — still has old config (no database section, no absolute path) — needs update
- ✅ absinthe — absolute path config written
- ✅ livebook — absolute path config written
- ✅ swoosh — absolute path config written
- ✅ finch — absolute path config written

## Index Command Template
```bash
uv run chunkhound index --force-reindex /home/johan/workbench/elixir/{repo}
```
