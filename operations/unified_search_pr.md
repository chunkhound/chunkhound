# feat(search): stream transient git and websearch in-process

## Summary

- Replace bulk git-diff embedding and the `_quickresearch` subprocess with one in-process pipeline: streamed chunk batches → bounded `VectorCache` (XXH3 keys, float32, LRU + last-access TTL) → top-K heap → optional both-mode merge with DB HNSW. Git history and fetched pages search without temp files or new config knobs.
- `TransientSearchService` overlays streaming semantic search on the existing DB service. Hybrid (`vector_source=both`) freezes complete source snapshots, ranks consistently across pages, fuses full snapshots (not partial heaps), and keeps `total` numeric: a lower bound while a source has more, exact once both are exhausted. Stream failures cancel sibling work and propagate.
- Git liveness timeout now counts only time blocked on git, not embedding consumer latency. Stderr drain is bounded. Argument-injection guards and empty-tree retry stay on the streaming runner. Websearch no longer writes fetched pages to disk; research uses an in-memory page stream and a progress-event interface.

## Why

Issue [#336](https://github.com/chunkhound/chunkhound/issues/336): large commit ranges and web pages were buffered, truncated (`MAX_DIFF_CHUNKS`), or researched in a child process with tmpdirs. Streaming ingest keeps memory in the same order for 500 vs 5k chunks, lets a chunk past the old cap win if it matches, and removes the subprocess/disk path.

## User-visible / contract

- CLI/MCP search and research flags unchanged (no new cache/overfetch settings).
- `--last-n N --vector-source both` no longer prints `Results: 10 of None`.
- No truncation warning for large diffs; no files written for fetched pages.
- MCP websearch schema and error mapping (URLError → MCPError, empty results, timeout, cancel) stay; research runs in-process.

## Removed

- `run_git_diff` / `_communicate_git_diff` (callers moved to `stream_git_diff_file_blocks`)
- `_quickresearch`, `fetch_and_save` / tmpdir postprocess, `IndexingConfig.reset_user_include_exclude`
- Always-None truncation plumbing after `MAX_DIFF_CHUNKS`

## Test plan

- [ ] `uv run pytest tests/test_smoke.py -v -n auto`
- [ ] `uv run pytest tests/unit/test_transient_search_service.py tests/unit/test_vector_cache.py tests/unit/test_web_research_service.py tests/unit/test_diff_aware_search_service.py tests/unit/test_git_diff_runner.py -v`
- [ ] `uv run pytest tests/integration/test_websearch_command.py tests/integration/test_deep_research_with_diff.py tests/integration/test_websearch_mcp.py -v`
- [ ] Semantic search over a large `--commit-range` (thousands of chunks): first page returns, no 30s git kill during embedding
- [ ] `--vector-source both` pagination: ranking stable across pages, `total` is a number
- [ ] `websearch`: no files under tmpdir for fetched pages; URLError / empty-results exits unchanged
- [ ] Concurrent searches sharing the cache: disjoint queries, no mixed-provider hits
