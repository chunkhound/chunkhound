# Snapshot Chunk-Systems (Isolated Branch) Spec

## Status
- Branch: `feat/snapshot-chunk-systems-isolated`
- Scope: extracted `snapshot --chunk-systems` implementation
- Date: 2026-02-17

## Purpose
This branch isolates chunk-systems analysis so it can be improved and reviewed without the broader snapshot/themes/systems and gap-tool surface.

## Scope
### In
- Chunk-systems compute from scoped, pre-indexed embeddings
- Partitioner modes: `auto`, `cc`, `leiden`
- Optional LLM labeling with batching
- Deterministic JSON/MD artifacts
- Runtime progress tasks for DB fetch, compute, and LLM labeling

### Out
- `snapshot.themes.*`
- `snapshot.systems.*`
- snapshot-eval workflows
- gap command/tooling

## CLI Contract
Representative invocation:

```bash
uv run chunkhound snapshot <scope_root> \
  --config <config_path> \
  --out-dir <out_dir> \
  --chunk-systems \
  --embedding-provider <provider> \
  --embedding-model <model> \
  --embedding-dims <dims> \
  [--matryoshka-dims <dims>] \
  [--chunk-systems-k <int>] \
  [--chunk-systems-tau <float>] \
  [--chunk-systems-min-degree <int>] \
  [--chunk-systems-fallback-tau <float>] \
  [--chunk-systems-fallback-path-mode any|same_file|same_dir] \
  [--chunk-systems-max-nodes <int>] \
  [--chunk-systems-min-cluster-size <int>] \
  [--chunk-systems-partitioner auto|cc|leiden] \
  [--chunk-systems-leiden-resolution <float>] \
  [--chunk-systems-leiden-seed <int>] \
  [--chunk-systems-leiden-resolutions <csv>] \
  [--labeler llm|heuristic] \
  [--llm-dry-run] \
  [--llm-label-batching|--no-llm-label-batching] \
  [--llm-label-batch-max-items <int>] \
  [--llm-label-batch-max-tokens <int>] \
  [--md-labels heuristic|llm]
```

Compatibility-only flags retained in isolated mode:
- `--themes/--no-themes`
- `--systems/--no-systems`
- `--view`

Default behavior:
- `--chunk-systems-min-cluster-size` defaults to `2` (singleton clusters are dropped from the operator view by default).

## Architecture
- Orchestration: `chunkhound/api/cli/commands/snapshot.py`
- Parser: `chunkhound/api/cli/parsers/snapshot_parser.py`
- Compute core: `chunkhound/snapshot/chunk_systems.py`
- Local partition primitives: `chunkhound/snapshot/partitioning.py`

Decoupling decision:
- Snapshot chunk-systems uses local partitioning primitives and does not depend on `chunkhound/gap/*`.

## Data Flow
1. Resolve selector (`provider/model/dims`) and scope roots.
2. Read scoped code chunks + vectors from DuckDB.
3. Optional Matryoshka truncation.
4. Build directed kNN neighbors and mutual edges.
   - Optional recall fallback: add extra edges to reach a minimum degree per node.
5. Partition graph (`auto|cc|leiden`).
6. Emit base chunk-system artifacts.
7. Optionally label chunk-systems via LLM batch requests.
8. Overlay LLM labels into markdown when requested.
9. Finalize run metrics/stages.

## Progress UX
Rich progress tasks are first-class runtime output:
- `Snapshot: DB fetch`
- `Snapshot: chunk systems` (with kNN advancement)
- `Snapshot: label chunk systems` (batch + per-item progress)

Status reporting is not modeled as warning logs.

## Artifacts
- `snapshot.run.json` (`snapshot.run.v1`)
- `snapshot.chunk_systems.json` (`snapshot.chunk_systems.v1`)
- `snapshot.chunk_systems.pruned.json` (when `--chunk-systems-min-cluster-size > 1`)
- `snapshot.chunk_systems.dropped.json` (when `--chunk-systems-min-cluster-size > 1`)
- `snapshot.chunk_systems.md`
- `snapshot.labels.json` (`snapshot.labels.v1`, only when LLM labeler path is active)

## Validation Coverage
Focused tests in this branch:
- `tests/test_snapshot_chunk_systems.py`
- `tests/test_snapshot_chunk_systems_out_dir_contract.py`
- `tests/test_snapshot_llm_label_batching.py`
- `tests/test_snapshot_llm_missing_fails_fast.py`
- `tests/test_snapshot_missing_embeddings_fails_fast.py`
- `tests/test_snapshot_multi_scope_contract.py`
- smoke: `tests/test_smoke.py`

## Known Limits
- DuckDB path is the supported DB reader in this isolated implementation.
- Rich progress bars require a TTY-compatible terminal.
- LLM labeling requires LLM config unless `--llm-dry-run` or `--labeler heuristic`.

## Rationale
This extraction provides an intentionally narrow development target for:
- community quality improvements (kNN/partition tuning),
- LLM batch labeling reliability,
- operator UX and observability hardening,
without coupling changes to broader snapshot/gap domains.
