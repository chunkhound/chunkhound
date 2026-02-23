"""Snapshot command argument parser for ChunkHound CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common_arguments import add_common_arguments, add_config_arguments


def add_snapshot_subparser(subparsers: Any) -> argparse.ArgumentParser:
    """Add snapshot command subparser to the main parser."""
    snapshot_parser = subparsers.add_parser(
        "snapshot",
        help="Emit snapshot chunk-systems from an already-indexed DB",
        description=(
            "Reads files/chunks/embeddings from the configured database and emits "
            "snapshot chunk-systems artifacts without reindexing or embedding generation."
        ),
    )

    snapshot_parser.add_argument(
        "scope_root",
        nargs="?",
        type=Path,
        default=Path("."),
        help=(
            "Scope root path-prefix for selecting DB rows (default: current directory, "
            "i.e. no prefix filter)"
        ),
    )
    snapshot_parser.add_argument(
        "--scope-root",
        action="append",
        dest="scope_roots",
        type=Path,
        default=None,
        help=(
            "Repeatable scope root path-prefix for selecting DB rows "
            "(union of scopes). If provided, do not also pass the positional "
            "scope_root."
        ),
    )

    add_common_arguments(snapshot_parser)
    add_config_arguments(snapshot_parser, ["database", "llm"])

    snapshot_parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for snapshot artifacts (required)",
    )

    snapshot_parser.add_argument(
        "--tui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Launch terminal UI (TUI) after snapshot when running interactively "
            "(default: enabled)"
        ),
    )
    snapshot_parser.add_argument(
        "--out-dir-mode",
        type=str,
        choices=["prompt", "reuse", "force"],
        default="prompt",
        help=(
            "Behavior when --out-dir exists and is non-empty. "
            "prompt: ask to reuse/force/abort (default). "
            "reuse: validate required artifacts and skip compute. "
            "force: recompute and overwrite known snapshot artifacts in place."
        ),
    )
    snapshot_parser.add_argument(
        "--refresh-run-metadata",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When reusing an existing snapshot out-dir, refresh snapshot.run.json with "
            "best-effort git HEAD SHAs and current configured LLM synthesis provider/model "
            "(including any --llm-* CLI overrides). Default: disabled."
        ),
    )

    snapshot_parser.add_argument(
        "--themes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compatibility flag (ignored in isolated chunk-systems mode)",
    )
    snapshot_parser.add_argument(
        "--systems",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compatibility flag (ignored in isolated chunk-systems mode)",
    )

    snapshot_parser.add_argument(
        "--chunk-systems",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Emit chunk-level systems artifacts "
            "(snapshot.chunk_systems.*). Default: disabled"
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-k",
        type=int,
        default=30,
        help="(chunk systems) kNN fanout k (default: 30)",
    )
    snapshot_parser.add_argument(
        "--chunk-systems-tau",
        type=float,
        default=0.25,
        help="(chunk systems) cosine similarity cutoff tau (default: 0.25)",
    )
    snapshot_parser.add_argument(
        "--chunk-systems-min-degree",
        type=int,
        default=0,
        help=(
            "(chunk systems) Optional recall-oriented fallback: ensure each node has at "
            "least this undirected degree by adding edges from directed kNN lists "
            "(default: 0, disabled)"
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-fallback-tau",
        type=float,
        default=None,
        help=(
            "(chunk systems) Similarity cutoff for fallback edges when "
            "--chunk-systems-min-degree > 0. Defaults to --chunk-systems-tau."
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-fallback-path-mode",
        type=str,
        choices=["any", "same_file", "same_dir"],
        default="any",
        help=(
            "(chunk systems) Restrict min-degree fallback edges by path. "
            "any: allow cross-path fallback edges (default). "
            "same_file: only within the same file path. "
            "same_dir: only within the same parent directory."
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-max-nodes",
        type=int,
        default=20000,
        help="(chunk systems) safety cap on chunk nodes (default: 20000)",
    )
    snapshot_parser.add_argument(
        "--chunk-systems-min-cluster-size",
        type=int,
        default=2,
        help=(
            "(chunk systems) Reporting/view filter: only include clusters with size >= N "
            "in snapshot.chunk_systems.md and in snapshot.chunk_systems.pruned.json. "
            "Default: 2 (drop singleton clusters). Use 1 to disable pruning."
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-partitioner",
        type=str,
        choices=["auto", "cc", "leiden"],
        default="auto",
        help="(chunk systems) graph partitioner (default: auto)",
    )
    snapshot_parser.add_argument(
        "--chunk-systems-leiden-resolution",
        type=float,
        default=1.0,
        help="(chunk systems) Leiden resolution when partitioner=leiden (default: 1.0)",
    )
    snapshot_parser.add_argument(
        "--chunk-systems-leiden-seed",
        type=int,
        default=0,
        help="(chunk systems) Leiden random seed (default: 0)",
    )
    snapshot_parser.add_argument(
        "--chunk-systems-leiden-resolutions",
        type=str,
        default=None,
        help=(
            "(chunk systems) Optional CSV of Leiden resolutions to try when "
            "partitioner=auto (e.g. '0.5,1.0,2.0')"
        ),
    )

    snapshot_parser.add_argument(
        "--chunk-systems-auto-selector",
        type=str,
        choices=["legacy", "objective_stable"],
        default="legacy",
        help=(
            "(chunk systems) Experimental: Leiden auto-selection strategy when "
            "partitioner=auto. legacy: existing heuristic (default). objective_stable: "
            "pick a bounded, seed-stable partition using Leiden objective + ARI stability."
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-auto-stability-seeds",
        type=str,
        default="0,1,2",
        help=(
            "(chunk systems) Experimental (objective_stable): CSV of Leiden seeds to run "
            "per resolution for stability estimation (default: 0,1,2)"
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-auto-stability-min",
        type=float,
        default=0.90,
        help=(
            "(chunk systems) Experimental (objective_stable): minimum average ARI across "
            "seed runs to treat a resolution as stable (default: 0.90)"
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-auto-min-avg-size",
        type=int,
        default=10,
        help=(
            "(chunk systems) Experimental (objective_stable): lower bound for average "
            "cluster size (default: 10)"
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-auto-max-avg-size",
        type=int,
        default=80,
        help=(
            "(chunk systems) Experimental (objective_stable): upper bound for average "
            "cluster size (default: 80)"
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-auto-max-largest-frac",
        type=float,
        default=0.20,
        help=(
            "(chunk systems) Experimental (objective_stable): reject partitions where the "
            "largest cluster exceeds this fraction of nodes (default: 0.20)"
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-auto-max-singleton-frac",
        type=float,
        default=0.02,
        help=(
            "(chunk systems) Experimental (objective_stable): reject partitions where "
            "singleton nodes exceed this fraction of nodes (default: 0.02)"
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-auto-write-sweep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "(chunk systems) Experimental (objective_stable): write an extra JSON artifact "
            "snapshot.chunk_systems.partition_sweep.json with per-resolution sweep results "
            "(default: disabled)"
        ),
    )

    snapshot_parser.add_argument(
        "--chunk-systems-write-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "(chunk systems) Experimental: write the chunk-level graph used for partitioning "
            "as JSONL artifacts (snapshot.chunk_systems.graph.*.jsonl). Default: disabled."
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-write-adjacency",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "(chunk systems) Experimental: write system-to-system adjacency meta-graph with "
            "evidence (snapshot.chunk_systems.system_adjacency.json). Default: disabled."
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-viz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "(chunk systems) Experimental: write a self-contained HTML visualization of the "
            "system adjacency graph with evidence panel "
            "(snapshot.chunk_systems.viz.html). Implies --chunk-systems-write-adjacency. "
            "Default: disabled."
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-adjacency-evidence-k",
        type=int,
        default=5,
        help=(
            "(chunk systems) Experimental (adjacency): max evidence edges to store per system "
            "pair (default: 5)"
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-adjacency-max-neighbors",
        type=int,
        default=20,
        help=(
            "(chunk systems) Experimental (adjacency): cap neighbors per system by weight_sum "
            "(default: 20)"
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-adjacency-mode",
        type=str,
        choices=["mutual", "directed"],
        default="mutual",
        help=(
            "(chunk systems) Experimental (adjacency): which graph to aggregate into "
            "system links. 'mutual' uses mutual-kNN undirected edges (default). "
            "'directed' uses directed kNN arcs aggregated into directed system links."
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-system-groups",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "(chunk systems) Experimental (viz): compute and write multi-resolution "
            "systems-of-systems groupings (snapshot.chunk_systems.system_groups.json). "
            "Implied by --chunk-systems-viz. Default: disabled."
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-system-groups-resolutions",
        type=str,
        default="0.25,0.5,0.75,1,1.25,1.5,2,2.5,3",
        help=(
            "(chunk systems) Experimental (system groups): CSV of Leiden resolutions "
            "to precompute for the system adjacency meta-graph."
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-system-groups-seed",
        type=int,
        default=0,
        help="(chunk systems) Experimental (system groups): Leiden random seed (default: 0)",
    )
    snapshot_parser.add_argument(
        "--chunk-systems-system-groups-weight",
        type=str,
        choices=["weight_sum", "weight_max"],
        default="weight_sum",
        help=(
            "(chunk systems) Experimental (system groups): which system-link score "
            "to use for meta-Leiden grouping (default: weight_sum)"
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-system-groups-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "(chunk systems) Experimental (system groups): label each system-group at each "
            "granularity via LLM and write snapshot.chunk_systems.system_group_labels.json. "
            "Uses --llm-dry-run if set. Default: disabled."
        ),
    )
    snapshot_parser.add_argument(
        "--chunk-systems-system-groups-labels-prompt-mode",
        type=str,
        choices=["systems_only", "full"],
        default="systems_only",
        help=(
            "(chunk systems) Experimental (system groups labels): how much evidence to include "
            "in LLM prompts when labeling system-groups. "
            "'systems_only' includes only member system names (default). "
            "'full' also includes aggregated top files/symbols."
        ),
    )

    snapshot_parser.add_argument(
        "--view",
        action="append",
        default=None,
        help="Compatibility flag (ignored in isolated chunk-systems mode)",
    )

    snapshot_parser.add_argument(
        "--embedding-provider",
        type=str,
        default=None,
        help="Embedding provider selector for stored embeddings (e.g., openai)",
    )
    snapshot_parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model selector for stored embeddings",
    )
    snapshot_parser.add_argument(
        "--embedding-dims",
        type=int,
        default=1536,
        help="Embedding dims selector for stored embeddings (default: 1536)",
    )
    snapshot_parser.add_argument(
        "--matryoshka-dims",
        type=int,
        default=None,
        help=(
            "Optional effective embedding dimensionality to use by truncating "
            "stored embeddings to first N dims; must be <= --embedding-dims"
        ),
    )

    snapshot_parser.add_argument(
        "--llm-dry-run",
        action="store_true",
        help=(
            "(labeler=llm) Emit LLM prompts only (no calls); writes placeholder labels"
        ),
    )
    snapshot_parser.add_argument(
        "--labeler",
        type=str,
        choices=["llm", "heuristic"],
        default="llm",
        help=(
            "Labeling mode for chunk-systems. "
            "LLM runs only when --chunk-systems is enabled and --labeler is explicitly "
            "provided on command line."
        ),
    )
    snapshot_parser.add_argument(
        "--md-labels",
        type=str,
        choices=["heuristic", "llm"],
        default="heuristic",
        help=(
            "Which labels to render into snapshot.chunk_systems.md. "
            "Use 'llm' to overlay snapshot.labels.json labels."
        ),
    )

    snapshot_parser.add_argument(
        "--llm-label-batching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "(labeler=llm) Batch multiple labeling tasks into a single LLM call. "
            "Default: enabled."
        ),
    )
    snapshot_parser.add_argument(
        "--llm-label-batch-max-items",
        type=int,
        default=16,
        help="(labeler=llm) Max labels per LLM call when batching (default: 16)",
    )
    snapshot_parser.add_argument(
        "--llm-label-batch-max-tokens",
        type=int,
        default=20000,
        help="(labeler=llm) Approx max input tokens per LLM call when batching (default: 20000)",
    )
    snapshot_parser.add_argument(
        "--llm-label-concurrency",
        type=int,
        default=None,
        help=(
            "(labeler=llm) Max concurrent labeling calls. "
            "Default: provider recommendation."
        ),
    )

    return snapshot_parser


__all__ = ["add_snapshot_subparser"]
