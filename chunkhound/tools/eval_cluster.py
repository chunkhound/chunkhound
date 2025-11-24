"""Clustering evaluation harness for ChunkHound.

This tool evaluates the K-means + token-bounded clustering behavior used in
deep research, using synthetic corpora generated under:

    .chunkhound/benches/<bench-id>/source/

It derives ground-truth topic labels from directory structure and computes
standard external clustering metrics (ARI, NMI, V-measure, purity) plus
token/cluster-size statistics.

Usage (via Makefile target or directly):
    uv run python -m chunkhound.tools.eval_cluster --help
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from loguru import logger
from sklearn.metrics import (  # type: ignore[import-not-found]
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)

from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.database_factory import create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services.clustering_service import ClusteringService


def _bench_root(bench_id: str) -> Path:
    """Return the source root for a given bench ID."""
    return Path.cwd() / ".chunkhound" / "benches" / bench_id / "source"


def _derive_label(rel_path: Path) -> str:
    """Derive a ground-truth label from a bench-relative path.

    Expected layouts:
        uniform/uniform_topic_1/file_000.txt      -> label: "uniform_topic_1"
        mixed_sizes/budget_pressure/small_000.txt -> label: "budget_pressure"
        noise/noise_000.txt                       -> label: "noise"
    """
    parts = rel_path.parts
    if not parts:
        raise ValueError(f"Cannot derive label from empty path: {rel_path!r}")

    if parts[0] == "uniform" and len(parts) >= 2:
        return parts[1]
    if parts[0] == "mixed_sizes" and len(parts) >= 2:
        return parts[1]
    if parts[0] == "noise":
        return "noise"
    if parts[0] in {"overlap", "cross_topic"} and len(parts) >= 2:
        return parts[1]

    # Fallback: use top-level directory as label
    return parts[0]


def _load_bench_files(bench_id: str) -> tuple[dict[str, str], list[str], list[str]]:
    """Load bench files and derive true labels.

    Returns:
        files:      Mapping of relative path -> file content
        file_keys:  Ordered list of keys used for metrics
        true_labels:True label per file_key entry
    """
    root = _bench_root(bench_id)
    if not root.exists():
        raise RuntimeError(
            f"Bench source directory not found at {root}. "
            "Generate it via: uv run python scripts/generate_cluster_bench.py "
            f"--bench-id {bench_id}"
        )

    files: dict[str, str] = {}
    file_keys: list[str] = []
    true_labels: list[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        key = rel.as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback for non-UTF-8 files: decode bytes with replacement
            raw = path.read_bytes()
            logger.warning(
                f"Non-UTF-8 file encountered in bench '{bench_id}': {key} "
                "- decoding with replacement characters."
            )
            content = raw.decode("utf-8", errors="ignore")
        label = _derive_label(rel)

        files[key] = content
        file_keys.append(key)
        true_labels.append(label)

    if not files:
        raise RuntimeError(f"No files found under bench source directory: {root}")

    logger.info(
        f"Loaded {len(files)} files from bench '{bench_id}' "
        f"with {len(set(true_labels))} topics"
    )

    return files, file_keys, true_labels


def _build_config(bench_id: str, config_path: str | None) -> Config:
    """Construct Config for clustering evaluation.

    Uses project-root configuration (or explicit --config) but routes the
    database path into the bench directory for isolation.
    """
    args: Any | None = None
    if config_path:
        args = argparse.Namespace(config=config_path, path=str(Path.cwd()))

    config = Config(args=args)

    db_path = (
        Path.cwd()
        / ".chunkhound"
        / "benches"
        / bench_id
        / "cluster_eval.db"
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config.database.path = db_path

    return config


async def _cluster_and_evaluate(
    bench_id: str,
    config_path: str | None,
    max_tokens_per_cluster: int,
) -> dict[str, Any]:
    """Run clustering on a bench corpus and compute metrics."""
    files, file_keys, true_labels = _load_bench_files(bench_id)

    # Build config and services bundle (configures registry consistently)
    config = _build_config(bench_id, config_path)

    # Initialize embedding manager and provider (required for embeddings)
    embedding_manager = EmbeddingManager()
    if not config.embedding:
        raise RuntimeError(
            "Embedding provider not configured. "
            "Configure an embedding provider via .chunkhound.json or environment "
            "variables before running clustering evaluation."
        )

    try:
        embedding_provider = EmbeddingProviderFactory.create_provider(config.embedding)
        embedding_manager.register_provider(embedding_provider, set_default=True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"Failed to initialize embedding provider: {exc}")
        raise

    # Initialize LLM manager and provider (used for token estimation)
    if not config.llm:
        raise RuntimeError(
            "LLM provider not configured. "
            "Configure an LLM provider via .chunkhound.json or environment "
            "variables before running clustering evaluation."
        )

    try:
        utility_config, synthesis_config = config.llm.get_provider_configs()
        llm_manager = LLMManager(utility_config, synthesis_config)
        llm_provider = llm_manager.get_synthesis_provider()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"Failed to initialize LLM provider: {exc}")
        raise

    # Configure registry and database services (ensures SerialDatabaseProvider invariants)
    create_services(
        db_path=config.database.path,
        config=config,
        embedding_manager=embedding_manager,
    )

    # Build clustering service with real providers
    clustering_service = ClusteringService(
        embedding_provider=embedding_manager.get_provider(),
        llm_provider=llm_provider,
        max_tokens_per_cluster=max_tokens_per_cluster,
    )

    cluster_groups, metadata = await clustering_service.cluster_files(files)

    # Map file paths to cluster IDs
    cluster_id_by_path: dict[str, int] = {}
    for group in cluster_groups:
        for fp in group.file_paths:
            cluster_id_by_path[fp] = group.cluster_id

    # Align predicted labels with file_keys order
    pred_labels: list[int] = []
    for key in file_keys:
        if key not in cluster_id_by_path:
            raise RuntimeError(
                f"Cluster assignment missing for file {key!r}. "
                "This should not happen if clustering service returns all files."
            )
        pred_labels.append(cluster_id_by_path[key])

    # External clustering metrics
    ari = float(adjusted_rand_score(true_labels, pred_labels))
    nmi = float(normalized_mutual_info_score(true_labels, pred_labels))
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        true_labels, pred_labels
    )

    # Purity computation
    index_by_path = {path: idx for idx, path in enumerate(file_keys)}
    labels_by_cluster: dict[int, list[str]] = defaultdict(list)
    for path, cluster_id in cluster_id_by_path.items():
        idx = index_by_path[path]
        labels_by_cluster[cluster_id].append(true_labels[idx])

    cluster_purities: list[float] = []
    for cluster_id, labels in labels_by_cluster.items():
        counts = Counter(labels)
        majority = max(counts.values())
        purity = float(majority) / float(len(labels)) if labels else 0.0
        cluster_purities.append(purity)
        logger.debug(
            f"Cluster {cluster_id}: size={len(labels)}, "
            f"majority_label={counts.most_common(1)[0][0]}, purity={purity:.3f}"
        )

    purity_global = statistics.mean(cluster_purities) if cluster_purities else 0.0

    # Token and cluster-size statistics
    cluster_token_counts = [group.total_tokens for group in cluster_groups]
    cluster_sizes = [len(group.file_paths) for group in cluster_groups]

    total_files = metadata.get("total_files", len(files))
    total_tokens = metadata.get("total_tokens", sum(cluster_token_counts))
    num_clusters = metadata.get("num_clusters", len(cluster_groups))

    max_tokens_per_cluster_obs = max(cluster_token_counts) if cluster_token_counts else 0
    mean_tokens_per_cluster = (
        statistics.mean(cluster_token_counts) if cluster_token_counts else 0.0
    )

    max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
    mean_cluster_size = statistics.mean(cluster_sizes) if cluster_sizes else 0.0

    topics = sorted(set(true_labels))

    return {
        "bench_id": bench_id,
        "num_files": total_files,
        "topics": topics,
        "metrics": {
            "ari": ari,
            "nmi": float(nmi),
            "homogeneity": float(homogeneity),
            "completeness": float(completeness),
            "v_measure": float(v_measure),
            "purity": purity_global,
        },
        "clusters": {
            "num_clusters": num_clusters,
            "total_tokens": total_tokens,
            "avg_tokens_per_cluster": metadata.get(
                "avg_tokens_per_cluster", int(mean_tokens_per_cluster)
            ),
            "max_tokens_per_cluster": max_tokens_per_cluster_obs,
            "mean_tokens_per_cluster": mean_tokens_per_cluster,
            "max_cluster_size": max_cluster_size,
            "mean_cluster_size": mean_cluster_size,
        },
    }


def _format_human_summary(payload: dict[str, Any]) -> None:
    """Print a concise human-readable summary."""
    bench_id = payload["bench_id"]
    num_files = payload["num_files"]
    topics = payload["topics"]
    metrics = payload["metrics"]
    clusters = payload["clusters"]

    print(f"Bench: {bench_id}")
    print(f"Files: {num_files}, topics: {len(topics)} ({', '.join(topics)})")

    print("\nExternal clustering metrics:")
    print(f"  ARI         : {metrics['ari']:.4f}")
    print(f"  NMI         : {metrics['nmi']:.4f}")
    print(
        "  Homogeneity : "
        f"{metrics['homogeneity']:.4f}, "
        "Completeness : "
        f"{metrics['completeness']:.4f}, "
        "V-measure : "
        f"{metrics['v_measure']:.4f}"
    )
    print(f"  Purity      : {metrics['purity']:.4f}")

    print("\nCluster/budget stats:")
    print(f"  num_clusters           : {clusters['num_clusters']}")
    print(
        "  total_tokens           : "
        f"{clusters['total_tokens']:,}, "
        "avg_tokens_per_cluster : "
        f"{clusters['avg_tokens_per_cluster']:,}"
    )
    print(
        "  max_tokens_per_cluster : "
        f"{clusters['max_tokens_per_cluster']:,}, "
        "mean_tokens_per_cluster: "
        f"{clusters['mean_tokens_per_cluster']:.1f}"
    )
    print(
        "  max_cluster_size       : "
        f"{clusters['max_cluster_size']}, "
        "mean_cluster_size      : "
        f"{clusters['mean_cluster_size']:.1f}"
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clustering evaluation harness for ChunkHound.\n"
            "Evaluates K-means + token-bounded clustering on synthetic corpora "
            "under .chunkhound/benches/<bench-id>/source."
        )
    )
    parser.add_argument(
        "--bench-id",
        type=str,
        default="cluster-stress-dev",
        help="Benchmark ID (default: cluster-stress-dev).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a ChunkHound config file.",
    )
    parser.add_argument(
        "--max-tokens-per-cluster",
        type=int,
        default=30000,
        help="Maximum tokens allowed per cluster (default: 30000).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON metrics report.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


async def _async_main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        payload = await _cluster_and_evaluate(
            bench_id=args.bench_id,
            config_path=args.config,
            max_tokens_per_cluster=args.max_tokens_per_cluster,
        )
    except Exception as exc:
        logger.error(f"Clustering evaluation failed: {exc}")
        return 1

    _format_human_summary(payload)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON metrics report to {output_path}")

    return 0


def main() -> None:
    """Entry point for python -m chunkhound.tools.eval_cluster."""
    raise SystemExit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
