#!/usr/bin/env python
"""Generate synthetic corpora to stress ChunkHound clustering.

Creates token-heavy and topic-structured files under:
    .chunkhound/benches/<bench-id>/source/

This is a one-off utility intended for local benchmarking. The generated
bench is git-ignored and can be reused across runs.

Usage:
    uv run python scripts/generate_cluster_bench.py \
        --bench-id cluster-stress-dev
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def _project_root() -> Path:
    """Return the repository root (directory containing this script)."""
    return Path(__file__).resolve().parents[1]


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _generate_topic_block(topic: str, size: int) -> str:
    """Generate a simple block of text for a given topic and approximate size."""
    base = (
        f"This file belongs to the {topic} topic. "
        "It is used to stress-test ChunkHound clustering by providing "
        "semantically coherent content with repeated markers. "
    )
    # Repeat until we reach the desired character size
    chunks: list[str] = []
    total = 0
    while total < size:
        chunks.append(base)
        total += len(base)
    return "".join(chunks)


def _generate_realistic_topic_block(
    topic: str, shared_context: str, specific_signals: list[str], size: int
) -> str:
    """Generate a more realistic, overlapping topic block.

    This mimics design docs / ADRs where multiple services share vocabulary,
    but each topic has its own emphasis and signals.
    """
    signals = ", ".join(specific_signals)
    base = (
        f"This document describes the {topic} subsystem in a production codebase. "
        f"It shares concerns with {shared_context} and frequently mentions {signals}. "
        "The text imitates design docs, incident postmortems, and architecture notes. "
    )
    chunks: list[str] = []
    total = 0
    while total < size:
        chunks.append(base)
        total += len(base)
    return "".join(chunks)


def _add_uniform_topics(root: Path, num_topics: int, files_per_topic: int) -> None:
    """Create several topics with uniform file sizes."""
    for t in range(1, num_topics + 1):
        topic_name = f"uniform_topic_{t}"
        for i in range(files_per_topic):
            rel = Path("uniform") / topic_name / f"file_{i:03d}.txt"
            content = _generate_topic_block(topic_name, size=1500)
            _write_file(root / rel, content)


def _add_mixed_sizes(root: Path, topic: str, small_count: int, large_count: int) -> None:
    """Create a mix of small and large files for a single topic."""
    # Small files
    for i in range(small_count):
        rel = Path("mixed_sizes") / topic / f"small_{i:03d}.txt"
        content = _generate_topic_block(topic, size=500)
        _write_file(root / rel, content)

    # Large files
    for i in range(large_count):
        rel = Path("mixed_sizes") / topic / f"large_{i:03d}.txt"
        content = _generate_topic_block(topic, size=8000)
        _write_file(root / rel, content)


def _add_noise_files(root: Path, count: int) -> None:
    """Create noise files with very weak topic signal."""
    base = (
        "This is a largely unstructured document with various unrelated facts. "
        "It is intended to act as noise in clustering benchmarks and should not "
        "form a strong coherent cluster by itself. "
    )
    for i in range(count):
        rel = Path("noise") / f"noise_{i:03d}.txt"
        content = base * 40
        _write_file(root / rel, content)


def _add_overlapping_search_topics(
    root: Path, files_per_topic: int, size: int
) -> None:
    """Create several search-related topics with overlapping vocabulary.

    These are intentionally confusable (shared words like 'search', 'index',
    'ranking') to better approximate real-world services.
    """
    shared_context = (
        "search infrastructure, vector indexes, ranking pipelines, and logging"
    )
    topics = [
        ("search_indexing", ["index builds", "incremental updates", "backfills"]),
        ("search_ranking", ["learning-to-rank models", "click logs", "A/B tests"]),
        ("vector_search", ["embedding indexes", "ANN queries", "reranking"]),
    ]

    for topic, signals in topics:
        for i in range(files_per_topic):
            rel = Path("overlap") / topic / f"file_{i:03d}.txt"
            content = _generate_realistic_topic_block(
                topic=topic,
                shared_context=shared_context,
                specific_signals=signals,
                size=size,
            )
            _write_file(root / rel, content)


def _add_cross_topic_bridge_files(root: Path, count: int) -> None:
    """Create files that deliberately mix multiple topics.

    These act as bridge documents that mention several services in the same
    file, making clustering more challenging and more realistic.
    """
    topic_pairs = [
        ("uniform_topic_1", "uniform_topic_2"),
        ("uniform_topic_2", "uniform_topic_3"),
        ("uniform_topic_3", "uniform_topic_4"),
        ("uniform_topic_1", "budget_pressure"),
        ("vector_search", "search_ranking"),
        ("search_indexing", "budget_pressure"),
    ]

    base = (
        "This document describes how two subsystems interact in a real-world "
        "architecture. It covers shared incidents, joint rollouts, and how "
        "metrics are interpreted when ownership is split across teams. "
    )

    for i in range(count):
        t1, t2 = topic_pairs[i % len(topic_pairs)]
        pair_name = f"{t1}_AND_{t2}"
        rel = Path("cross_topic") / pair_name / f"file_{i:03d}.txt"
        content = (
            base
            + f"It focuses on the interaction between {t1} and {t2}, including "
            "trade-offs, incident patterns, and ambiguous ownership boundaries. "
        )
        # Make files moderately long so they affect token budgets
        content = content * 10
        _write_file(root / rel, content)


def _build_corpora(root: Path) -> None:
    """Create all synthetic corpora variants in the given bench root."""
    # 1) Several uniform topics, medium-size files
    _add_uniform_topics(root, num_topics=4, files_per_topic=80)

    # 2) One topic with mixed file sizes to pressure token budgets
    _add_mixed_sizes(root, topic="budget_pressure", small_count=120, large_count=12)

    # 3) Noise files to degrade cluster separation
    _add_noise_files(root, count=60)

    # 4) Overlapping search topics with shared vocabulary to approximate
    #    real-world microservices (indexing / ranking / vector search).
    _add_overlapping_search_topics(root, files_per_topic=40, size=2000)

    # 5) Cross-topic bridge documents that explicitly mention multiple
    #    subsystems, making cluster boundaries fuzzier.
    _add_cross_topic_bridge_files(root, count=40)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic corpora under .chunkhound/benches/<bench-id>/source "
            "to stress ChunkHound clustering."
        )
    )
    parser.add_argument(
        "--bench-id",
        type=str,
        default="cluster-stress-dev",
        help="Benchmark ID (default: cluster-stress-dev).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    root = _project_root() / ".chunkhound" / "benches" / args.bench_id / "source"
    root.mkdir(parents=True, exist_ok=True)
    _build_corpora(root)
    print(f"Generated clustering bench corpus under: {root}")


if __name__ == "__main__":
    main()
