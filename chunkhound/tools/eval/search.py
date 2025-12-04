"""Evaluation harness for ChunkHound search.

This tool builds small synthetic corpora that exercise every parser-supported
language, runs regex and semantic searches, and computes retrieval metrics
plus latency statistics.

Usage (via Makefile target or directly):
    uv run python -m chunkhound.tools.eval.search --help
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.core.types.common import Language
from chunkhound.database_factory import DatabaseServices, create_services
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from chunkhound.utils.windows_constants import IS_WINDOWS, WINDOWS_DB_CLEANUP_DELAY

from .language_samples import QueryDefinition, create_corpus, parse_languages_arg
from .metrics import (
    AggregateMetrics,
    EvalResult,
    QueryMetrics,
    aggregate_metrics,
    build_json_payload,
    format_human_summary,
)


def _cleanup_services(services: DatabaseServices | None) -> None:
    """Best-effort database cleanup with Windows handle release.

    Ensures DuckDB connections are closed before temporary directories are
    removed so Windows can delete chunks.db without WinError 32.
    """
    if services is None:
        return

    provider = services.provider
    try:
        if hasattr(provider, "close"):
            provider.close()
        elif hasattr(provider, "disconnect"):
            provider.disconnect()
    except Exception as e:  # pragma: no cover - defensive cleanup
        logger.warning(f"Error during eval_search database cleanup: {e}")

    # Help the interpreter release any remaining references promptly
    gc.collect()

    # On Windows, allow a short delay for the OS to release file handles
    if IS_WINDOWS:
        time.sleep(WINDOWS_DB_CLEANUP_DELAY)


def _build_config(project_dir: Path, config_path: str | None) -> Config:
    """Construct Config for the evaluation run.

    When a config file is provided, it is passed via a minimal argparse-style
    namespace so Config can apply its usual precedence rules. The database path
    is always overridden to live inside the temporary project directory.
    """
    args: Any | None = None
    if config_path:
        args = SimpleNamespace(config=config_path, path=str(project_dir))

    config = Config(args=args, target_dir=project_dir)

    db_path = project_dir / ".chunkhound" / "eval.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config.database.path = db_path

    return config


async def _index_corpus(
    project_dir: Path,
    config: Config,
    db_path: Path,
    with_embeddings: bool,
) -> None:
    """Index the corpus using the standard service stack.

    When with_embeddings is True, missing embeddings are generated using the
    configured embedding provider. This is required for semantic evaluation
    (search_mode=semantic).
    """
    services: DatabaseServices | None = None
    try:
        services = create_services(db_path=db_path, config=config)

        indexing_service = DirectoryIndexingService(
            indexing_coordinator=services.indexing_coordinator,
            config=config,
            progress_callback=lambda msg: logger.debug(f"[index] {msg}"),
            progress=None,
        )

        stats = await indexing_service.process_directory(
            project_dir,
            no_embeddings=not with_embeddings,
        )

        if stats.errors_encountered:
            logger.warning(
                f"Indexing completed with {len(stats.errors_encountered)} errors"
            )
    finally:
        # Ensure DuckDB connections are closed so temp dirs can be removed on Windows
        _cleanup_services(services)


async def _run_queries(
    config: Config,
    db_path: Path,
    queries: list[QueryDefinition],
    ks: list[int],
    search_mode: str,
) -> list[QueryMetrics]:
    """Run queries and collect per-query metrics."""
    services: DatabaseServices | None = None
    try:
        services = create_services(db_path=db_path, config=config)
        search_service = services.search_service

        per_query_metrics: list[QueryMetrics] = []
        max_k = max(ks) if ks else 10

        for query in queries:
            start = time.perf_counter()
            if search_mode == "semantic":
                search_text = query.semantic_query
                results, _ = await search_service.search_semantic(
                    query=search_text,
                    page_size=max_k,
                    offset=0,
                )
                search_type = "semantic"
            else:
                search_text = query.pattern
                results, _ = await search_service.search_regex_async(
                    pattern=search_text,
                    page_size=max_k,
                    offset=0,
                    path_filter=None,
                )
                search_type = "regex"

            latency_ms = (time.perf_counter() - start) * 1000.0

            result_paths = [r.get("file_path") for r in results if r.get("file_path")]
            relevant = set(query.relevant_paths)

            first_rank: int | None = None
            for idx, path in enumerate(result_paths, start=1):
                if path in relevant:
                    first_rank = idx
                    break

            metrics_by_k: dict[int, dict[str, float]] = {}

            for k in ks:
                top_k = result_paths[:k]
                hits = len(relevant.intersection(top_k))
                total_relevant = len(relevant)

                recall = float(hits) / float(total_relevant) if total_relevant else 0.0
                if k > 0:
                    precision = (
                        float(hits) / float(min(k, len(result_paths)))
                        if result_paths
                        else 0.0
                    )
                else:
                    precision = 0.0

                metrics_by_k[k] = {
                    "recall": recall,
                    "precision": precision,
                    "hit_count": float(hits),
                }

            per_query_metrics.append(
                QueryMetrics(
                    query_id=query.id,
                    language=query.language,
                    pattern=search_text,
                    search_type=search_type,
                    latency_ms=latency_ms,
                    total_results=len(result_paths),
                    first_relevant_rank=first_rank,
                    metrics_by_k=metrics_by_k,
                )
            )

        return per_query_metrics
    finally:
        # Ensure DuckDB connections are closed so temp dirs can be removed on Windows
        _cleanup_services(services)


async def _run_mode_mixed(
    languages: list[Language],
    ks: list[int],
    search_mode: str,
    config_path: str | None,
    bench_root: Path | None,
) -> EvalResult:
    """Run evaluation with a single mixed-language corpus."""
    if bench_root is not None:
        project_dir = bench_root
        project_dir.mkdir(parents=True, exist_ok=True)

        # Build or refresh corpus in persistent bench directory
        _, queries = create_corpus(project_dir, languages)

        config = _build_config(project_dir, config_path)
        db_path = config.database.path

        logger.info(
            f"Mixed-mode evaluation: {len(languages)} languages, "
            f"{len(queries)} queries, db={db_path}"
        )

        if search_mode == "semantic":
            errors = config.validate_for_command("search")
            if errors:
                raise RuntimeError(
                    "Semantic search requires a configured embedding provider.\n"
                    + "\n".join(errors)
                )

        await _index_corpus(
            project_dir,
            config,
            db_path,
            with_embeddings=(search_mode == "semantic"),
        )
        per_query = await _run_queries(config, db_path, queries, ks, search_mode)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "project"
            project_dir.mkdir(parents=True, exist_ok=True)

            # Build corpus
            _, queries = create_corpus(project_dir, languages)

            # Configure database inside project directory
            config = _build_config(project_dir, config_path)
            db_path = config.database.path

            logger.info(
                f"Mixed-mode evaluation: {len(languages)} languages, "
                f"{len(queries)} queries, db={db_path}"
            )

            if search_mode == "semantic":
                errors = config.validate_for_command("search")
                if errors:
                    raise RuntimeError(
                        "Semantic search requires a configured embedding provider.\n"
                        + "\n".join(errors)
                    )

            await _index_corpus(
                project_dir,
                config,
                db_path,
                with_embeddings=(search_mode == "semantic"),
            )
            per_query = await _run_queries(config, db_path, queries, ks, search_mode)

    # Aggregate metrics after temp directory cleanup
    per_language: dict[str, AggregateMetrics] = {}
    for language in languages:
        lang_queries = [q for q in per_query if q.language == language]
        per_language[language.value] = aggregate_metrics(lang_queries, ks)

    global_metrics = aggregate_metrics(per_query, ks)

    return EvalResult(
        mode="mixed",
        search_mode=search_mode,
        languages=languages,
        ks=ks,
        per_query=per_query,
        per_language=per_language,
        global_metrics=global_metrics,
    )


async def _run_mode_per_language(
    languages: list[Language],
    ks: list[int],
    search_mode: str,
    config_path: str | None,
    bench_root: Path | None,
) -> EvalResult:
    """Run evaluation with one corpus per language."""
    all_queries: list[QueryMetrics] = []
    per_language: dict[str, AggregateMetrics] = {}

    for language in languages:
        if bench_root is not None:
            project_dir = bench_root
            project_dir.mkdir(parents=True, exist_ok=True)

            _, queries = create_corpus(project_dir, [language])

            config = _build_config(project_dir, config_path)
            db_path = config.database.path

            logger.info(
                f"Per-language evaluation (bench): {language.value}, "
                f"{len(queries)} queries, db={db_path}"
            )

            if search_mode == "semantic":
                errors = config.validate_for_command("search")
                if errors:
                    raise RuntimeError(
                        "Semantic search requires a configured embedding provider.\n"
                        + "\n".join(errors)
                    )

            await _index_corpus(
                project_dir,
                config,
                db_path,
                with_embeddings=(search_mode == "semantic"),
            )
            per_query = await _run_queries(config, db_path, queries, ks, search_mode)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                project_dir = Path(tmp) / "project"
                project_dir.mkdir(parents=True, exist_ok=True)

                _, queries = create_corpus(project_dir, [language])

                config = _build_config(project_dir, config_path)
                db_path = config.database.path

                logger.info(
                    f"Per-language evaluation: {language.value}, "
                    f"{len(queries)} queries, db={db_path}"
                )

                if search_mode == "semantic":
                    errors = config.validate_for_command("search")
                    if errors:
                        raise RuntimeError(
                            "Semantic search requires a configured embedding provider.\n"
                            + "\n".join(errors)
                        )

                await _index_corpus(
                    project_dir,
                    config,
                    db_path,
                    with_embeddings=(search_mode == "semantic"),
                )
                per_query = await _run_queries(
                    config, db_path, queries, ks, search_mode
                )

        all_queries.extend(per_query)
        per_language[language.value] = aggregate_metrics(per_query, ks)

    global_metrics = aggregate_metrics(all_queries, ks)

    return EvalResult(
        mode="per-language",
        search_mode=search_mode,
        languages=languages,
        ks=ks,
        per_query=all_queries,
        per_language=per_language,
        global_metrics=global_metrics,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local evaluation harness for ChunkHound search.\n"
            "Builds small synthetic corpora per language, runs regex or semantic "
            "queries, and computes recall@k/precision@k and latency metrics."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["mixed", "per-language"],
        default="mixed",
        help="Evaluation mode: mixed corpus or one corpus per language (default: mixed)",
    )
    parser.add_argument(
        "--search-mode",
        choices=["regex", "semantic"],
        default="regex",
        help="Search type to evaluate (default: regex).",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="all",
        help="Comma-separated list of languages by enum value (or 'all', default).",
    )
    parser.add_argument(
        "--k",
        dest="ks",
        type=int,
        action="append",
        default=None,
        help="Top-k values to evaluate (can be repeated, default: 1,5,10).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a config file for embedding/indexing settings.",
    )
    parser.add_argument(
        "--bench-id",
        type=str,
        default=None,
        help=(
            "Optional benchmark ID. When provided, corpus is stored under "
            ".chunkhound/benches/<bench-id>/source/ by default."
        ),
    )
    parser.add_argument(
        "--bench-root",
        type=str,
        default=None,
        help=(
            "Optional base directory for persistent bench corpora. "
            "If set, corpus is stored under <bench-root>/<bench-id>/source/."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON metrics report.",
    )

    return parser.parse_args(argv)


async def _async_main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        languages = parse_languages_arg(args.languages)
    except ValueError as e:
        logger.error(str(e))
        return 1

    ks: list[int] = sorted(set(args.ks or [1, 5, 10]))
    if not ks:
        logger.error("No k values specified for evaluation.")
        return 1

    search_mode = args.search_mode
    config_path = args.config

    bench_root: Path | None = None
    if args.bench_id:
        if args.bench_root:
            bench_root = Path(args.bench_root) / args.bench_id / "source"
        else:
            bench_root = (
                Path.cwd() / ".chunkhound" / "benches" / args.bench_id / "source"
            )

    try:
        if args.mode == "mixed":
            result = await _run_mode_mixed(
                languages, ks, search_mode, config_path, bench_root
            )
        else:
            result = await _run_mode_per_language(
                languages, ks, search_mode, config_path, bench_root
            )
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"Evaluation failed: {e}")
        return 1

    format_human_summary(result)

    if args.output:
        payload = build_json_payload(result)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON metrics report to {output_path}")

    return 0


def main() -> None:
    """Entry point for python -m chunkhound.tools.eval.search."""
    raise SystemExit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
