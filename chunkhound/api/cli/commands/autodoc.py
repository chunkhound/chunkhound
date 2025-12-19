"""Autodoc command module - generates scoped architecture/operations docs.

This command uses a two-phase pipeline:
1. Run a shallow deep-research call to identify points of interest for the
   requested scope (overview plan). The count depends on the chosen
   comprehensiveness setting.
2. For each point of interest, run a dedicated deep-research pass and assemble
   the results into a single flowing document, along with a simple coverage
   summary based on referenced files and chunks.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

from chunkhound.api.cli.utils import (
    apply_autodoc_workspace_overrides,
    verify_database_exists,
)
from chunkhound.autodoc import pipeline as autodoc_pipeline
from chunkhound.autodoc.coverage import compute_unreferenced_scope_files
from chunkhound.autodoc.metadata import build_generation_stats_with_coverage
from chunkhound.autodoc.orchestrator import AutodocOrchestrator
from chunkhound.autodoc.render import render_overview_document
from chunkhound.autodoc.service import (
    AutodocNoPointsError,
    run_autodoc_overview_only,
    run_autodoc_pipeline,
)
from chunkhound.autodoc.writer import write_autodoc_outputs
from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.database_factory import create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager

from ..utils.rich_output import RichOutputFormatter
from ..utils.tree_progress import TreeProgressDisplay



async def autodoc_command(args: argparse.Namespace, config: Config) -> None:
    """Execute the autodoc command using deep code research."""
    formatter = RichOutputFormatter(verbose=args.verbose)

    # Resolve workspace root from explicit config file when provided.
    apply_autodoc_workspace_overrides(config=config, args=args)

    # Autodoc always writes artifacts; keep the CLI contract explicit.
    out_dir_arg = getattr(args, "out_dir", None)
    if out_dir_arg is None:
        formatter.error(
            "Autodoc requires --out-dir so it can write an index and per-topic files."
        )
        sys.exit(2)

    llm_manager: LLMManager | None = None

    # Overview-only mode should be lightweight: only HyDE planning + stdout,
    # plus best-effort prompt persistence under --out-dir.
    if getattr(args, "overview_only", False):
        try:
            out_dir = Path(out_dir_arg).resolve()
        except Exception:
            out_dir = None

        try:
            if config.llm:
                utility_config, synthesis_config = config.llm.get_provider_configs()
                llm_manager = LLMManager(utility_config, synthesis_config)
        except Exception:
            llm_manager = None

        orchestrator = AutodocOrchestrator(
            config=config,
            args=args,
            llm_manager=llm_manager,
        )
        scope = orchestrator.resolve_scope()
        run_context = orchestrator.run_context()
        meta_bundle = orchestrator.metadata_bundle(
            scope_path=scope.scope_path,
            target_dir=scope.target_dir,
            overview_only=True,
        )

        try:
            overview_answer, points_of_interest = await run_autodoc_overview_only(
                llm_manager=llm_manager,
                target_dir=scope.target_dir,
                scope_path=scope.scope_path,
                scope_label=scope.scope_label,
                max_points=run_context.max_points,
                comprehensiveness=run_context.comprehensiveness,
                out_dir=out_dir,
                assembly_provider=meta_bundle.assembly_provider,
                indexing_cfg=getattr(config, "indexing", None),
            )
        except AutodocNoPointsError as exc:
            formatter.error(
                "Autodoc could not extract any points of interest from the overview."
            )
            print("\n--- Overview answer ---\n")
            print(exc.overview_answer)
            sys.exit(1)

        meta_bundle.meta.generation_stats["autodoc_comprehensiveness"] = (
            run_context.comprehensiveness
        )
        print(
            render_overview_document(
                meta=meta_bundle.meta,
                scope_label=scope.scope_label,
                overview_answer=overview_answer,
            ).rstrip("\n")
        )
        return

    # Verify database exists and get paths
    try:
        db_path = verify_database_exists(config)
    except (ValueError, FileNotFoundError) as e:
        formatter.error(str(e))
        sys.exit(1)

    # Initialize embedding manager (required for deep research)
    embedding_manager = EmbeddingManager()
    try:
        if config.embedding:
            provider = EmbeddingProviderFactory.create_provider(config.embedding)
            embedding_manager.register_provider(provider, set_default=True)
        else:
            raise ValueError("No embedding provider configured for autodoc")
    except ValueError as e:
        formatter.error(f"Embedding provider setup failed: {e}")
        formatter.info(
            "Configure an embedding provider via:\n"
            "1. Create a .chunkhound.json config file with embeddings, OR\n"
            "2. Set CHUNKHOUND_EMBEDDING__API_KEY environment variable"
        )
        sys.exit(1)
    except Exception as e:
        formatter.error(f"Unexpected error setting up embedding provider: {e}")
        logger.exception("Full error details:")
        sys.exit(1)

    # Initialize LLM manager (required for deep research)
    try:
        if config.llm:
            utility_config, synthesis_config = config.llm.get_provider_configs()
            llm_manager = LLMManager(utility_config, synthesis_config)
        else:
            raise ValueError("No LLM provider configured for autodoc")
    except ValueError as e:
        formatter.error(f"LLM provider setup failed: {e}")
        formatter.info(
            "Configure an LLM provider via:\n"
            "1. Create a .chunkhound.json config file with llm configuration, OR\n"
            "2. Set CHUNKHOUND_LLM_API_KEY environment variable"
        )
        sys.exit(1)
    except Exception as e:
        formatter.error(f"Unexpected error setting up LLM provider: {e}")
        logger.exception("Full error details:")
        sys.exit(1)

    # Create services using unified factory (exactly like MCP/CLI research)
    try:
        services = create_services(
            db_path=db_path,
            config=config,
            embedding_manager=embedding_manager,
        )
    except Exception as e:
        formatter.error(f"Failed to initialize services: {e}")
        sys.exit(1)

    orchestrator = AutodocOrchestrator(
        config=config,
        args=args,
        llm_manager=llm_manager,
    )
    scope = orchestrator.resolve_scope()
    run_context = orchestrator.run_context()
    meta_bundle = orchestrator.metadata_bundle(
        scope_path=scope.scope_path,
        target_dir=scope.target_dir,
        overview_only=False,
    )

    # Phase 1 + 2: run overview (HyDE-based) and per-point deep research with a shared
    # TUI.

    # Optional depth override (currently disabled by default): wiring for
    # experiments where ultra mode may increase BFS depth beyond the global
    # default. For now we keep depth at the standard value unless an explicit
    # environment override is provided so coverage remains focused on breadth.
    original_depth_env = os.getenv("CH_CODE_RESEARCH_MAX_DEPTH")
    depth_overridden = False
    if (
        run_context.comprehensiveness == "ultra"
        and os.getenv("CH_AUTODOC_ULTRA_USE_DEPTH", "0") == "1"
    ):
        os.environ["CH_CODE_RESEARCH_MAX_DEPTH"] = "2"
        depth_overridden = True

    try:
        with TreeProgressDisplay() as tree_progress:
            try:
                pipeline_result = await run_autodoc_pipeline(
                    services=services,
                    embedding_manager=embedding_manager,
                    llm_manager=llm_manager,
                    target_dir=scope.target_dir,
                    scope_path=scope.scope_path,
                    scope_label=scope.scope_label,
                    path_filter=scope.path_filter,
                    comprehensiveness=run_context.comprehensiveness,
                    max_points=run_context.max_points,
                    out_dir=Path(out_dir_arg),
                    assembly_provider=meta_bundle.assembly_provider,
                    indexing_cfg=getattr(config, "indexing", None),
                    progress=tree_progress,
                    log_info=formatter.info,
                    log_warning=formatter.warning,
                    log_error=formatter.error,
                )
            except AutodocNoPointsError as exc:
                formatter.error(
                    "Autodoc could not extract any points of interest from the "
                    "overview."
                )
                print("\n--- Overview answer ---\n")
                print(exc.overview_answer)
                sys.exit(1)
            except Exception as e:
                formatter.error(f"Autodoc research failed: {e}")
                logger.exception("Full error details:")
                sys.exit(1)
    finally:
        # Restore original depth override so other commands/tests are not
        # affected by the ultra-mode setting.
        if depth_overridden:
            if original_depth_env is None:
                os.environ.pop("CH_CODE_RESEARCH_MAX_DEPTH", None)
            else:
                os.environ["CH_CODE_RESEARCH_MAX_DEPTH"] = original_depth_env

    overview_result = pipeline_result.overview_result
    poi_sections = pipeline_result.poi_sections
    unified_source_files = pipeline_result.unified_source_files
    unified_chunks_dedup = pipeline_result.unified_chunks_dedup
    total_files_global = pipeline_result.total_files_global
    total_chunks_global = pipeline_result.total_chunks_global
    scope_total_files = pipeline_result.scope_total_files
    scope_total_chunks = pipeline_result.scope_total_chunks

    total_research_calls = len(poi_sections)
    generation_stats, coverage = build_generation_stats_with_coverage(
        generator_mode="code_research",
        total_research_calls=total_research_calls,
        unified_source_files=unified_source_files,
        unified_chunks_dedup=unified_chunks_dedup,
        scope_label=scope.scope_label,
        scope_total_files=scope_total_files,
        scope_total_chunks=scope_total_chunks,
        total_files_global=total_files_global,
        total_chunks_global=total_chunks_global,
    )
    generation_stats["autodoc_comprehensiveness"] = run_context.comprehensiveness
    meta_bundle.meta.generation_stats = generation_stats

    coverage_lines = autodoc_pipeline._coverage_summary_lines(
        referenced_files=coverage.referenced_files,
        referenced_chunks=coverage.referenced_chunks,
        files_denominator=coverage.files_denominator,
        chunks_denominator=coverage.chunks_denominator,
        scope_total_files=coverage.scope_total_files,
        scope_total_chunks=coverage.scope_total_chunks,
    )

    unreferenced = None
    if not getattr(args, "overview_only", False):
        unreferenced = compute_unreferenced_scope_files(
            services=services,
            scope_label=scope.scope_label,
            referenced_files=unified_source_files,
        )

    out_dir = Path(out_dir_arg).resolve()
    include_combined = os.getenv("CH_AUTODOC_WRITE_COMBINED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )
    write_result = write_autodoc_outputs(
        out_dir=out_dir,
        scope_label=scope.scope_label,
        meta=meta_bundle.meta,
        overview_answer=overview_result.get("answer", "").strip(),
        poi_sections=poi_sections,
        coverage_lines=coverage_lines,
        include_topics=not getattr(args, "overview_only", False),
        include_combined=include_combined,
        unreferenced_files=unreferenced,
    )

    formatter.success("Autodoc complete.")
    if write_result.doc_path is not None:
        formatter.info(f"Wrote combined doc: {write_result.doc_path}")
    if write_result.index_path is not None:
        formatter.info(f"Wrote topics index: {write_result.index_path}")
