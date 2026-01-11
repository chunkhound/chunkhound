"""Code Mapper command module - generates scoped architecture/operations docs.

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
from typing import ParamSpec

from loguru import logger

from chunkhound.api.cli.utils import (
    apply_code_mapper_workspace_overrides,
    verify_database_exists,
)
from chunkhound.code_mapper import llm as code_mapper_llm
from chunkhound.code_mapper import pipeline as code_mapper_pipeline
from chunkhound.code_mapper.coverage import compute_unreferenced_scope_files
from chunkhound.code_mapper.metadata import build_generation_stats_with_coverage
from chunkhound.code_mapper.models import CodeMapperPOI
from chunkhound.code_mapper.orchestrator import CodeMapperOrchestrator
from chunkhound.code_mapper.render import render_overview_document
from chunkhound.code_mapper.service import (
    CodeMapperInvalidConcurrencyError,
    CodeMapperNoPointsError,
    run_code_mapper_pipeline,
)
from chunkhound.code_mapper.writer import write_code_mapper_outputs
from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.database_factory import create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.utils.text import safe_scope_label

from ..utils.rich_output import RichOutputFormatter
from ..utils.tree_progress import TreeProgressDisplay

P = ParamSpec("P")


async def run_code_mapper_overview_hyde(
    *args: P.args, **kwargs: P.kwargs
) -> tuple[str, list[CodeMapperPOI]]:
    """Delegate to pipeline helper (wrapper for test monkeypatching)."""
    return await code_mapper_pipeline.run_code_mapper_overview_hyde(*args, **kwargs)


# Re-export for tests that monkeypatch HyDE provider metadata wiring.
build_llm_metadata_and_map_hyde = code_mapper_llm.build_llm_metadata_and_map_hyde


async def code_mapper_command(args: argparse.Namespace, config: Config) -> None:
    """Execute the Code Mapper command using deep code research."""
    formatter = RichOutputFormatter(verbose=args.verbose)

    # Resolve workspace root from explicit config file when provided.
    apply_code_mapper_workspace_overrides(config=config, args=args)

    context_text: str | None = None
    context_arg = getattr(args, "context", None)
    if context_arg is not None:
        try:
            context_path = Path(context_arg).expanduser()
            if not context_path.is_absolute():
                workspace_root = getattr(config, "target_dir", None)
                if isinstance(workspace_root, Path):
                    context_path = workspace_root / context_path
            context_text = context_path.read_text(encoding="utf-8")
        except (OSError, RuntimeError, ValueError) as exc:
            formatter.error(f"Failed to read --context file: {exc}")
            sys.exit(2)

        if not context_text.strip():
            formatter.error("--context file is empty.")
            sys.exit(2)

    # Code Mapper always writes artifacts; keep the CLI contract explicit.
    out_dir_arg = getattr(args, "out", None)
    if out_dir_arg is None:
        formatter.error(
            "Map requires --out so it can write an index and per-topic "
            "files."
        )
        sys.exit(2)

    llm_manager: LLMManager | None = None

    # Overview-only mode should be lightweight: only HyDE planning + stdout,
    # plus best-effort prompt persistence under --out.
    if getattr(args, "overview_only", False):
        try:
            out_dir = Path(out_dir_arg).resolve()
        except (OSError, RuntimeError):
            out_dir = None

        try:
            if config.llm:
                utility_config, synthesis_config = config.llm.get_provider_configs()
                llm_manager = LLMManager(utility_config, synthesis_config)
        except (AttributeError, TypeError, ValueError):
            llm_manager = None

        orchestrator = CodeMapperOrchestrator(
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
            overview_answer, points_of_interest = await run_code_mapper_overview_hyde(
                llm_manager=llm_manager,
                target_dir=scope.target_dir,
                scope_path=scope.scope_path,
                scope_label=scope.scope_label,
                meta=meta_bundle.meta,
                context=context_text,
                max_points=run_context.max_points,
                comprehensiveness=run_context.comprehensiveness,
                out_dir=out_dir,
                persist_prompt=True,
                map_hyde_provider=meta_bundle.map_hyde_provider,
                indexing_cfg=getattr(config, "indexing", None),
            )
        except code_mapper_pipeline.CodeMapperHyDEError as exc:
            formatter.error("Code Mapper HyDE planning failed.")
            print("\n--- HyDE error ---\n")
            print(exc.hyde_message)
            sys.exit(1)
        if not points_of_interest:
            exc = CodeMapperNoPointsError(overview_answer)
            formatter.error(
                "Code Mapper could not extract any points of interest from the "
                "overview."
            )
            print("\n--- Overview answer ---\n")
            print(exc.overview_answer)
            sys.exit(1)

        meta_bundle.meta.generation_stats["code_mapper_comprehensiveness"] = (
            run_context.comprehensiveness
        )
        overview_doc = render_overview_document(
            meta=meta_bundle.meta,
            scope_label=scope.scope_label,
            overview_answer=overview_answer,
        ).rstrip("\n")
        print(overview_doc)
        if out_dir is not None:
            try:
                safe_scope = safe_scope_label(scope.scope_label)
                overview_path = out_dir / f"{safe_scope}_overview.md"
                overview_path.parent.mkdir(parents=True, exist_ok=True)
                overview_path.write_text(overview_doc + "\n", encoding="utf-8")
            except OSError as exc:
                logger.debug(f"Code Mapper: failed to write overview artifact: {exc}")
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
            raise ValueError("No embedding provider configured for Code Mapper")
    except ValueError as e:
        formatter.error(f"Embedding provider setup failed: {e}")
        formatter.info(
            "Configure an embedding provider via:\n"
            "1. Create a .chunkhound.json config file with embeddings, OR\n"
            "2. Set CHUNKHOUND_EMBEDDING__API_KEY environment variable"
        )
        sys.exit(1)
    except (OSError, RuntimeError, TypeError) as e:
        formatter.error(f"Unexpected error setting up embedding provider: {e}")
        logger.exception("Full error details:")
        sys.exit(1)

    # Initialize LLM manager (required for deep research)
    try:
        if config.llm:
            utility_config, synthesis_config = config.llm.get_provider_configs()
            llm_manager = LLMManager(utility_config, synthesis_config)
        else:
            raise ValueError("No LLM provider configured for Code Mapper")
    except ValueError as e:
        formatter.error(f"LLM provider setup failed: {e}")
        formatter.info(
            "Configure an LLM provider via:\n"
            "1. Create a .chunkhound.json config file with llm configuration, OR\n"
            "2. Set CHUNKHOUND_LLM_API_KEY environment variable"
        )
        sys.exit(1)
    except (OSError, RuntimeError, TypeError) as e:
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
    except (OSError, RuntimeError, TypeError, ValueError) as e:
        formatter.error(f"Failed to initialize services: {e}")
        logger.exception("Full error details:")
        sys.exit(1)

    orchestrator = CodeMapperOrchestrator(
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

    with TreeProgressDisplay() as tree_progress:
        try:
            audience = str(getattr(args, "audience", "balanced") or "balanced")
            pipeline_result = await run_code_mapper_pipeline(
                services=services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                target_dir=scope.target_dir,
                scope_path=scope.scope_path,
                scope_label=scope.scope_label,
                path_filter=scope.path_filter,
                meta=meta_bundle.meta,
                context=context_text,
                comprehensiveness=run_context.comprehensiveness,
                max_points=run_context.max_points,
                out_dir=Path(out_dir_arg),
                map_hyde_provider=meta_bundle.map_hyde_provider,
                indexing_cfg=getattr(config, "indexing", None),
                poi_jobs=getattr(args, "jobs", None),
                progress=tree_progress,
                audience=audience,
                log_info=formatter.info,
                log_warning=formatter.warning,
                log_error=formatter.error,
            )
        except CodeMapperNoPointsError as exc:
            formatter.error(
                "Code Mapper could not extract any points of interest from the "
                "overview."
            )
            formatter.text_block(exc.overview_answer, title="Overview answer")
            sys.exit(1)
        except code_mapper_pipeline.CodeMapperHyDEError as exc:
            formatter.error("Code Mapper HyDE planning failed.")
            formatter.text_block(exc.hyde_message, title="HyDE error")
            sys.exit(1)
        except CodeMapperInvalidConcurrencyError as exc:
            formatter.error(str(exc))
            sys.exit(2)
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            formatter.error(f"Code Mapper research failed: {e}")
            logger.exception("Full error details:")
            sys.exit(1)

    overview_result = pipeline_result.overview_result
    poi_sections = pipeline_result.poi_sections
    poi_sections_indexed = pipeline_result.poi_sections_indexed
    failed_poi_sections = pipeline_result.failed_poi_sections
    unified_source_files = pipeline_result.unified_source_files
    unified_chunks_dedup = pipeline_result.unified_chunks_dedup
    total_files_global = pipeline_result.total_files_global
    total_chunks_global = pipeline_result.total_chunks_global
    scope_total_files = pipeline_result.scope_total_files
    scope_total_chunks = pipeline_result.scope_total_chunks

    total_research_calls = pipeline_result.total_points_of_interest
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
    generation_stats["code_mapper_comprehensiveness"] = run_context.comprehensiveness
    audience = str(getattr(args, "audience", "balanced") or "balanced")
    generation_stats["code_mapper_audience"] = audience
    meta_bundle.meta.generation_stats = generation_stats

    coverage_lines = code_mapper_pipeline._coverage_summary_lines(
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
    combined_arg = getattr(args, "combined", None)
    if isinstance(combined_arg, bool):
        include_combined = combined_arg
    else:
        combined_env = os.getenv("CH_CODE_MAPPER_WRITE_COMBINED", "0").strip().lower()
        include_combined = combined_env in ("1", "true", "yes", "y", "on")
    write_result = write_code_mapper_outputs(
        out_dir=out_dir,
        scope_label=scope.scope_label,
        meta=meta_bundle.meta,
        overview_answer=overview_result.get("answer", "").strip(),
        poi_sections=poi_sections,
        poi_sections_indexed=poi_sections_indexed,
        failed_poi_sections=failed_poi_sections,
        coverage_lines=coverage_lines,
        include_topics=not getattr(args, "overview_only", False),
        include_combined=include_combined,
        unreferenced_files=unreferenced,
    )

    if failed_poi_sections:
        formatter.warning(
            f"{len(failed_poi_sections)}/{total_research_calls} topics failed after a "
            "retry; see the topics index for '(failed)' entries. The combined doc "
            "includes only successful topics."
        )

    formatter.success("Code Mapper complete.")
    if write_result.doc_path is not None:
        formatter.info(f"Wrote combined doc: {write_result.doc_path}")
    if write_result.index_path is not None:
        formatter.info(f"Wrote topics index: {write_result.index_path}")
