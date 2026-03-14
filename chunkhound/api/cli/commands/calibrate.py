"""Calibrate command module - benchmarks batch sizes for optimal performance."""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.tools.calibrate_batch_size import (
    BatchSizeCalibrator,
    CalibrationConfig,
    CalibrationResult,
)

from ..utils.rich_output import RichOutputFormatter


async def calibrate_command(args: argparse.Namespace, config: Config) -> None:
    """Execute the calibrate command to benchmark batch sizes.

    Args:
        args: Parsed command-line arguments
        config: Pre-validated configuration instance
    """
    # Initialize Rich output formatter
    formatter = RichOutputFormatter(verbose=args.verbose)

    formatter.section_header("ChunkHound Batch Size Calibration")

    # Setup embedding provider
    try:
        if not config.embedding:
            formatter.error("No embedding configuration found")
            formatter.info(
                "Configure an embedding provider via:\n"
                "1. Create .chunkhound.json with embedding configuration, OR\n"
                "2. Set CHUNKHOUND_EMBEDDING__API_KEY and other environment variables\n"
                "3. Use --provider and --model command-line arguments"
            )
            sys.exit(1)

        # Model and provider come from config
        # (via --embedding-model and --embedding-provider args)
        formatter.info(
            f"Testing provider: {config.embedding.provider}, "
            f"model: {config.embedding.model}"
        )

        provider = EmbeddingProviderFactory.create_provider(config.embedding)

    except ValueError as e:
        formatter.error(f"Embedding provider setup failed: {e}")
        sys.exit(1)
    except Exception as e:
        formatter.error(f"Unexpected error setting up embedding provider: {e}")
        logger.exception("Full error details:")
        sys.exit(1)

    # Create calibration configuration
    calibration_config = CalibrationConfig(
        embedding_batch_sizes=args.embedding_batch_sizes,
        reranking_batch_sizes=args.reranking_batch_sizes,
        test_document_count=args.test_document_count,
        num_test_runs=args.num_test_runs,
    )

    formatter.info(
        f"Test parameters: {calibration_config.test_document_count} documents, "
        f"{calibration_config.num_test_runs} runs per batch size"
    )

    # Run calibration
    try:
        calibrator = BatchSizeCalibrator(provider, calibration_config)

        formatter.info("\nStarting calibration (this may take a few minutes)...\n")

        result = await calibrator.calibrate()

        # Display results
        if args.output_format == "json":
            output_data = result.to_dict()
            json_output = json.dumps(output_data, indent=2)

            if args.output_file:
                output_path = Path(args.output_file)
                output_path.write_text(json_output, encoding="utf-8")
                formatter.success(f"Results saved to {output_path}")
            else:
                # Use formatter for consistent output (no styling for JSON)
                formatter.info(json_output)

        else:  # text format
            _display_text_results(formatter, result)

            if args.output_file:
                formatter.warning(
                    "Text format output to file is not recommended. "
                    "Use --output-format json for machine-readable results."
                )

    except Exception as e:
        formatter.error(f"Calibration failed: {e}")
        logger.exception("Full error details:")
        sys.exit(1)
    finally:
        # Cleanup provider
        try:
            await provider.shutdown()
        except Exception as e:
            logger.warning(f"Provider shutdown error: {e}")


def _display_text_results(
    formatter: RichOutputFormatter, result: CalibrationResult
) -> None:
    """Display calibration results in human-readable text format.

    Args:
        formatter: Output formatter for styled text
        result: CalibrationResult object
    """
    formatter.section_header("Calibration Results")

    # Embedding results
    formatter.info("\nEmbedding Batch Size Results:")
    formatter.info("-" * 60)

    for r in result.embedding_results:
        if r.success:
            is_recommended = r.batch_size == result.recommended_embedding_batch_size
            marker = " ⭐ RECOMMENDED" if is_recommended else ""

            formatter.info(
                f"  Batch size {r.batch_size:>4}: "
                f"{r.throughput_docs_per_sec:>7.1f} docs/sec | "
                f"Latency: {r.latency_p50_ms:>6.1f}ms (p50), "
                f"{r.latency_p95_ms:>6.1f}ms (p95){marker}"
            )
        else:
            formatter.error(
                f"  Batch size {r.batch_size:>4}: FAILED - {r.error_message}"
            )

    formatter.success(
        f"\n✓ Recommended embedding batch size: "
        f"{result.recommended_embedding_batch_size}"
    )

    # Reranking results (if available)
    if result.reranking_results:
        formatter.info("\n\nReranking Batch Size Results:")
        formatter.info("-" * 60)

        for r in result.reranking_results:
            if r.success:
                is_recommended = r.batch_size == result.recommended_reranking_batch_size
                marker = " ⭐ RECOMMENDED" if is_recommended else ""

                formatter.info(
                    f"  Batch size {r.batch_size:>4}: "
                    f"{r.throughput_docs_per_sec:>7.1f} docs/sec | "
                    f"Latency: {r.latency_p50_ms:>6.1f}ms (p50), "
                    f"{r.latency_p95_ms:>6.1f}ms (p95){marker}"
                )
            else:
                formatter.error(
                    f"  Batch size {r.batch_size:>4}: FAILED - {r.error_message}"
                )

        formatter.success(
            f"\n✓ Recommended reranking batch size: "
            f"{result.recommended_reranking_batch_size}"
        )

    # Configuration suggestion
    formatter.info("\n\nSuggested Configuration:")
    formatter.info("-" * 60)

    config_snippet = {
        "embedding": {
            "provider": result.provider_name,
            "model": result.model_name,
            "batch_size": result.recommended_embedding_batch_size,
        }
    }

    if result.recommended_reranking_batch_size:
        config_snippet["embedding"]["rerank_batch_size"] = (
            result.recommended_reranking_batch_size
        )

    formatter.info("\nAdd to your .chunkhound.json:")
    formatter.info(json.dumps(config_snippet, indent=2))
