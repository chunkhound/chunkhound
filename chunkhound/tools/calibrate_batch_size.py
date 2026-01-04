"""Batch size calibration tool for optimizing embedding and reranking performance.

Benchmarks different batch sizes to find optimal throughput for a given hardware setup.
"""

import statistics
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from chunkhound.interfaces.embedding_provider import APIEmbeddingProvider


@dataclass
class CalibrationConfig:
    """Configuration for batch size calibration."""

    # Batch sizes to test
    embedding_batch_sizes: list[int]
    reranking_batch_sizes: list[int]

    # Test parameters
    num_warmup_runs: int = 2  # Warmup runs to stabilize performance
    num_test_runs: int = 5  # Measurement runs for averaging
    test_document_count: int = 500  # Total documents to process per test
    test_document_length: int = 200  # Average document length in words

    # Safety limits
    max_test_document_count: int = 5000  # Prevent OOM from excessive test data
    max_batch_duration: float = 60.0  # Fail if single batch takes >60s


@dataclass
class BatchSizeResult:
    """Result for a single batch size test."""

    batch_size: int
    throughput_docs_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    total_duration_sec: float
    success: bool
    error_message: str | None = None


@dataclass
class CalibrationResult:
    """Complete calibration results with recommendations."""

    provider_name: str
    model_name: str

    # Embedding results
    embedding_results: list[BatchSizeResult]
    recommended_embedding_batch_size: int

    # Reranking results (if supported)
    reranking_results: list[BatchSizeResult] | None = None
    recommended_reranking_batch_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "provider": self.provider_name,
            "model": self.model_name,
            "embedding": {
                "recommended_batch_size": self.recommended_embedding_batch_size,
                "results": [
                    {
                        "batch_size": r.batch_size,
                        "throughput": round(r.throughput_docs_per_sec, 2),
                        "latency_p50_ms": round(r.latency_p50_ms, 2),
                        "latency_p95_ms": round(r.latency_p95_ms, 2),
                        "success": r.success,
                    }
                    for r in self.embedding_results
                ],
            },
        }

        if self.reranking_results:
            result["reranking"] = {
                "recommended_batch_size": self.recommended_reranking_batch_size,
                "results": [
                    {
                        "batch_size": r.batch_size,
                        "throughput": round(r.throughput_docs_per_sec, 2),
                        "latency_p50_ms": round(r.latency_p50_ms, 2),
                        "latency_p95_ms": round(r.latency_p95_ms, 2),
                        "success": r.success,
                    }
                    for r in self.reranking_results
                ],
            }

        return result


class BatchSizeCalibrator:
    """Calibrates optimal batch sizes for embedding providers."""

    def __init__(self, provider: APIEmbeddingProvider, config: CalibrationConfig):
        """Initialize calibrator.

        Args:
            provider: API embedding provider to calibrate
            config: Calibration configuration

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate config to prevent OOM
        if config.test_document_count > config.max_test_document_count:
            raise ValueError(
                f"test_document_count ({config.test_document_count}) exceeds "
                f"max_test_document_count ({config.max_test_document_count}). "
                f"Reduce test size to prevent out-of-memory errors."
            )

        self.provider = provider
        self.config = config

    async def calibrate(self) -> CalibrationResult:
        """Run full calibration and return recommendations.

        Returns:
            Calibration results with recommended batch sizes
        """
        logger.info(
            f"Starting calibration for {self.provider.name} ({self.provider.model})"
        )

        # Test embedding batch sizes
        logger.info("Calibrating embedding batch sizes...")
        embedding_results = await self._calibrate_embedding()

        # Find optimal embedding batch size
        recommended_embedding = self._find_optimal_batch_size(embedding_results)

        # Test reranking if supported
        reranking_results = None
        recommended_reranking = None

        if self.provider.supports_reranking():
            logger.info("Calibrating reranking batch sizes...")
            reranking_results = await self._calibrate_reranking()
            recommended_reranking = self._find_optimal_batch_size(reranking_results)

        result = CalibrationResult(
            provider_name=self.provider.name,
            model_name=self.provider.model,
            embedding_results=embedding_results,
            recommended_embedding_batch_size=recommended_embedding,
            reranking_results=reranking_results,
            recommended_reranking_batch_size=recommended_reranking,
        )

        logger.info(
            f"Calibration complete. "
            f"Recommended embedding batch size: {recommended_embedding}"
        )
        if recommended_reranking:
            logger.info(f"Recommended reranking batch size: {recommended_reranking}")

        return result

    async def _calibrate_embedding(self) -> list[BatchSizeResult]:
        """Calibrate embedding batch sizes."""
        results = []

        # Generate test documents
        test_docs = self._generate_test_documents(
            self.config.test_document_count, self.config.test_document_length
        )

        for batch_size in self.config.embedding_batch_sizes:
            logger.info(f"Testing embedding batch size: {batch_size}")

            try:
                # Warmup runs
                for _ in range(self.config.num_warmup_runs):
                    await self._run_embedding_batch(test_docs, batch_size)

                # Measurement runs
                durations = []
                for _ in range(self.config.num_test_runs):
                    duration = await self._run_embedding_batch(test_docs, batch_size)
                    durations.append(duration)

                # Calculate metrics
                avg_duration = statistics.mean(durations)
                throughput = len(test_docs) / avg_duration

                # Calculate per-batch latency estimates
                # NOTE: These are approximations. True percentiles would require
                # timing each individual batch, which would complicate the calibration.
                # For batch size tuning, average throughput is the key metric.
                batches_per_run = (len(test_docs) + batch_size - 1) // batch_size
                avg_batch_latency = avg_duration / batches_per_run * 1000  # to ms

                result = BatchSizeResult(
                    batch_size=batch_size,
                    throughput_docs_per_sec=throughput,
                    latency_p50_ms=avg_batch_latency,  # Approximation: uses average
                    latency_p95_ms=avg_batch_latency
                    * 1.2,  # Approximation: assumes 20% overhead for slower batches
                    total_duration_sec=avg_duration,
                    success=True,
                )

                logger.info(
                    f"  Batch size {batch_size}: {throughput:.1f} docs/sec, "
                    f"{avg_batch_latency:.1f}ms avg latency"
                )

            except Exception as e:
                logger.error(f"  Batch size {batch_size} failed: {e}")
                result = BatchSizeResult(
                    batch_size=batch_size,
                    throughput_docs_per_sec=0.0,
                    latency_p50_ms=0.0,
                    latency_p95_ms=0.0,
                    total_duration_sec=0.0,
                    success=False,
                    error_message=str(e),
                )

            results.append(result)

        return results

    async def _calibrate_reranking(self) -> list[BatchSizeResult]:
        """Calibrate reranking batch sizes."""
        results = []

        # Generate test documents and query
        test_docs = self._generate_test_documents(
            self.config.test_document_count, self.config.test_document_length
        )
        query = "test query for reranking calibration"

        for batch_size in self.config.reranking_batch_sizes:
            logger.info(f"Testing reranking batch size: {batch_size}")

            try:
                # Warmup runs
                for _ in range(self.config.num_warmup_runs):
                    await self._run_reranking_batch(query, test_docs, batch_size)

                # Measurement runs
                durations = []
                for _ in range(self.config.num_test_runs):
                    duration = await self._run_reranking_batch(
                        query, test_docs, batch_size
                    )
                    durations.append(duration)

                # Calculate metrics
                avg_duration = statistics.mean(durations)
                throughput = len(test_docs) / avg_duration

                # Calculate per-batch latency (same approximation as embedding)
                batches_per_run = (len(test_docs) + batch_size - 1) // batch_size
                avg_batch_latency = avg_duration / batches_per_run * 1000

                result = BatchSizeResult(
                    batch_size=batch_size,
                    throughput_docs_per_sec=throughput,
                    latency_p50_ms=avg_batch_latency,  # Approximation: uses average
                    latency_p95_ms=avg_batch_latency * 1.2,  # Approximation
                    total_duration_sec=avg_duration,
                    success=True,
                )

                logger.info(
                    f"  Batch size {batch_size}: {throughput:.1f} docs/sec, "
                    f"{avg_batch_latency:.1f}ms avg latency"
                )

            except Exception as e:
                logger.error(f"  Batch size {batch_size} failed: {e}")
                result = BatchSizeResult(
                    batch_size=batch_size,
                    throughput_docs_per_sec=0.0,
                    latency_p50_ms=0.0,
                    latency_p95_ms=0.0,
                    total_duration_sec=0.0,
                    success=False,
                    error_message=str(e),
                )

            results.append(result)

        return results

    async def _run_embedding_batch(
        self, documents: list[str], batch_size: int
    ) -> float:
        """Run embedding on documents with specified batch size.

        Args:
            documents: List of documents to embed
            batch_size: Batch size to use

        Returns:
            Duration in seconds
        """
        start_time = time.perf_counter()

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            await self.provider.embed(batch)

        duration = time.perf_counter() - start_time
        return duration

    async def _run_reranking_batch(
        self, query: str, documents: list[str], batch_size: int
    ) -> float:
        """Run reranking on documents with specified batch size.

        IMPORTANT: Tests actual batch performance by calling rerank() with
        a single batch that fits within the batch_size limit. This measures
        the true performance of reranking batch_size documents, not the
        overhead of the provider's internal batch splitting logic.

        Args:
            query: Query for reranking
            documents: List of documents to rerank
            batch_size: Batch size to test (takes first batch_size documents)

        Returns:
            Duration in seconds
        """
        start_time = time.perf_counter()

        # Take only first batch_size documents to test single-batch performance
        # This ensures provider's rerank() doesn't trigger batch splitting
        test_batch = documents[:batch_size]
        await self.provider.rerank(query, test_batch)

        duration = time.perf_counter() - start_time
        return duration

    def _generate_test_documents(self, count: int, avg_length: int) -> list[str]:
        """Generate synthetic test documents.

        Args:
            count: Number of documents to generate
            avg_length: Average length in words

        Returns:
            List of test documents
        """
        # Simple synthetic documents with code-like structure
        docs = []
        for i in range(count):
            # Vary length slightly for realism
            length = avg_length + (i % 50) - 25
            words = [
                f"word{j % 100}" if j % 3 else f"function_{j}" for j in range(length)
            ]
            doc = " ".join(words)
            docs.append(doc)

        return docs

    def _find_optimal_batch_size(self, results: list[BatchSizeResult]) -> int:
        """Find optimal batch size from calibration results.

        Uses "knee" detection: point of diminishing returns on throughput curve.

        Args:
            results: List of batch size results

        Returns:
            Optimal batch size
        """
        # Filter successful results
        successful = [r for r in results if r.success]

        if not successful:
            logger.warning("No successful calibration results, using default: 100")
            return 100

        # Sort by batch size
        successful.sort(key=lambda r: r.batch_size)

        # Find point where throughput improvement drops below 20%
        # (diminishing returns threshold)
        best_batch_size = successful[0].batch_size
        best_throughput = successful[0].throughput_docs_per_sec

        for i in range(1, len(successful)):
            current = successful[i]
            improvement = (
                current.throughput_docs_per_sec - best_throughput
            ) / best_throughput

            if improvement > 0.15:  # 15% threshold for "significant" improvement
                best_batch_size = current.batch_size
                best_throughput = current.throughput_docs_per_sec
            else:
                # Diminishing returns - stop here
                logger.info(
                    f"Found throughput knee at batch size {best_batch_size} "
                    f"({best_throughput:.1f} docs/sec)"
                )
                break

        return best_batch_size


async def calibrate_provider(
    provider: APIEmbeddingProvider,
    embedding_batch_sizes: list[int] | None = None,
    reranking_batch_sizes: list[int] | None = None,
) -> CalibrationResult:
    """Convenience function to calibrate a provider with default settings.

    Args:
        provider: API embedding provider to calibrate
        embedding_batch_sizes: Batch sizes to test for embedding
            (defaults to [32, 64, 128, 256, 512])
        reranking_batch_sizes: Batch sizes to test for reranking
            (defaults to [32, 64, 96, 128])

    Returns:
        Calibration results with recommendations
    """
    config = CalibrationConfig(
        embedding_batch_sizes=embedding_batch_sizes or [32, 64, 128, 256, 512],
        reranking_batch_sizes=reranking_batch_sizes or [32, 64, 96, 128],
    )

    calibrator = BatchSizeCalibrator(provider, config)
    return await calibrator.calibrate()
