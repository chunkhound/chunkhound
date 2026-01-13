"""Performance analyzer for detecting degradation in batch processing."""

import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from scipy.stats import linregress  # type: ignore[import-untyped]

from .batch_metrics import BatchMetricsCollector, BatchTiming


@dataclass
class RegressionResult:
    """Linear regression analysis result for batch latency over time."""

    slope: float
    intercept: float
    r_squared: float
    p_value: float
    stderr: float

    @property
    def degradation_detected(self) -> bool:
        """Detect performance degradation via significant positive trend."""
        return bool(
            self.slope > 0.05 and self.r_squared > 0.80 and self.p_value < 0.05
        )


@dataclass
class OutlierBatch:
    """A batch identified as an outlier based on z-score analysis."""

    batch_index: int
    latency_ms: float
    z_score: float
    chunk_count: int
    reason: str


@dataclass
class PerformanceDiagnostics:
    """Complete performance diagnostics for a batch processing run."""

    total_batches: int
    total_chunks: int
    total_duration_sec: float
    mean_batch_latency_ms: float
    std_batch_latency_ms: float
    throughput_chunks_per_sec: float
    regression: RegressionResult | None
    outliers: list[OutlierBatch]
    batch_metrics: list[dict[str, Any]]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict with version and timestamp."""
        return {
            "version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_batches": self.total_batches,
                "total_chunks": self.total_chunks,
                "total_duration_sec": self.total_duration_sec,
                "mean_batch_latency_ms": self.mean_batch_latency_ms,
                "std_batch_latency_ms": self.std_batch_latency_ms,
                "throughput_chunks_per_sec": self.throughput_chunks_per_sec,
            },
            "regression": (
                {
                    "slope": self.regression.slope,
                    "intercept": self.regression.intercept,
                    "r_squared": self.regression.r_squared,
                    "p_value": self.regression.p_value,
                    "stderr": self.regression.stderr,
                    "degradation_detected": self.regression.degradation_detected,
                }
                if self.regression
                else None
            ),
            "outliers": [
                {
                    "batch_index": o.batch_index,
                    "latency_ms": o.latency_ms,
                    "z_score": o.z_score,
                    "chunk_count": o.chunk_count,
                    "reason": o.reason,
                }
                for o in self.outliers
            ],
            "batch_metrics": self.batch_metrics,
            "warnings": self.warnings,
        }


class PerfAnalyzer:
    """Analyzes batch processing performance to detect degradation and outliers."""

    def __init__(
        self,
        min_batches: int = 5,
        r_squared_threshold: float = 0.80,
        outlier_sigma: float = 2.0,
    ) -> None:
        self.min_batches = min_batches
        self.r_squared_threshold = r_squared_threshold
        self.outlier_sigma = outlier_sigma

    def analyze(self, collector: BatchMetricsCollector) -> PerformanceDiagnostics:
        """Analyze collected batch metrics for performance issues."""
        batches = collector.batches

        if len(batches) < self.min_batches:
            return self._build_summary_only(batches)

        latencies = [b.total_latency_ms for b in batches]
        mean_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

        total_chunks = sum(b.chunk_count for b in batches)
        total_duration = self._calculate_total_duration(batches)
        throughput = total_chunks / total_duration if total_duration > 0 else 0.0

        regression = self._run_regression(batches)
        outliers = self._detect_outliers(batches, mean_latency, std_latency)
        warnings = self._generate_warnings(regression, outliers)
        batch_metrics = self._build_batch_metrics(batches)

        return PerformanceDiagnostics(
            total_batches=len(batches),
            total_chunks=total_chunks,
            total_duration_sec=total_duration,
            mean_batch_latency_ms=mean_latency,
            std_batch_latency_ms=std_latency,
            throughput_chunks_per_sec=throughput,
            regression=regression,
            outliers=outliers,
            batch_metrics=batch_metrics,
            warnings=warnings,
        )

    def _run_regression(self, batches: list[BatchTiming]) -> RegressionResult:
        """Run linear regression on batch latencies over time."""
        x = list(range(len(batches)))
        y = [b.total_latency_ms for b in batches]

        result = linregress(x, y)

        return RegressionResult(
            slope=result.slope,
            intercept=result.intercept,
            r_squared=result.rvalue**2,
            p_value=result.pvalue,
            stderr=result.stderr,
        )

    def _detect_outliers(
        self, batches: list[BatchTiming], mean: float, std: float
    ) -> list[OutlierBatch]:
        """Detect outlier batches using z-score analysis."""
        if std == 0:
            return []

        outliers: list[OutlierBatch] = []
        for batch in batches:
            latency = batch.total_latency_ms
            z_score = (latency - mean) / std

            if abs(z_score) > self.outlier_sigma:
                reason = "high latency" if z_score > 0 else "unusually fast"
                outliers.append(
                    OutlierBatch(
                        batch_index=batch.batch_index,
                        latency_ms=latency,
                        z_score=z_score,
                        chunk_count=batch.chunk_count,
                        reason=reason,
                    )
                )

        return outliers

    def _generate_warnings(
        self, regression: RegressionResult | None, outliers: list[OutlierBatch]
    ) -> list[str]:
        """Generate human-readable warnings based on analysis results."""
        warnings: list[str] = []

        if regression and regression.degradation_detected:
            warnings.append(
                f"Performance degradation detected: latency increasing by "
                f"{regression.slope:.2f}ms per batch (R²={regression.r_squared:.2f}, "
                f"p={regression.p_value:.4f})"
            )

        high_latency_outliers = [o for o in outliers if o.z_score > 0]
        if len(high_latency_outliers) > 0:
            count = len(high_latency_outliers)
            warnings.append(f"{count} batch(es) with unusually high latency")

        return warnings

    def _build_summary_only(
        self, batches: list[BatchTiming]
    ) -> PerformanceDiagnostics:
        """Build diagnostics when insufficient batches for regression."""
        if not batches:
            msg = (
                f"Insufficient data: 0 batches collected "
                f"(minimum {self.min_batches} required for regression)"
            )
            return PerformanceDiagnostics(
                total_batches=0,
                total_chunks=0,
                total_duration_sec=0.0,
                mean_batch_latency_ms=0.0,
                std_batch_latency_ms=0.0,
                throughput_chunks_per_sec=0.0,
                regression=None,
                outliers=[],
                batch_metrics=[],
                warnings=[msg],
            )

        latencies = [b.total_latency_ms for b in batches]
        mean_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

        total_chunks = sum(b.chunk_count for b in batches)
        total_duration = self._calculate_total_duration(batches)
        throughput = total_chunks / total_duration if total_duration > 0 else 0.0

        msg = (
            f"Insufficient data: {len(batches)} batches collected "
            f"(minimum {self.min_batches} required for regression)"
        )
        return PerformanceDiagnostics(
            total_batches=len(batches),
            total_chunks=total_chunks,
            total_duration_sec=total_duration,
            mean_batch_latency_ms=mean_latency,
            std_batch_latency_ms=std_latency,
            throughput_chunks_per_sec=throughput,
            regression=None,
            outliers=[],
            batch_metrics=self._build_batch_metrics(batches),
            warnings=[msg],
        )

    def _calculate_total_duration(self, batches: list[BatchTiming]) -> float:
        """Calculate total duration from first batch start to last batch end."""
        if not batches:
            return 0.0

        start = min(b.start_time for b in batches)
        end = max(b.end_time for b in batches if b.end_time is not None)
        return end - start if end else 0.0

    def _build_batch_metrics(
        self, batches: list[BatchTiming]
    ) -> list[dict[str, Any]]:
        """Build list of per-batch metrics dictionaries."""
        return [
            {
                "batch_index": b.batch_index,
                "chunk_count": b.chunk_count,
                "total_latency_ms": b.total_latency_ms,
                "embed_api_ms": b.embed_api_ms,
                "db_insert_ms": b.db_insert_ms,
            }
            for b in batches
        ]
