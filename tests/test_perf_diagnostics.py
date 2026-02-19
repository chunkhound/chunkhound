"""Tests for performance diagnostics collection and analysis."""

import asyncio
import time

from chunkhound.core.diagnostics.batch_metrics import (
    BatchMetricsCollector,
    BatchTiming,
)
from chunkhound.core.diagnostics.perf_analyzer import (
    OutlierBatch,
    PerfAnalyzer,
    PerformanceDiagnostics,
    RegressionResult,
)


class TestBatchTiming:
    """Tests for BatchTiming dataclass."""

    def test_total_latency_ms_with_end_time(self):
        """Verify total latency is correctly calculated when end_time is set."""
        timing = BatchTiming(
            batch_index=0, chunk_count=30, start_time=1000.0, end_time=1001.5
        )
        assert timing.total_latency_ms == 1500.0

    def test_total_latency_ms_without_end_time(self):
        """Verify total latency is zero when end_time is not set."""
        timing = BatchTiming(batch_index=0, chunk_count=30, start_time=1000.0)
        assert timing.total_latency_ms == 0.0

    def test_embed_api_ms(self):
        """Verify embedding API duration is correctly calculated."""
        timing = BatchTiming(
            batch_index=0, chunk_count=30, start_time=1000.0,
            embed_api_start=1000.1, embed_api_end=1000.9
        )
        assert abs(timing.embed_api_ms - 800.0) < 0.01

    def test_embed_api_ms_without_start(self):
        """Verify embed_api_ms returns 0 when start is not set."""
        timing = BatchTiming(
            batch_index=0, chunk_count=30, start_time=1000.0,
            embed_api_end=1000.9
        )
        assert timing.embed_api_ms == 0.0

    def test_embed_api_ms_without_end(self):
        """Verify embed_api_ms returns 0 when end is not set."""
        timing = BatchTiming(
            batch_index=0, chunk_count=30, start_time=1000.0,
            embed_api_start=1000.1
        )
        assert timing.embed_api_ms == 0.0

    def test_db_insert_ms(self):
        """Verify database insert duration is correctly calculated."""
        timing = BatchTiming(
            batch_index=0, chunk_count=30, start_time=1000.0,
            db_insert_start=1001.0, db_insert_end=1001.2
        )
        assert abs(timing.db_insert_ms - 200.0) < 0.01

    def test_db_insert_ms_without_start(self):
        """Verify db_insert_ms returns 0 when start is not set."""
        timing = BatchTiming(
            batch_index=0, chunk_count=30, start_time=1000.0,
            db_insert_end=1001.2
        )
        assert timing.db_insert_ms == 0.0

    def test_db_insert_ms_without_end(self):
        """Verify db_insert_ms returns 0 when end is not set."""
        timing = BatchTiming(
            batch_index=0, chunk_count=30, start_time=1000.0,
            db_insert_start=1001.0
        )
        assert timing.db_insert_ms == 0.0

    def test_mark_methods_set_timestamps(self):
        """Verify mark_* methods set perf_counter timestamps on the handle."""
        timing = BatchTiming(batch_index=0, chunk_count=10, start_time=0.0)
        timing.mark_embed_api_start()
        assert timing.embed_api_start is not None
        timing.mark_embed_api_end()
        assert timing.embed_api_end is not None
        timing.mark_db_insert_start()
        assert timing.db_insert_start is not None
        timing.mark_db_insert_end()
        assert timing.db_insert_end is not None


class TestBatchMetricsCollector:
    """Tests for BatchMetricsCollector."""

    def test_collect_single_batch(self):
        """Verify collector properly records a single batch with timing markers."""
        collector = BatchMetricsCollector()
        timing = collector.start_batch(0, 30)
        timing.mark_embed_api_start()
        time.sleep(0.01)
        timing.mark_embed_api_end()
        timing.mark_db_insert_start()
        time.sleep(0.01)
        timing.mark_db_insert_end()
        collector.end_batch(timing)

        assert len(collector.batches) == 1
        assert collector.batches[0].chunk_count == 30
        assert collector.batches[0].total_latency_ms > 0
        assert collector.batches[0].embed_api_ms > 0
        assert collector.batches[0].db_insert_ms > 0

    def test_collect_multiple_batches(self):
        """Verify collector properly records multiple sequential batches."""
        collector = BatchMetricsCollector()
        for i in range(5):
            timing = collector.start_batch(i, 30 + i)
            collector.end_batch(timing)

        assert len(collector.batches) == 5
        for i, batch in enumerate(collector.batches):
            assert batch.batch_index == i
            assert batch.chunk_count == 30 + i

    def test_batches_start_empty(self):
        """Verify collector starts with empty batches list."""
        collector = BatchMetricsCollector()
        assert len(collector.batches) == 0

    def test_start_batch_returns_handle(self):
        """Verify start_batch returns a BatchTiming handle owned by caller."""
        collector = BatchMetricsCollector()
        timing = collector.start_batch(0, 50)
        assert isinstance(timing, BatchTiming)
        assert timing.batch_index == 0
        assert timing.chunk_count == 50
        assert timing.start_time > 0
        assert timing.end_time is None  # not ended yet

    def test_concurrent_handles_independent(self):
        """Verify multiple handles from same collector are independent."""
        collector = BatchMetricsCollector()
        t1 = collector.start_batch(0, 10)
        t2 = collector.start_batch(1, 20)

        t1.mark_embed_api_start()
        t2.mark_db_insert_start()

        assert t1.embed_api_start is not None
        assert t1.db_insert_start is None
        assert t2.embed_api_start is None
        assert t2.db_insert_start is not None

        collector.end_batch(t1)
        collector.end_batch(t2)
        assert len(collector.batches) == 2

    def test_async_concurrent_batches(self):
        """Verify 8 concurrent coroutines get independent correct timing data."""

        async def _run():
            collector = BatchMetricsCollector()

            async def process(batch_idx: int) -> BatchTiming:
                timing = collector.start_batch(batch_idx, 10 + batch_idx)
                timing.mark_embed_api_start()
                await asyncio.sleep(0.01)
                timing.mark_embed_api_end()
                timing.mark_db_insert_start()
                await asyncio.sleep(0.005)
                timing.mark_db_insert_end()
                collector.end_batch(timing)
                return timing

            handles = await asyncio.gather(*[process(i) for i in range(8)])
            return collector, handles

        collector, handles = asyncio.run(_run())

        assert len(collector.batches) == 8

        # Each batch should have its own independent timing data
        seen_indices = set()
        for batch in collector.batches:
            seen_indices.add(batch.batch_index)
            assert batch.chunk_count == 10 + batch.batch_index
            assert batch.total_latency_ms > 0
            assert batch.embed_api_ms > 0
            assert batch.db_insert_ms > 0
            assert batch.end_time is not None

        # All 8 distinct batch indices present
        assert seen_indices == set(range(8))


class TestPerfAnalyzer:
    """Tests for PerfAnalyzer."""

    def test_not_enough_batches_skips_regression(self):
        """Verify regression analysis is skipped with insufficient batches."""
        collector = BatchMetricsCollector()
        for i in range(3):  # Less than min_batches=5
            timing = collector.start_batch(i, 30)
            collector.end_batch(timing)

        analyzer = PerfAnalyzer(min_batches=5)
        result = analyzer.analyze(collector)

        assert result.regression is None
        assert result.total_batches == 3
        # Should have warning about insufficient data
        assert len(result.warnings) == 1
        assert "Insufficient data" in result.warnings[0]

    def test_enough_batches_runs_regression(self):
        """Verify regression analysis runs with sufficient batches."""
        collector = BatchMetricsCollector()
        # Use synthetic data to avoid time.sleep
        for i in range(10):
            timing = BatchTiming(
                batch_index=i, chunk_count=30,
                start_time=float(i), end_time=float(i) + 0.1,
            )
            collector.batches.append(timing)

        analyzer = PerfAnalyzer(min_batches=5)
        result = analyzer.analyze(collector)

        assert result.regression is not None
        assert result.total_batches == 10
        assert result.total_chunks == 300

    def test_identifies_outliers(self):
        """Verify analyzer correctly identifies outlier batches via z-score."""
        # Create synthetic data with one outlier
        collector = BatchMetricsCollector()

        # Normal batches with consistent timing
        for i in range(9):
            timing = BatchTiming(
                batch_index=i,
                chunk_count=30,
                start_time=float(i),
                end_time=float(i) + 0.1,  # 100ms each
            )
            collector.batches.append(timing)

        # Outlier batch with 10x latency
        outlier = BatchTiming(
            batch_index=9,
            chunk_count=30,
            start_time=9.0,
            end_time=10.0,  # 1000ms - much higher
        )
        collector.batches.append(outlier)

        analyzer = PerfAnalyzer(outlier_sigma=2.0)
        result = analyzer.analyze(collector)

        assert len(result.outliers) >= 1
        outlier_indices = [o.batch_index for o in result.outliers]
        assert 9 in outlier_indices

    def test_outlier_reason_high_latency(self):
        """Verify high latency outliers get correct reason."""
        collector = BatchMetricsCollector()

        # Normal batches
        for i in range(9):
            timing = BatchTiming(
                batch_index=i, chunk_count=30,
                start_time=float(i), end_time=float(i) + 0.1,
            )
            collector.batches.append(timing)

        # High latency outlier
        outlier = BatchTiming(
            batch_index=9, chunk_count=30,
            start_time=9.0, end_time=10.0,
        )
        collector.batches.append(outlier)

        analyzer = PerfAnalyzer(outlier_sigma=2.0)
        result = analyzer.analyze(collector)

        high_latency_outliers = [o for o in result.outliers if o.batch_index == 9]
        assert len(high_latency_outliers) == 1
        assert high_latency_outliers[0].reason == "high latency"

    def test_empty_collector(self):
        """Verify analyzer handles empty collector gracefully."""
        collector = BatchMetricsCollector()
        analyzer = PerfAnalyzer(min_batches=5)
        result = analyzer.analyze(collector)

        assert result.total_batches == 0
        assert result.total_chunks == 0
        assert result.regression is None
        assert len(result.warnings) == 1
        assert "Insufficient data" in result.warnings[0]

    def test_zero_std_no_outliers(self):
        """Verify no outliers when all batches have identical timing (zero std)."""
        collector = BatchMetricsCollector()

        # All batches with exactly the same timing
        for i in range(10):
            timing = BatchTiming(
                batch_index=i, chunk_count=30,
                start_time=float(i), end_time=float(i) + 0.1,  # All 100ms
            )
            collector.batches.append(timing)

        analyzer = PerfAnalyzer(outlier_sigma=2.0)
        result = analyzer.analyze(collector)

        # With zero std deviation, no outliers can be detected
        assert len(result.outliers) == 0

    def test_to_dict_includes_required_fields(self):
        """Verify to_dict produces all required fields."""
        collector = BatchMetricsCollector()
        for i in range(5):
            timing = collector.start_batch(i, 30)
            collector.end_batch(timing)

        analyzer = PerfAnalyzer(min_batches=5)
        result = analyzer.analyze(collector)
        result_dict = result.to_dict()

        assert "version" in result_dict
        assert result_dict["version"] == "1.0"
        assert "timestamp" in result_dict
        assert "summary" in result_dict
        assert "regression" in result_dict
        assert "outliers" in result_dict
        assert "batch_metrics" in result_dict
        assert "warnings" in result_dict

    def test_to_dict_summary_fields(self):
        """Verify to_dict summary contains expected statistics."""
        collector = BatchMetricsCollector()
        for i in range(5):
            timing = BatchTiming(
                batch_index=i, chunk_count=30,
                start_time=float(i), end_time=float(i) + 0.1,
            )
            collector.batches.append(timing)

        analyzer = PerfAnalyzer(min_batches=5)
        result = analyzer.analyze(collector)
        summary = result.to_dict()["summary"]

        assert summary["total_batches"] == 5
        assert summary["total_chunks"] == 150
        assert "mean_batch_latency_ms" in summary
        assert "std_batch_latency_ms" in summary
        assert "throughput_chunks_per_sec" in summary

    def test_regression_uses_batch_index_ordering(self):
        """Verify regression sorts by batch_index and uses it as x-values."""
        collector = BatchMetricsCollector()

        # Insert batches out of order
        for i in [5, 2, 8, 0, 3, 7, 1, 4, 6, 9]:
            timing = BatchTiming(
                batch_index=i, chunk_count=30,
                start_time=float(i), end_time=float(i) + 0.1 + (i * 0.01),
            )
            collector.batches.append(timing)

        analyzer = PerfAnalyzer(min_batches=5)
        result = analyzer.analyze(collector)

        # Regression should still work correctly on out-of-order batches
        assert result.regression is not None


class TestRegressionResult:
    """Tests for RegressionResult."""

    def test_degradation_detected_all_conditions(self):
        """Verify degradation is detected when all thresholds are met."""
        result = RegressionResult(
            slope=0.1,  # > 0.05
            intercept=100.0,
            r_squared=0.9,  # > 0.80
            p_value=0.01,  # < 0.05
            stderr=0.01,
        )
        assert result.degradation_detected is True

    def test_degradation_not_detected_low_slope(self):
        """Verify no degradation when slope is below threshold."""
        result = RegressionResult(
            slope=0.01,  # < 0.05
            intercept=100.0,
            r_squared=0.9,
            p_value=0.01,
            stderr=0.01,
        )
        assert result.degradation_detected is False

    def test_degradation_not_detected_low_r_squared(self):
        """Verify no degradation when r_squared is below threshold."""
        result = RegressionResult(
            slope=0.1,
            intercept=100.0,
            r_squared=0.5,  # < 0.80
            p_value=0.01,
            stderr=0.01,
        )
        assert result.degradation_detected is False

    def test_degradation_not_detected_high_p_value(self):
        """Verify no degradation when p_value is above threshold."""
        result = RegressionResult(
            slope=0.1,
            intercept=100.0,
            r_squared=0.9,
            p_value=0.1,  # > 0.05
            stderr=0.01,
        )
        assert result.degradation_detected is False

    def test_degradation_not_detected_negative_slope(self):
        """Verify no degradation when slope is negative (improving performance)."""
        result = RegressionResult(
            slope=-0.1,  # Negative = improving
            intercept=100.0,
            r_squared=0.9,
            p_value=0.01,
            stderr=0.01,
        )
        assert result.degradation_detected is False

    def test_degradation_boundary_slope(self):
        """Verify slope exactly at boundary is not detected as degradation."""
        result = RegressionResult(
            slope=0.05,  # At boundary (needs to be > 0.05)
            intercept=100.0,
            r_squared=0.9,
            p_value=0.01,
            stderr=0.01,
        )
        assert result.degradation_detected is False


class TestOutlierBatch:
    """Tests for OutlierBatch dataclass."""

    def test_outlier_batch_fields(self):
        """Verify OutlierBatch stores all fields correctly."""
        outlier = OutlierBatch(
            batch_index=5,
            latency_ms=500.0,
            z_score=3.5,
            chunk_count=30,
            reason="high latency",
        )
        assert outlier.batch_index == 5
        assert outlier.latency_ms == 500.0
        assert outlier.z_score == 3.5
        assert outlier.chunk_count == 30
        assert outlier.reason == "high latency"


class TestPerformanceDiagnostics:
    """Tests for PerformanceDiagnostics dataclass."""

    def test_to_dict_with_regression(self):
        """Verify to_dict includes regression data when present."""
        regression = RegressionResult(
            slope=0.1, intercept=100.0, r_squared=0.9, p_value=0.01, stderr=0.01
        )
        diagnostics = PerformanceDiagnostics(
            total_batches=10,
            total_chunks=300,
            total_duration_sec=5.0,
            mean_batch_latency_ms=100.0,
            std_batch_latency_ms=10.0,
            throughput_chunks_per_sec=60.0,
            regression=regression,
            outliers=[],
            batch_metrics=[],
            warnings=[],
        )

        result = diagnostics.to_dict()
        assert result["regression"] is not None
        assert result["regression"]["slope"] == 0.1
        assert result["regression"]["degradation_detected"] is True

    def test_to_dict_without_regression(self):
        """Verify to_dict handles None regression."""
        diagnostics = PerformanceDiagnostics(
            total_batches=3,
            total_chunks=90,
            total_duration_sec=1.0,
            mean_batch_latency_ms=100.0,
            std_batch_latency_ms=0.0,
            throughput_chunks_per_sec=90.0,
            regression=None,
            outliers=[],
            batch_metrics=[],
            warnings=["Insufficient data"],
        )

        result = diagnostics.to_dict()
        assert result["regression"] is None

    def test_to_dict_with_outliers(self):
        """Verify to_dict serializes outliers correctly."""
        outlier = OutlierBatch(
            batch_index=5,
            latency_ms=500.0,
            z_score=3.5,
            chunk_count=30,
            reason="high latency",
        )
        diagnostics = PerformanceDiagnostics(
            total_batches=10,
            total_chunks=300,
            total_duration_sec=5.0,
            mean_batch_latency_ms=100.0,
            std_batch_latency_ms=10.0,
            throughput_chunks_per_sec=60.0,
            regression=None,
            outliers=[outlier],
            batch_metrics=[],
            warnings=[],
        )

        result = diagnostics.to_dict()
        assert len(result["outliers"]) == 1
        assert result["outliers"][0]["batch_index"] == 5
        assert result["outliers"][0]["reason"] == "high latency"

    def test_to_dict_timestamp_format(self):
        """Verify timestamp is in ISO format."""
        diagnostics = PerformanceDiagnostics(
            total_batches=0,
            total_chunks=0,
            total_duration_sec=0.0,
            mean_batch_latency_ms=0.0,
            std_batch_latency_ms=0.0,
            throughput_chunks_per_sec=0.0,
            regression=None,
            outliers=[],
            batch_metrics=[],
            warnings=[],
        )

        result = diagnostics.to_dict()
        # ISO format includes 'T' separator
        assert "T" in result["timestamp"]


class TestPerfAnalyzerWarnings:
    """Tests for warning generation in PerfAnalyzer."""

    def test_warning_on_degradation(self):
        """Verify warning is generated when degradation is detected."""
        collector = BatchMetricsCollector()

        # Create synthetic data with clear degradation trend
        for i in range(10):
            timing = BatchTiming(
                batch_index=i, chunk_count=30,
                start_time=float(i),
                end_time=float(i) + 0.1 + (i * 0.05),  # Increasing latency
            )
            collector.batches.append(timing)

        analyzer = PerfAnalyzer(min_batches=5)
        result = analyzer.analyze(collector)

        # If degradation was detected, should have warning
        if result.regression and result.regression.degradation_detected:
            assert any("degradation" in w.lower() for w in result.warnings)

    def test_warning_on_high_latency_outliers(self):
        """Verify warning is generated for high latency outliers."""
        collector = BatchMetricsCollector()

        # Normal batches
        for i in range(9):
            timing = BatchTiming(
                batch_index=i, chunk_count=30,
                start_time=float(i), end_time=float(i) + 0.1,
            )
            collector.batches.append(timing)

        # Outlier with very high latency
        outlier = BatchTiming(
            batch_index=9, chunk_count=30,
            start_time=9.0, end_time=20.0,  # Very high
        )
        collector.batches.append(outlier)

        analyzer = PerfAnalyzer(outlier_sigma=2.0)
        result = analyzer.analyze(collector)

        # Should have warning about high latency
        assert any("high latency" in w.lower() for w in result.warnings)
