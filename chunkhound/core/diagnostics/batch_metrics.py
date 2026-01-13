import time
from dataclasses import dataclass, field


@dataclass
class BatchTiming:
    """Timing data for a single embedding batch."""

    batch_index: int
    chunk_count: int
    start_time: float
    end_time: float | None = None
    embed_api_start: float | None = None
    embed_api_end: float | None = None
    db_insert_start: float | None = None
    db_insert_end: float | None = None

    @property
    def total_latency_ms(self) -> float:
        """Total batch processing time in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    @property
    def embed_api_ms(self) -> float:
        """Embedding API call duration in milliseconds."""
        if self.embed_api_start is None or self.embed_api_end is None:
            return 0.0
        return (self.embed_api_end - self.embed_api_start) * 1000

    @property
    def db_insert_ms(self) -> float:
        """Database insert duration in milliseconds."""
        if self.db_insert_start is None or self.db_insert_end is None:
            return 0.0
        return (self.db_insert_end - self.db_insert_start) * 1000


@dataclass
class BatchMetricsCollector:
    """Collects timing metrics across all batches during embedding generation."""

    batches: list[BatchTiming] = field(default_factory=list)
    _current_batch: BatchTiming | None = field(default=None, repr=False)

    def start_batch(self, batch_index: int, chunk_count: int) -> None:
        """Start timing a new batch."""
        self._current_batch = BatchTiming(
            batch_index=batch_index,
            chunk_count=chunk_count,
            start_time=time.perf_counter(),
        )

    def mark_embed_api_start(self) -> None:
        """Mark the start of embedding API call."""
        if self._current_batch:
            self._current_batch.embed_api_start = time.perf_counter()

    def mark_embed_api_end(self) -> None:
        """Mark the end of embedding API call."""
        if self._current_batch:
            self._current_batch.embed_api_end = time.perf_counter()

    def mark_db_insert_start(self) -> None:
        """Mark the start of database insert."""
        if self._current_batch:
            self._current_batch.db_insert_start = time.perf_counter()

    def mark_db_insert_end(self) -> None:
        """Mark the end of database insert."""
        if self._current_batch:
            self._current_batch.db_insert_end = time.perf_counter()

    def end_batch(self) -> None:
        """End timing the current batch and store it."""
        if self._current_batch:
            self._current_batch.end_time = time.perf_counter()
            self.batches.append(self._current_batch)
            self._current_batch = None
