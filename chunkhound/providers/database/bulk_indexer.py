"""Bulk indexer for deferred quality checks during batch operations."""

from types import TracebackType
from typing import TYPE_CHECKING, Literal

from chunkhound.core.config.sharding_config import ShardingConfig

if TYPE_CHECKING:
    from chunkhound.providers.database.duckdb_provider import DuckDBProvider


class BulkIndexer:
    """Tracks batch state for deferred quality checks during bulk indexing.

    Routes fix_pass calls through the provider's executor to ensure
    database operations run in the correct thread with connection access.

    Usage:
        with provider.get_bulk_indexer() as bulk:
            for batch in batches:
                provider.insert_embeddings_batch(batch)
                bulk.on_batch_completed()
        # Final quality check runs on context exit
    """

    def __init__(self, provider: "DuckDBProvider", config: ShardingConfig) -> None:
        self._provider = provider
        self._config = config
        self._batches_since_quality_check = 0
        self._needs_final_quality_check = False

    def __enter__(self) -> "BulkIndexer":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        if self._needs_final_quality_check:
            self._provider.run_fix_pass(check_quality=True)
        return False

    def on_batch_completed(self) -> None:
        """Notify that a batch was completed. May trigger deferred fix_pass."""
        self._batches_since_quality_check += 1
        self._needs_final_quality_check = True

        # Check if we should run fix_pass based on interval
        if self._batches_since_quality_check >= self._config.quality_check_interval:
            should_check_quality = self._config.quality_check_mode == "immediate"
            self._provider.run_fix_pass(check_quality=should_check_quality)
            self._batches_since_quality_check = 0
            if should_check_quality:
                self._needs_final_quality_check = False
