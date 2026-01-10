"""Shard state metrics for derived-state architecture.

Provides ShardState dataclass for inspecting shard health and the get_shard_state
function to derive metrics from DuckDB and USearch index files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from chunkhound.providers.database import usearch_wrapper


@dataclass
class ShardState:
    """Derived shard state - always correct, never stale.

    Metrics are computed dynamically from DuckDB (source of truth) and
    USearch index files (derived state). No stored counters.

    Attributes:
        index_live: Live vectors in USearch index (excludes tombstones)
        index_nodes: Total nodes in USearch including tombstones
        db_count: DuckDB COUNT(*) for this shard - should equal index_live
        self_recall: Quality measurement (0.0-1.0), 1.0 if not measured
    """

    index_live: int
    index_nodes: int
    db_count: int
    self_recall: float

    @property
    def tombstone_count(self) -> int:
        """Tombstones tracked by USearch, not DuckDB."""
        return self.index_nodes - self.index_live

    @property
    def tombstone_ratio(self) -> float:
        """Ratio of tombstones to total nodes in USearch index."""
        return self.tombstone_count / self.index_nodes if self.index_nodes > 0 else 0.0


def get_shard_state(
    shard_id: UUID,
    db_connection: Any,
    file_path: str | Path,
    dims: int,
    measure_quality: bool = False,
) -> ShardState:
    """Query current shard state from DuckDB and USearch index file.

    Derives metrics dynamically rather than storing counters, ensuring
    consistency and eliminating stale state.

    Args:
        shard_id: The shard UUID to inspect
        db_connection: Active DuckDB connection for COUNT query
        file_path: Path to the .usearch index file
        dims: Embedding dimensions (determines table name)
        measure_quality: If True, run self_recall benchmark (~100-500ms)

    Returns:
        ShardState with current metrics

    Raises:
        FileNotFoundError: If index file does not exist
        RuntimeError: If index cannot be loaded
    """
    file_path = Path(file_path)

    # Open USearch index as memory-mapped view
    index = usearch_wrapper.open_view(file_path)

    # USearch metrics
    index_live = len(index)
    index_nodes = index.stats.nodes if hasattr(index.stats, "nodes") else index_live

    # DuckDB COUNT(*) - embeddings assigned to this shard
    table_name = f"embeddings_{dims}"
    result = db_connection.execute(
        f"SELECT COUNT(*) FROM {table_name} WHERE shard_id = ?",
        [str(shard_id)],
    ).fetchone()
    db_count = result[0] if result else 0

    # Quality measurement (expensive, optional)
    quality = 1.0
    if measure_quality and index_live > 0:
        sample_size = min(1000, index_live)
        quality = usearch_wrapper.measure_quality(file_path, sample_size=sample_size)

    return ShardState(
        index_live=index_live,
        index_nodes=index_nodes,
        db_count=db_count,
        self_recall=quality,
    )
