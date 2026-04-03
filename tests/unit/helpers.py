"""Shared test stubs for unit tests."""


class _Cfg:
    """Minimal Config stub for IndexingCoordinator tests."""

    class _Indexing:
        cleanup = False
        force_reindex = False
        per_file_timeout_seconds = 0.0
        min_dirs_for_parallel = 4
        max_discovery_workers = 4
        parallel_discovery = False

    indexing = _Indexing()
