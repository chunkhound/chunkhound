"""Service layer for ChunkHound - business logic coordination and dependency injection."""

from .base_service import BaseService
from .daemon_manager import DaemonManager, DaemonStatus, get_daemon_manager
from .embedding_service import EmbeddingService
from .indexing_coordinator import IndexingCoordinator
from .project_registry import ProjectInfo, ProjectRegistry
from .search_service import SearchService
from .watcher_manager import WatcherManager

__all__ = [
    "BaseService",
    "DaemonManager",
    "DaemonStatus",
    "EmbeddingService",
    "IndexingCoordinator",
    "ProjectInfo",
    "ProjectRegistry",
    "SearchService",
    "WatcherManager",
    "get_daemon_manager",
]
