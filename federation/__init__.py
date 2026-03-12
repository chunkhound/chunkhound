"""Cross-Repository Federation for ChunkHound.

Queries multiple ChunkHound indexes simultaneously through a single
MCP session.  Searches fan out concurrently, results merge via
cross-repo reranking.
"""

from chunkhound.federation.config import FederationConfig, RepoConfig
from chunkhound.federation.service import FederatedSearchService

__all__: list[str] = [
    "FederationConfig",
    "FederatedSearchService",
    "RepoConfig",
]