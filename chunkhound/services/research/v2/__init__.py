"""V2 Coverage-First Research implementation.

This module contains the coverage-first deep research implementation,
which uses a 4-phase approach:
1. Phase 1: Full Coverage Retrieval (multi-hop semantic + regex search)
2. Phase 1.5: Depth Exploration (aspect-based queries for top-K files)
3. Phase 2: Gap Detection and Filling (LLM-based gap analysis + parallel fill)
4. Phase 3: Synthesis (compression loop + compound query context)
"""

from chunkhound.services.research.v2.compression_service import CompressionService
from chunkhound.services.research.v2.coverage_research_service import (
    CoverageResearchService,
)
from chunkhound.services.research.v2.coverage_synthesis import CoverageSynthesisEngine
from chunkhound.services.research.v2.depth_exploration import DepthExplorationService
from chunkhound.services.research.v2.gap_detection import GapDetectionService
from chunkhound.services.research.v2.models import (
    GapCandidate,
    UnifiedGap,
)

__all__ = [
    # Main service
    "CoverageResearchService",
    # Phase 1.5 service
    "DepthExplorationService",
    # Phase 2 service
    "GapDetectionService",
    # Phase 3 service
    "CoverageSynthesisEngine",
    # Phase 3 helper
    "CompressionService",
    # Models
    "GapCandidate",
    "UnifiedGap",
]
