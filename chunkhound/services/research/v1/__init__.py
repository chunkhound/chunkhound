"""v1 implementation of deep research using BFS exploration.

This module contains the original BFS-based deep research implementation,
including question generation, quality validation, and synthesis.
"""

from chunkhound.services.research.v1.bfs_research_service import BFSResearchService
from chunkhound.services.research.v1.quality_validator import QualityValidator
from chunkhound.services.research.v1.question_generator import QuestionGenerator
from chunkhound.services.research.v1.synthesis_engine import SynthesisEngine

__all__ = [
    "BFSResearchService",
    "QualityValidator",
    "QuestionGenerator",
    "SynthesisEngine",
]
