"""v1 implementation of deep research with pluggable exploration.

This module contains the original research implementation with pluggable
exploration strategies, question generation, quality validation, and synthesis.
"""

from chunkhound.services.research.v1.pluggable_research_service import (
    PluggableResearchService,
)

# Backward compatibility alias
BFSResearchService = PluggableResearchService

from chunkhound.services.research.v1.quality_validator import QualityValidator
from chunkhound.services.research.v1.question_generator import QuestionGenerator
from chunkhound.services.research.v1.synthesis_engine import SynthesisEngine

__all__ = [
    "PluggableResearchService",
    "BFSResearchService",  # Backward compatibility alias
    "QualityValidator",
    "QuestionGenerator",
    "SynthesisEngine",
]
