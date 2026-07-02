"""Prompt templates for deep research service.

This module contains all LLM prompts used in the deep research service,
extracted for maintainability and optimization.
"""

from .followup_generation import SYSTEM_MESSAGE as FOLLOWUP_GENERATION_SYSTEM
from .followup_generation import USER_TEMPLATE as FOLLOWUP_GENERATION_USER
from .query_expansion import SYSTEM_MESSAGE as QUERY_EXPANSION_SYSTEM
from .query_expansion import USER_TEMPLATE as QUERY_EXPANSION_USER
from .question_filtering import SYSTEM_MESSAGE as QUESTION_FILTERING_SYSTEM
from .question_filtering import USER_TEMPLATE as QUESTION_FILTERING_USER
from .question_synthesis import SYSTEM_MESSAGE as QUESTION_SYNTHESIS_SYSTEM
from .question_synthesis import USER_TEMPLATE as QUESTION_SYNTHESIS_USER
from .synthesis import CITATION_REQUIREMENTS
from .synthesis import USER_TEMPLATE as SYNTHESIS_USER
from .synthesis import get_system_message as SYNTHESIS_SYSTEM_BUILDER
from .websearch_expansion import SYSTEM_MESSAGE as WEBSEARCH_EXPANSION_SYSTEM
from .websearch_expansion import USER_TEMPLATE as WEBSEARCH_EXPANSION_USER

__all__ = [
    "QUERY_EXPANSION_SYSTEM",
    "QUERY_EXPANSION_USER",
    "FOLLOWUP_GENERATION_SYSTEM",
    "FOLLOWUP_GENERATION_USER",
    "QUESTION_SYNTHESIS_SYSTEM",
    "QUESTION_SYNTHESIS_USER",
    "QUESTION_FILTERING_SYSTEM",
    "QUESTION_FILTERING_USER",
    "SYNTHESIS_SYSTEM_BUILDER",
    "SYNTHESIS_USER",
    "CITATION_REQUIREMENTS",
    "WEBSEARCH_EXPANSION_SYSTEM",
    "WEBSEARCH_EXPANSION_USER",
]
