"""Backwards compatibility shim for QuestionGenerator.

This module re-exports QuestionGenerator from v1 to maintain backwards compatibility
with code that imports from chunkhound.services.research.question_generator.

New code should import directly from chunkhound.services.research.v1.question_generator.
"""

from chunkhound.services.research.v1.question_generator import QuestionGenerator

__all__ = ["QuestionGenerator"]
