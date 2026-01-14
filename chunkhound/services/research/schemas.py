"""Pydantic models for structured LLM outputs in research services.

These models define the expected response schemas for structured completions,
providing type safety and automatic JSON Schema generation for the API.
"""

from pydantic import BaseModel, Field


class QueryExpansionResponse(BaseModel):
    """Response schema for query expansion."""

    queries: list[str] = Field(
        description="Expanded search queries (semantically complete sentences)"
    )

    model_config = {"extra": "forbid"}


class FollowupQuestionsResponse(BaseModel):
    """Response schema for follow-up question generation."""

    questions: list[str] = Field(description="Follow-up research questions")

    model_config = {"extra": "forbid"}


class QuestionSynthesisResponse(BaseModel):
    """Response schema for question synthesis with reasoning."""

    reasoning: str = Field(
        description=(
            "Brief explanation of synthesis strategy and why these questions "
            "explore different unexplored aspects"
        )
    )
    questions: list[str] = Field(
        description="Synthesized research questions, each exploring a distinct aspect"
    )

    model_config = {"extra": "forbid"}
