"""Schemas for the deep research workflow."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClarifyWithUser(BaseModel):
    """Decision on whether the research task needs clarification."""

    need_clarification: bool = Field(
        description="Whether a clarification question is needed before research starts."
    )
    question: str = Field(description="A single concise clarification question.")
    verification: str = Field(description="A short acknowledgement when research can start.")


class ResearchQuestion(BaseModel):
    """Structured research brief."""

    research_brief: str = Field(description="Detailed brief used to guide deep research.")


class ResearchPlan(BaseModel):
    """Parallel research topics."""

    topics: list[str] = Field(
        min_length=1,
        max_length=4,
        description="Specific research topics that can be investigated in parallel.",
    )


class SearchQueries(BaseModel):
    """Search queries for a topic."""

    queries: list[str] = Field(
        min_length=1,
        max_length=3,
        description="Search queries for the research topic.",
    )


class SourceResult(BaseModel):
    """Normalized result from any research source."""

    source: str
    title: str
    url: str
    content: str
