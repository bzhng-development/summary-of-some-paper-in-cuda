"""Pydantic schemas for the paper summarization pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PitchOutput(BaseModel):
    title: str = Field(
        description="The exact title of the paper as it appears in the PDF",
    )
    pitch: str = Field(
        description=(
            "A compelling 2-3 sentence pitch that captures the paper's core "
            "contribution and why it matters"
        ),
    )


class CategoryOutput(BaseModel):
    category: str = Field(
        description="Best matching category from the available list",
    )
