"""Domain model for historical sports facts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


DifficultyLevel = Literal["easy", "medium", "hard"]


class SportsFact(BaseModel):
    """Typed representation of one historical sports fact record."""

    id: str = Field(min_length=3)
    sport: str = Field(min_length=2)
    difficulty: DifficultyLevel
    title: str = Field(min_length=3)
    fact: str = Field(min_length=20)
    year: int = Field(ge=1800, le=2100)
    era: str = Field(min_length=3)
    source: str = Field(min_length=3)
    tags: list[str] = Field(min_length=1)

    @field_validator("sport")
    @classmethod
    def normalize_sport(cls, value: str) -> str:
        return value.strip()

    @field_validator("difficulty")
    @classmethod
    def normalize_difficulty(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        cleaned_tags = [tag.strip().lower() for tag in value if tag.strip()]
        if not cleaned_tags:
            msg = "tags must include at least one non-empty value"
            raise ValueError(msg)
        return cleaned_tags

