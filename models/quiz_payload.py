"""Typed models for quiz generation output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class QuizQuestion(BaseModel):
    """One multiple-choice quiz question."""

    question: str = Field(min_length=5)
    options: list[str] = Field(min_length=4, max_length=4)
    correct_answer: str = Field(min_length=1)
    explanation: str = Field(min_length=5)
    source_context: list[str] = Field(min_length=1)

    @field_validator("options")
    @classmethod
    def normalize_options(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values]

    @field_validator("correct_answer")
    @classmethod
    def normalize_answer(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_answer_in_options(self) -> "QuizQuestion":
        if self.correct_answer not in self.options:
            msg = "correct_answer must be present in options."
            raise ValueError(msg)
        return self


class QuizPayload(BaseModel):
    """Structured quiz output returned by the LLM pipeline."""

    status: Literal["ok", "insufficient_context"]
    sport: str = ""
    difficulty: str = ""
    questions: list[QuizQuestion] = Field(default_factory=list)

