"""Business validation rules for parsed quiz payloads."""

from __future__ import annotations

from dataclasses import dataclass

from models import QuizPayload


class QuizValidationError(Exception):
    """Raised when parsed quiz payload violates business rules."""


@dataclass(frozen=True)
class QuizValidationResult:
    """Validation summary for parsed quiz payload."""

    is_valid: bool
    reason: str


class QuizValidator:
    """Applies business-level guardrails on parsed quiz payload."""

    def validate(
        self,
        *,
        payload: QuizPayload,
        expected_sport: str,
        expected_difficulty: str,
        expected_question_count: int,
    ) -> QuizValidationResult:
        """Validate payload consistency and question-level quality."""

        if payload.status == "insufficient_context":
            return QuizValidationResult(is_valid=True, reason="Insufficient context outcome accepted.")

        sport = expected_sport.strip().lower()
        difficulty = expected_difficulty.strip().lower()

        if payload.sport.strip().lower() != sport:
            raise QuizValidationError("Quiz sport does not match requested sport.")
        if payload.difficulty.strip().lower() != difficulty:
            raise QuizValidationError("Quiz difficulty does not match requested difficulty.")
        if len(payload.questions) != expected_question_count:
            raise QuizValidationError("Quiz question count does not match requested count.")

        for index, question in enumerate(payload.questions, start=1):
            if len(question.options) != 4:
                raise QuizValidationError(f"Question {index} does not contain exactly 4 options.")
            if question.correct_answer not in question.options:
                raise QuizValidationError(f"Question {index} has correct answer outside options.")
            if len(question.source_context) < 1:
                raise QuizValidationError(f"Question {index} missing source context.")

        return QuizValidationResult(is_valid=True, reason="Quiz payload is valid.")

