"""Unit tests for quiz business validation rules."""

import pytest

from core import QuizValidationError, QuizValidator
from models import QuizPayload


def _ok_payload() -> QuizPayload:
    return QuizPayload.model_validate(
        {
            "status": "ok",
            "sport": "Football",
            "difficulty": "easy",
            "questions": [
                {
                    "question": "Who won the first FIFA World Cup?",
                    "options": ["Uruguay", "Brazil", "Argentina", "Germany"],
                    "correct_answer": "Uruguay",
                    "explanation": "Uruguay won the 1930 final.",
                    "source_context": ["Uruguay won the first FIFA World Cup in 1930."],
                }
            ],
        }
    )


def test_validate_ok_payload_success() -> None:
    validator = QuizValidator()
    result = validator.validate(
        payload=_ok_payload(),
        expected_sport="Football",
        expected_difficulty="easy",
        expected_question_count=1,
    )
    assert result.is_valid is True


def test_validate_insufficient_context_success() -> None:
    validator = QuizValidator()
    payload = QuizPayload(status="insufficient_context", questions=[])
    result = validator.validate(
        payload=payload,
        expected_sport="Football",
        expected_difficulty="easy",
        expected_question_count=5,
    )
    assert result.is_valid is True


def test_validate_wrong_question_count_fails() -> None:
    validator = QuizValidator()
    with pytest.raises(QuizValidationError, match="question count"):
        validator.validate(
            payload=_ok_payload(),
            expected_sport="Football",
            expected_difficulty="easy",
            expected_question_count=2,
        )

