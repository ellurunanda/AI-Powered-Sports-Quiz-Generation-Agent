"""Unit tests for quiz session-state manager."""

import pytest

from core import QuizSessionManager
from models import QuizPayload


def _payload() -> QuizPayload:
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
                    "explanation": "Uruguay won in 1930.",
                    "source_context": ["Uruguay won the first FIFA World Cup in 1930."],
                }
            ],
        }
    )


def test_session_manager_sets_defaults() -> None:
    state = {}
    manager = QuizSessionManager(state)
    assert state["quiz_payload"] is None
    assert state["selected_answers"] == {}
    assert state["quiz_submitted"] is False
    assert state["quiz_history"] == []
    assert manager is not None


def test_submit_quiz_scores_and_history() -> None:
    state = {}
    manager = QuizSessionManager(state)
    manager.set_current_quiz(_payload())
    manager.set_answer(0, "Uruguay")
    score, total = manager.submit_quiz()
    assert score == 1
    assert total == 1
    assert state["quiz_submitted"] is True
    assert len(state["quiz_history"]) == 1


def test_submit_without_payload_fails() -> None:
    manager = QuizSessionManager({})
    with pytest.raises(ValueError, match="No active quiz payload"):
        manager.submit_quiz()

