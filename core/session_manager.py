"""Session-state management for quiz workflow."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any

from models import QuizPayload


@dataclass
class QuizHistoryItem:
    """Serializable quiz history item stored in session state."""

    sport: str
    difficulty: str
    score: int
    total_questions: int
    payload: dict[str, Any] = field(default_factory=dict)


class QuizSessionManager:
    """Manages quiz session fields over a mutable state mapping."""

    def __init__(self, state: MutableMapping[str, Any]) -> None:
        self.state = state
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        self.state.setdefault("quiz_payload", None)
        self.state.setdefault("selected_answers", {})
        self.state.setdefault("quiz_submitted", False)
        self.state.setdefault("quiz_history", [])

    def set_current_quiz(self, payload: QuizPayload) -> None:
        self.state["quiz_payload"] = payload.model_dump()
        self.state["selected_answers"] = {}
        self.state["quiz_submitted"] = False

    def set_answer(self, question_index: int, selected_option: str) -> None:
        answers = self.state["selected_answers"]
        answers[str(question_index)] = selected_option
        self.state["selected_answers"] = answers

    def submit_quiz(self) -> tuple[int, int]:
        payload_dict = self.state.get("quiz_payload")
        if not payload_dict:
            raise ValueError("No active quiz payload found in session.")

        payload = QuizPayload.model_validate(payload_dict)
        score = 0
        total = len(payload.questions)
        for idx, question in enumerate(payload.questions):
            selected = self.state["selected_answers"].get(str(idx))
            if selected == question.correct_answer:
                score += 1

        self.state["quiz_submitted"] = True
        history_entry = QuizHistoryItem(
            sport=payload.sport,
            difficulty=payload.difficulty,
            score=score,
            total_questions=total,
            payload=payload.model_dump(),
        )
        self.state["quiz_history"].append(history_entry.__dict__)
        return score, total

    def reset_current_quiz(self) -> None:
        self.state["quiz_payload"] = None
        self.state["selected_answers"] = {}
        self.state["quiz_submitted"] = False

