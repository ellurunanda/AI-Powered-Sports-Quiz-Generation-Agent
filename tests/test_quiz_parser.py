"""Unit tests for quiz response parser."""

import pytest

from services import QuizParseError, QuizResponseParser


def test_parse_insufficient_context_short_circuit() -> None:
    parser = QuizResponseParser()
    payload = parser.parse("Insufficient context.")
    assert payload.status == "insufficient_context"
    assert payload.questions == []


def test_parse_valid_json_payload() -> None:
    parser = QuizResponseParser()
    raw = """
{
  "status": "ok",
  "sport": "Football",
  "difficulty": "easy",
  "questions": [
    {
      "question": "Who won the first FIFA World Cup?",
      "options": ["Uruguay", "Brazil", "Argentina", "Germany"],
      "correct_answer": "Uruguay",
      "explanation": "Historical FIFA records show Uruguay won in 1930.",
      "source_context": ["Uruguay won the first FIFA World Cup in 1930."]
    }
  ]
}
"""
    payload = parser.parse(raw)
    assert payload.status == "ok"
    assert payload.sport == "Football"
    assert len(payload.questions) == 1


def test_parse_invalid_json_raises() -> None:
    parser = QuizResponseParser()
    with pytest.raises(QuizParseError, match="not valid JSON"):
        parser.parse("{invalid")

