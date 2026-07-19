"""Parser for Gemini quiz response text."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from models import QuizPayload


class QuizParseError(Exception):
    """Raised when quiz output cannot be parsed into expected schema."""


class QuizResponseParser:
    """Converts raw LLM output text into a typed quiz payload."""

    def parse(self, raw_text: str) -> QuizPayload:
        """Parse one raw LLM response into QuizPayload."""

        if not raw_text.strip():
            raise QuizParseError("Quiz response is empty.")

        normalized = raw_text.strip()
        if normalized == "Insufficient context.":
            return QuizPayload(status="insufficient_context", questions=[])

        json_text = self._extract_json(normalized)
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise QuizParseError("Quiz response is not valid JSON.") from exc

        try:
            return QuizPayload.model_validate(payload)
        except ValidationError as exc:
            raise QuizParseError("Quiz response does not match expected schema.") from exc

    def _extract_json(self, text: str) -> str:
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()
        return text

