"""Prompt builder for grounded sports quiz generation."""

from __future__ import annotations

from models import NewsResult, RetrievalChunk


def build_grounded_quiz_prompt(
    *,
    sport: str,
    difficulty: str,
    question_count: int,
    historical_chunks: list[RetrievalChunk],
    news_results: list[NewsResult],
) -> str:
    """Build a strict anti-hallucination prompt for Gemini quiz generation."""

    historical_context = "\n".join(
        f"- [Historical] ({chunk.source}) score={chunk.score:.4f}: {chunk.text}"
        for chunk in historical_chunks
    )
    news_context = "\n".join(
        f"- [News] ({item.url}) {item.title}: {item.snippet}"
        for item in news_results
    )

    return f"""
You are a factual sports quiz generator.

STRICT RULES:
1) Use ONLY the provided context.
2) Do NOT invent facts, names, dates, or statistics.
3) If context is insufficient to produce a reliable quiz, respond exactly with:
Insufficient context.
4) Each question must be objectively verifiable from context.
5) Each question must have exactly 4 options.
6) Return valid JSON only (no markdown fences).

OUTPUT JSON SCHEMA:
{{
  "status": "ok" | "insufficient_context",
  "sport": "string",
  "difficulty": "easy|medium|hard",
  "questions": [
    {{
      "question": "string",
      "options": ["string", "string", "string", "string"],
      "correct_answer": "string",
      "explanation": "string",
      "source_context": ["string", "string"]
    }}
  ]
}}

REQUEST:
- Sport: {sport}
- Difficulty: {difficulty}
- Number of questions: {question_count}

HISTORICAL CONTEXT:
{historical_context if historical_context else "- none"}

LATEST NEWS CONTEXT:
{news_context if news_context else "- none"}

Remember: if information is not enough, return exactly "Insufficient context.".
""".strip()

