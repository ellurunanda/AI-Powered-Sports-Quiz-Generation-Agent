"""Unit tests for grounded prompt construction."""

from models import NewsResult, RetrievalChunk
from prompts.quiz_prompt_builder import build_grounded_quiz_prompt


def test_build_grounded_quiz_prompt_contains_strict_rules() -> None:
    prompt = build_grounded_quiz_prompt(
        sport="Football",
        difficulty="easy",
        question_count=5,
        historical_chunks=[
            RetrievalChunk(
                id="f1",
                sport="Football",
                difficulty="easy",
                source="FIFA",
                text="Uruguay won the first FIFA World Cup in 1930.",
                score=0.91,
                metadata={},
            )
        ],
        news_results=[
            NewsResult(
                title="Latest Football Updates",
                snippet="Key transfer stories from top leagues.",
                url="https://example.com/news",
            )
        ],
    )
    assert "Use ONLY the provided context" in prompt
    assert "Insufficient context." in prompt
    assert "Football" in prompt
    assert "latest news" in prompt.lower()

