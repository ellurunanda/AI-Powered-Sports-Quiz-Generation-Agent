"""Unit tests for RAG quiz generation pipeline."""

from __future__ import annotations

from core.rag_quiz_pipeline import RAGQuizPipeline
from models import NewsResult, RetrievalChunk


class _FakeEmbeddingGenerator:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2]]


class _FakeVectorStore:
    def query_facts(self, *, query_embedding, top_k, sport=None, difficulty=None):
        return [
            RetrievalChunk(
                id="h1",
                sport="Football",
                difficulty="easy",
                source="FIFA",
                text="Uruguay won the first FIFA World Cup in 1930.",
                score=0.95,
                metadata={},
            )
        ]


class _FakeNewsSearch:
    def search_latest_news(self, *, sport: str, limit: int | None = None):
        return [
            NewsResult(
                title="Football update",
                snippet="League title race intensifies.",
                url="https://example.com/f1",
            )
        ]


class _FakeGemini:
    def generate_quiz_text(self, prompt: str) -> str:
        return '{"status":"ok","questions":[]}'


def test_rag_quiz_pipeline_generate_quiz_success() -> None:
    from config import Settings

    pipeline = RAGQuizPipeline(
        settings=Settings(gemini_api_key="key"),
        embedding_generator=_FakeEmbeddingGenerator(),
        vector_store=_FakeVectorStore(),
        news_search=_FakeNewsSearch(),
        gemini_generator=_FakeGemini(),
    )
    result = pipeline.generate_quiz(sport="Football", difficulty="easy")
    assert result.raw_response.startswith('{"status":"ok"')
    assert len(result.historical_chunks) == 1
    assert len(result.news_results) == 1
    assert len(result.merged_context) >= 1
    assert "Insufficient context." in result.prompt


def test_rag_quiz_pipeline_input_validation() -> None:
    from config import Settings

    pipeline = RAGQuizPipeline(
        settings=Settings(gemini_api_key="key"),
        embedding_generator=_FakeEmbeddingGenerator(),
        vector_store=_FakeVectorStore(),
        news_search=_FakeNewsSearch(),
        gemini_generator=_FakeGemini(),
    )
    try:
        pipeline.generate_quiz(sport="", difficulty="easy")
        assert False, "Expected ValueError for empty sport"
    except ValueError as exc:
        assert "sport cannot be empty" in str(exc)

