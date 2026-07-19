"""RAG pipeline orchestration for grounded sports quiz generation."""

from __future__ import annotations

from dataclasses import dataclass

from config import Settings, get_settings
from database import ChromaSportsStore
from models import NewsResult, RetrievalChunk
from prompts.quiz_prompt_builder import build_grounded_quiz_prompt
from services import DuckDuckGoNewsSearch, EmbeddingGenerator, GeminiQuizGenerator, NewsSearchError
from utils import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class RAGQuizPipelineResult:
    """Output payload of one RAG generation run."""

    prompt: str
    historical_chunks: list[RetrievalChunk]
    news_results: list[NewsResult]
    merged_context: list[str]
    raw_response: str


class RAGQuizPipeline:
    """End-to-end RAG pipeline for sports quiz generation."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        vector_store: ChromaSportsStore | None = None,
        news_search: DuckDuckGoNewsSearch | None = None,
        gemini_generator: GeminiQuizGenerator | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedding_generator = embedding_generator or EmbeddingGenerator(self.settings)
        self.vector_store = vector_store or ChromaSportsStore(self.settings)
        self.news_search = news_search or DuckDuckGoNewsSearch(self.settings)
        self.gemini_generator = gemini_generator or GeminiQuizGenerator(self.settings)

    def generate_quiz(self, *, sport: str, difficulty: str, include_news: bool = True) -> RAGQuizPipelineResult:
        """Generate grounded quiz output based on historical and latest-news context."""

        if not sport.strip():
            raise ValueError("sport cannot be empty.")
        if difficulty.strip().lower() not in {"easy", "medium", "hard"}:
            raise ValueError("difficulty must be one of: easy, medium, hard.")

        query_text = f"{sport.strip()} {difficulty.strip().lower()} historical facts and key events"
        query_embedding = self.embedding_generator.embed_texts([query_text])[0]
        historical_chunks = self.vector_store.query_facts(
            query_embedding=query_embedding,
            top_k=self.settings.top_k_historical,
            sport=sport.strip(),
            difficulty=difficulty.strip().lower(),
        )

        if include_news:
            try:
                news_results = self.news_search.search_latest_news(
                    sport=sport.strip(),
                    limit=self.settings.top_k_news,
                )
            except NewsSearchError as exc:
                LOGGER.warning("News retrieval failed; continuing with historical context only: %s", exc)
                news_results = []
        else:
            news_results = []

        merged_context = self._merge_and_rank_context(historical_chunks=historical_chunks, news_results=news_results)
        prompt = build_grounded_quiz_prompt(
            sport=sport.strip(),
            difficulty=difficulty.strip().lower(),
            question_count=self.settings.quiz_question_count,
            historical_chunks=historical_chunks,
            news_results=news_results,
        )
        raw_response = self.gemini_generator.generate_quiz_text(prompt)

        return RAGQuizPipelineResult(
            prompt=prompt,
            historical_chunks=historical_chunks,
            news_results=news_results,
            merged_context=merged_context,
            raw_response=raw_response,
        )

    def _merge_and_rank_context(
        self,
        *,
        historical_chunks: list[RetrievalChunk],
        news_results: list[NewsResult],
    ) -> list[str]:
        ranked_items: list[tuple[float, str]] = []
        for chunk in historical_chunks:
            ranked_items.append((chunk.score, f"[Historical] {chunk.text}"))

        for idx, result in enumerate(news_results):
            score = 0.70 - (idx * 0.01)
            ranked_items.append((score, f"[News] {result.title}: {result.snippet}"))

        ranked_items.sort(key=lambda item: item[0], reverse=True)

        deduped: list[str] = []
        seen: set[str] = set()
        for _, text in ranked_items:
            key = text.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(text)
        return deduped
