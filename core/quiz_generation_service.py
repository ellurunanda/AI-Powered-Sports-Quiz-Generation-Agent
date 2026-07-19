"""High-level quiz generation service combining RAG, parsing, and validation."""

from __future__ import annotations

from dataclasses import dataclass

from config import Settings, get_settings
from core.quiz_validator import QuizValidationError, QuizValidator
from core.rag_quiz_pipeline import RAGQuizPipeline, RAGQuizPipelineResult
from models import QuizPayload
from services import QuizResponseParser


@dataclass(frozen=True)
class QuizGenerationResult:
    """Final generation result ready for UI rendering."""

    payload: QuizPayload
    pipeline_result: RAGQuizPipelineResult
    warnings: list[str]


class QuizGenerationService:
    """Coordinates generation and quality validation of quiz payloads."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        rag_pipeline: RAGQuizPipeline | None = None,
        parser: QuizResponseParser | None = None,
        validator: QuizValidator | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.rag_pipeline = rag_pipeline or RAGQuizPipeline(settings=self.settings)
        self.parser = parser or QuizResponseParser()
        self.validator = validator or QuizValidator()

    def generate(self, *, sport: str, difficulty: str, include_news: bool = True) -> QuizGenerationResult:
        pipeline_result = self.rag_pipeline.generate_quiz(
            sport=sport,
            difficulty=difficulty,
            include_news=include_news,
        )
        payload = self.parser.parse(pipeline_result.raw_response)
        warnings: list[str] = []

        if payload.status == "ok":
            actual_count = len(payload.questions)
            expected_count = self.settings.quiz_question_count
            if actual_count == 0:
                raise QuizValidationError("Quiz generation returned zero questions.")
            if actual_count > expected_count:
                payload = payload.model_copy(update={"questions": payload.questions[:expected_count]})
                warnings.append(
                    f"LLM returned {actual_count} questions; trimmed to requested {expected_count}."
                )
            elif actual_count < expected_count:
                warnings.append(
                    f"LLM returned {actual_count} questions instead of requested {expected_count}."
                )

        self.validator.validate(
            payload=payload,
            expected_sport=sport,
            expected_difficulty=difficulty,
            expected_question_count=len(payload.questions),
        )
        return QuizGenerationResult(payload=payload, pipeline_result=pipeline_result, warnings=warnings)
