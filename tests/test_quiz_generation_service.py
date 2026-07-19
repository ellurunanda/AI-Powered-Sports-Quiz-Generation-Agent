"""Unit tests for high-level quiz generation service."""

from core.quiz_generation_service import QuizGenerationService
from models import QuizPayload


class _FakeRAGPipeline:
    def generate_quiz(self, *, sport: str, difficulty: str, include_news: bool = True):
        return type(
            "RagResult",
            (),
            {
                "raw_response": (
                    '{"status":"ok","sport":"Football","difficulty":"easy","questions":[{"question":"Q?",'
                    '"options":["A","B","C","D"],"correct_answer":"A","explanation":"Because.","source_context":["ctx"]}]}'
                ),
                "prompt": "prompt",
                "historical_chunks": [],
                "news_results": [],
                "merged_context": [],
            },
        )()


class _FakeParser:
    def parse(self, raw_text: str) -> QuizPayload:
        return QuizPayload.model_validate(
            {
                "status": "ok",
                "sport": "Football",
                "difficulty": "easy",
                "questions": [
                    {
                        "question": "Who won?",
                        "options": ["A", "B", "C", "D"],
                        "correct_answer": "A",
                        "explanation": "Based on context.",
                        "source_context": ["ctx"],
                    }
                ],
            }
        )


class _FakeValidator:
    def validate(self, **kwargs):
        return type("ValidationResult", (), {"is_valid": True, "reason": "ok"})()


def test_quiz_generation_service_success() -> None:
    from config import Settings

    service = QuizGenerationService(
        settings=Settings(gemini_api_key="x", quiz_question_count=1),
        rag_pipeline=_FakeRAGPipeline(),
        parser=_FakeParser(),
        validator=_FakeValidator(),
    )
    result = service.generate(sport="Football", difficulty="easy")
    assert result.payload.status == "ok"
    assert result.payload.sport == "Football"
    assert result.warnings == []


class _ManyQuestionsParser:
    def parse(self, raw_text: str) -> QuizPayload:
        return QuizPayload.model_validate(
            {
                "status": "ok",
                "sport": "Football",
                "difficulty": "easy",
                "questions": [
                    {
                        "question": "Question one?",
                        "options": ["A", "B", "C", "D"],
                        "correct_answer": "A",
                        "explanation": "Explanation one.",
                        "source_context": ["ctx1"],
                    },
                    {
                        "question": "Question two?",
                        "options": ["A", "B", "C", "D"],
                        "correct_answer": "A",
                        "explanation": "Explanation two.",
                        "source_context": ["ctx2"],
                    },
                ],
            }
        )


def test_quiz_generation_service_trims_extra_questions() -> None:
    from config import Settings

    service = QuizGenerationService(
        settings=Settings(gemini_api_key="x", quiz_question_count=1),
        rag_pipeline=_FakeRAGPipeline(),
        parser=_ManyQuestionsParser(),
        validator=_FakeValidator(),
    )
    result = service.generate(sport="Football", difficulty="easy")
    assert len(result.payload.questions) == 1
    assert result.warnings


class _FewerQuestionsParser:
    def parse(self, raw_text: str) -> QuizPayload:
        return QuizPayload.model_validate(
            {
                "status": "ok",
                "sport": "Football",
                "difficulty": "easy",
                "questions": [
                    {
                        "question": "Only question?",
                        "options": ["A", "B", "C", "D"],
                        "correct_answer": "A",
                        "explanation": "Single explanation text.",
                        "source_context": ["ctx"],
                    }
                ],
            }
        )


def test_quiz_generation_service_accepts_fewer_questions_with_warning() -> None:
    from config import Settings

    service = QuizGenerationService(
        settings=Settings(gemini_api_key="x", quiz_question_count=3),
        rag_pipeline=_FakeRAGPipeline(),
        parser=_FewerQuestionsParser(),
        validator=_FakeValidator(),
    )
    result = service.generate(sport="Football", difficulty="easy")
    assert len(result.payload.questions) == 1
    assert result.warnings
