"""Streamlit entrypoint for the AI Powered Sports Quiz Generation Agent."""

from __future__ import annotations

import streamlit as st

from config import get_settings
from core import (
    HistoricalFactsIndexer,
    QuizGenerationService,
    QuizSessionManager,
    QuizValidationError,
)
from database import VectorStoreError
from services import DataLoadError, EmbeddingGenerationError, GeminiServiceError, NewsSearchError, QuizParseError
from utils import configure_logging

SPORT_OPTIONS = [
    "Football",
    "Cricket",
    "Basketball",
    "Tennis",
    "Olympics",
    "Formula 1",
    "Baseball",
    "Athletics",
]
DIFFICULTY_OPTIONS = ["easy", "medium", "hard"]
UNSELECTED_OPTION = "-- Select an option --"


def main() -> None:
    st.set_page_config(
        page_title="AI Sports Quiz Agent",
        page_icon="🏅",
        layout="wide",
    )
    _apply_theme()

    settings = get_settings()
    configure_logging(settings)
    session = QuizSessionManager(st.session_state)
    st.session_state.setdefault("index_ready", False)

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">🏅 AI Powered Sports Quiz Generation Agent</div>
            <div class="hero-subtitle">Grounded quiz generation using historical facts + latest sports news (RAG)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Quiz Controls")
        selected_sport = st.selectbox("Sport", options=SPORT_OPTIONS, index=0)
        selected_difficulty = st.selectbox("Difficulty", options=DIFFICULTY_OPTIONS, index=0)
        generate_clicked = st.button("Generate Quiz", type="primary", use_container_width=True)
        reset_clicked = st.button("Reset Current Quiz", use_container_width=True)

        if reset_clicked:
            session.reset_current_quiz()
            st.success("Current quiz reset.")

        st.divider()
        st.subheader("History")
        history = st.session_state.get("quiz_history", [])
        if history:
            for idx, item in enumerate(reversed(history), start=1):
                st.write(
                    f"{idx}. {item['sport']} ({item['difficulty']}) — "
                    f"{item['score']}/{item['total_questions']}"
                )
        else:
            st.write("No attempts yet.")

    if generate_clicked:
        with st.spinner("Generating grounded quiz..."):
            try:
                if not st.session_state["index_ready"]:
                    _get_indexer().index_historical_facts()
                    st.session_state["index_ready"] = True
                generation_service = _get_generation_service()
                generation_result = generation_service.generate(
                    sport=selected_sport,
                    difficulty=selected_difficulty,
                    include_news=True,
                )
            except GeminiServiceError as exc:
                st.error(f"LLM configuration or generation error: {exc}")
                return
            except NewsSearchError as exc:
                st.warning(f"News retrieval failed, but you can retry: {exc}")
                return
            except QuizParseError as exc:
                st.error(f"Quiz parsing failed: {exc}")
                return
            except (QuizValidationError, DataLoadError, EmbeddingGenerationError, VectorStoreError) as exc:
                st.error(f"Quiz generation pipeline error: {exc}")
                return

        payload = generation_result.payload
        st.session_state["last_generation_context"] = {
            "historical_chunks": [chunk.model_dump() for chunk in generation_result.pipeline_result.historical_chunks],
            "news_results": [item.model_dump() for item in generation_result.pipeline_result.news_results],
            "merged_context": generation_result.pipeline_result.merged_context,
            "prompt": generation_result.pipeline_result.prompt,
        }

        if payload.status == "insufficient_context":
            st.warning("Insufficient context.")
        else:
            session.set_current_quiz(payload)
            st.success("Quiz generated successfully.")
            for warning in generation_result.warnings:
                st.warning(warning)

    _render_quiz(session)
    _render_source_context()


def _render_quiz(session: QuizSessionManager) -> None:
    payload_data = st.session_state.get("quiz_payload")
    if not payload_data:
        st.info("Generate a quiz from the sidebar to begin.")
        return

    from models import QuizPayload

    payload = QuizPayload.model_validate(payload_data)
    st.subheader(f"{payload.sport} Quiz ({payload.difficulty.title()})")

    for index, question in enumerate(payload.questions):
        st.markdown(f"### Q{index + 1}. {question.question}")
        options = [UNSELECTED_OPTION] + question.options
        selected = st.radio(
            f"Choose answer for question {index + 1}",
            options=options,
            key=f"answer_{index}",
            label_visibility="collapsed",
            index=0,
            disabled=st.session_state.get("quiz_submitted", False),
        )
        if selected != UNSELECTED_OPTION:
            session.set_answer(index, selected)

    submit_col, _ = st.columns([1, 3])
    with submit_col:
        submit_clicked = st.button(
            "Submit Answers",
            type="primary",
            disabled=st.session_state.get("quiz_submitted", False),
        )

    if submit_clicked:
        score, total = session.submit_quiz()
        st.success(f"Score: {score}/{total}")

    if st.session_state.get("quiz_submitted", False):
        _render_answers_and_explanations(payload)


def _render_answers_and_explanations(payload) -> None:
    st.divider()
    st.subheader("Answer Review")
    selected_answers = st.session_state.get("selected_answers", {})
    for index, question in enumerate(payload.questions):
        selected = selected_answers.get(str(index))
        is_correct = selected == question.correct_answer
        color = "#DCFCE7" if is_correct else "#FEE2E2"
        st.markdown(
            (
                f"<div style='border:1px solid #ddd;border-radius:10px;padding:12px;"
                f"background:{color};margin-bottom:10px;'>"
                f"<b>Q{index + 1}</b><br/>"
                f"Your answer: {selected or 'Not answered'}<br/>"
                f"Correct answer: {question.correct_answer}"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )
        with st.expander(f"Explanation for Q{index + 1}"):
            st.write(question.explanation)
            st.write("Source context:")
            for item in question.source_context:
                st.write(f"- {item}")


def _render_source_context() -> None:
    context = st.session_state.get("last_generation_context")
    if not context:
        return

    st.divider()
    st.subheader("Retrieved Source Context")
    with st.expander("Merged retrieval context", expanded=False):
        for line in context.get("merged_context", []):
            st.write(f"- {line}")

    with st.expander("Historical retrieval chunks", expanded=False):
        for item in context.get("historical_chunks", []):
            st.write(f"- ({item.get('source', 'unknown')}) {item.get('text', '')}")

    with st.expander("Latest news retrieval", expanded=False):
        for item in context.get("news_results", []):
            st.write(f"- {item.get('title', '')}: {item.get('snippet', '')}")
            st.write(item.get("url", ""))


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(120deg, #f8fafc 0%, #eef2ff 100%);
        }
        .hero-card {
            background: rgba(255,255,255,0.8);
            border: 1px solid #dbeafe;
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.2;
        }
        .hero-subtitle {
            margin-top: 6px;
            color: #475569;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _get_indexer() -> HistoricalFactsIndexer:
    return HistoricalFactsIndexer()


@st.cache_resource(show_spinner=False)
def _get_generation_service() -> QuizGenerationService:
    settings = get_settings()
    return QuizGenerationService(settings=settings)


if __name__ == "__main__":
    main()
