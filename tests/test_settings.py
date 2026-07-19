"""Unit tests for configuration settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from config.settings import Settings, get_settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.app_name == "AI Powered Sports Quiz Generation Agent"
    assert settings.environment == "development"
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.top_k_historical == 6
    assert settings.top_k_news == 5
    assert settings.quiz_question_count == 5


def test_path_properties() -> None:
    settings = Settings(root_dir=Path("C:/tmp/sports_quiz_agent"))
    assert settings.data_dir == Path("C:/tmp/sports_quiz_agent/data")
    assert settings.logs_dir == Path("C:/tmp/sports_quiz_agent/logs")
    assert settings.vector_db_dir == Path("C:/tmp/sports_quiz_agent/vector_db")
    assert settings.prompts_dir == Path("C:/tmp/sports_quiz_agent/prompts")


def test_generation_readiness_missing_key() -> None:
    settings = Settings(llm_provider="gemini", gemini_api_key=None)
    is_ready, message = settings.validate_generation_readiness()
    assert is_ready is False
    assert message == "Missing GEMINI_API_KEY."


def test_generation_readiness_valid_key() -> None:
    settings = Settings(llm_provider="gemini", gemini_api_key="test-key")
    is_ready, message = settings.validate_generation_readiness()
    assert is_ready is True
    assert message == "Configuration ready."


def test_openai_readiness_missing_key() -> None:
    settings = Settings(llm_provider="openai", openai_api_key=None)
    is_ready, message = settings.validate_generation_readiness()
    assert is_ready is False
    assert message == "Missing OPENAI_API_KEY."


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second


def test_settings_reads_env_variables(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    monkeypatch.setenv("QUIZ_QUESTION_COUNT", "7")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-a,gemini-b")
    settings = Settings()
    assert settings.gemini_api_key == "env-key"
    assert settings.quiz_question_count == 7
    assert settings.gemini_candidate_models == ["gemini-2.5-flash", "gemini-a", "gemini-b"]


def test_settings_invalid_numeric_env_raises(monkeypatch) -> None:
    monkeypatch.setenv("TOP_K_HISTORICAL", "0")
    with pytest.raises(ValidationError) as exc_info:
        Settings()
    assert "top_k_historical" in str(exc_info.value)
