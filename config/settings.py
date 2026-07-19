"""Application configuration management."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed settings container loaded from environment and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "AI Powered Sports Quiz Generation Agent"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    llm_provider: Literal["gemini", "openai"] = "gemini"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_fallback_models: str = "gemini-2.0-flash,gemini-2.0-flash-lite,gemini-1.5-flash,gemini-1.5-flash-8b"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_collection_name: str = "sports_facts"

    top_k_historical: int = Field(default=6, ge=1, le=20)
    top_k_news: int = Field(default=5, ge=1, le=20)
    quiz_question_count: int = Field(default=5, ge=1, le=10)
    request_timeout_seconds: int = Field(default=30, ge=5, le=120)
    duckduckgo_region: str = "wt-wt"
    news_cache_ttl_seconds: int = Field(default=300, ge=0, le=3600)

    root_dir: Path = Field(default_factory=_resolve_project_root)

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def logs_dir(self) -> Path:
        return self.root_dir / "logs"

    @property
    def vector_db_dir(self) -> Path:
        return self.root_dir / "vector_db"

    @property
    def prompts_dir(self) -> Path:
        return self.root_dir / "prompts"

    def validate_generation_readiness(self) -> tuple[bool, str]:
        if self.llm_provider == "openai":
            if not self.openai_api_key:
                return False, "Missing OPENAI_API_KEY."
            return True, "Configuration ready."

        if not self.gemini_api_key:
            return False, "Missing GEMINI_API_KEY."
        return True, "Configuration ready."

    @property
    def gemini_candidate_models(self) -> list[str]:
        candidates = [self.gemini_model]
        fallback_models = [item.strip() for item in self.gemini_fallback_models.split(",") if item.strip()]
        for model in fallback_models:
            if model not in candidates:
                candidates.append(model)
        return candidates


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache application settings."""

    return Settings()
