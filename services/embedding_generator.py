"""Embedding generation service using sentence-transformers."""

from __future__ import annotations

from typing import Any

from config import Settings, get_settings
from models import SportsFact
from utils import get_logger

LOGGER = get_logger(__name__)

SentenceTransformer: Any | None = None
_IMPORT_EXCEPTION: Exception | None = None


class EmbeddingGenerationError(Exception):
    """Raised when embeddings cannot be generated."""


class EmbeddingGenerator:
    """Generates embeddings for historical facts and query strings."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any = None
        self._text_embedding_cache: dict[str, list[float]] = {}

    def _ensure_model(self) -> Any:
        global SentenceTransformer, _IMPORT_EXCEPTION
        if self._model is not None:
            return self._model
        if SentenceTransformer is None:
            try:
                from sentence_transformers import SentenceTransformer as _SentenceTransformer
            except Exception as exc:  # pragma: no cover - runtime dependency check
                _IMPORT_EXCEPTION = exc
                msg = "sentence-transformers is unavailable in the current environment."
                raise EmbeddingGenerationError(msg) from _IMPORT_EXCEPTION
            SentenceTransformer = _SentenceTransformer
        try:
            self._model = SentenceTransformer(self.settings.embedding_model_name)
        except Exception as exc:
            msg = f"Failed to initialize embedding model: {self.settings.embedding_model_name}"
            LOGGER.error(msg)
            raise EmbeddingGenerationError(msg) from exc
        return self._model

    def compose_fact_document(self, fact: SportsFact) -> str:
        """Build retrieval-optimized text payload for one fact record."""

        tags_text = ", ".join(fact.tags)
        return (
            f"Sport: {fact.sport}\n"
            f"Difficulty: {fact.difficulty}\n"
            f"Year: {fact.year}\n"
            f"Era: {fact.era}\n"
            f"Title: {fact.title}\n"
            f"Fact: {fact.fact}\n"
            f"Source: {fact.source}\n"
            f"Tags: {tags_text}"
        )

    def embed_texts(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        """Generate dense vectors for text inputs."""

        if not texts:
            return []
        if batch_size < 1:
            msg = "batch_size must be >= 1."
            raise ValueError(msg)

        uncached_texts = [text for text in texts if text not in self._text_embedding_cache]
        if uncached_texts:
            model = self._ensure_model()
            try:
                embeddings = model.encode(
                    uncached_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
            except Exception as exc:
                msg = "Embedding generation failed during model.encode()."
                LOGGER.error(msg)
                raise EmbeddingGenerationError(msg) from exc

            for text, vector in zip(uncached_texts, embeddings):
                self._text_embedding_cache[text] = vector.tolist()

        return [self._text_embedding_cache[text] for text in texts]

    def embed_facts(self, facts: list[SportsFact], *, batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for typed fact records."""

        documents = [self.compose_fact_document(fact) for fact in facts]
        return self.embed_texts(documents, batch_size=batch_size)
