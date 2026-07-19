"""Unit tests for embedding generator service."""

from __future__ import annotations

import pytest

from config import Settings
from models import SportsFact
from services.embedding_generator import (
    EmbeddingGenerationError,
    EmbeddingGenerator,
)


class _FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _FakeModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> list[_FakeVector]:
        assert batch_size >= 1
        assert show_progress_bar is False
        assert normalize_embeddings is True
        return [_FakeVector([float(len(text)), 1.0]) for text in texts]


def _sample_fact() -> SportsFact:
    return SportsFact(
        id="football-1930-first-world-cup",
        sport="Football",
        difficulty="easy",
        title="First FIFA World Cup winner",
        fact="Uruguay won the first FIFA World Cup in 1930 by defeating Argentina 4-2.",
        year=1930,
        era="1930s",
        source="FIFA tournament archives",
        tags=["world-cup", "uruguay"],
    )


def test_compose_fact_document_contains_expected_fields() -> None:
    generator = EmbeddingGenerator(settings=Settings())
    document = generator.compose_fact_document(_sample_fact())
    assert "Sport: Football" in document
    assert "Difficulty: easy" in document
    assert "Tags: world-cup, uruguay" in document


def test_embed_texts_returns_vectors(monkeypatch) -> None:
    monkeypatch.setattr("services.embedding_generator.SentenceTransformer", _FakeModel)
    generator = EmbeddingGenerator(settings=Settings())
    vectors = generator.embed_texts(["alpha", "beta"], batch_size=8)
    assert vectors == [[5.0, 1.0], [4.0, 1.0]]


def test_embed_texts_empty_input_returns_empty_list(monkeypatch) -> None:
    monkeypatch.setattr("services.embedding_generator.SentenceTransformer", _FakeModel)
    generator = EmbeddingGenerator(settings=Settings())
    assert generator.embed_texts([]) == []


def test_embed_texts_invalid_batch_size_raises(monkeypatch) -> None:
    monkeypatch.setattr("services.embedding_generator.SentenceTransformer", _FakeModel)
    generator = EmbeddingGenerator(settings=Settings())
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        generator.embed_texts(["x"], batch_size=0)


def test_embed_facts_uses_compose_pipeline(monkeypatch) -> None:
    monkeypatch.setattr("services.embedding_generator.SentenceTransformer", _FakeModel)
    generator = EmbeddingGenerator(settings=Settings())
    vectors = generator.embed_facts([_sample_fact()], batch_size=4)
    assert len(vectors) == 1
    assert len(vectors[0]) == 2


def test_model_initialization_failure_is_wrapped(monkeypatch) -> None:
    def _broken_model(_: str) -> None:
        raise RuntimeError("model init failed")

    monkeypatch.setattr("services.embedding_generator.SentenceTransformer", _broken_model)
    generator = EmbeddingGenerator(settings=Settings())
    with pytest.raises(EmbeddingGenerationError, match="Failed to initialize embedding model"):
        generator.embed_texts(["hello"])


def test_embed_texts_uses_cache_for_repeated_inputs(monkeypatch) -> None:
    class _CountingModel(_FakeModel):
        call_count = 0

        def encode(self, texts: list[str], *, batch_size: int, show_progress_bar: bool, normalize_embeddings: bool):
            _CountingModel.call_count += 1
            return super().encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
            )

    monkeypatch.setattr("services.embedding_generator.SentenceTransformer", _CountingModel)
    generator = EmbeddingGenerator(settings=Settings())
    first = generator.embed_texts(["repeat-me"])
    second = generator.embed_texts(["repeat-me"])
    assert first == second
    assert _CountingModel.call_count == 1
