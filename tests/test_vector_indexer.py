"""Unit tests for historical facts indexing pipeline."""

from __future__ import annotations

from core import HistoricalFactsIndexer
from models import SportsFact


class _FakeLoader:
    def __init__(self, facts: list[SportsFact]) -> None:
        self._facts = facts

    def load_facts(self) -> list[SportsFact]:
        return self._facts


class _FakeEmbeddingGenerator:
    def compose_fact_document(self, fact: SportsFact) -> str:
        return f"{fact.sport} | {fact.fact}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.5] for text in texts]


class _FakeStore:
    def __init__(self, existing_ids: set[str] | None = None) -> None:
        self._existing_ids = existing_ids or set()
        self.upsert_calls = []

    def get_existing_ids(self) -> set[str]:
        return set(self._existing_ids)

    def upsert_facts(self, *, facts, documents, embeddings) -> None:
        self.upsert_calls.append(
            {
                "facts": facts,
                "documents": documents,
                "embeddings": embeddings,
            }
        )


def _fact(fact_id: str) -> SportsFact:
    return SportsFact(
        id=fact_id,
        sport="Football",
        difficulty="easy",
        title="Sample",
        fact="Uruguay won the first FIFA World Cup in 1930 by defeating Argentina 4-2.",
        year=1930,
        era="1930s",
        source="FIFA",
        tags=["world-cup"],
    )


def test_index_historical_facts_incremental() -> None:
    facts = [_fact("fact-1"), _fact("fact-2"), _fact("fact-3")]
    indexer = HistoricalFactsIndexer(
        loader=_FakeLoader(facts),
        embedding_generator=_FakeEmbeddingGenerator(),
        vector_store=_FakeStore(existing_ids={"fact-1"}),
    )
    result = indexer.index_historical_facts()
    assert result.total_records == 3
    assert result.indexed_records == 2
    assert result.skipped_records == 1


def test_index_historical_facts_force_reindex() -> None:
    facts = [_fact("fact-1"), _fact("fact-2")]
    store = _FakeStore(existing_ids={"fact-1", "fact-2"})
    indexer = HistoricalFactsIndexer(
        loader=_FakeLoader(facts),
        embedding_generator=_FakeEmbeddingGenerator(),
        vector_store=store,
    )
    result = indexer.index_historical_facts(force_reindex=True)
    assert result.indexed_records == 2
    assert result.skipped_records == 0
    assert len(store.upsert_calls) == 1


def test_index_historical_facts_no_pending_records() -> None:
    facts = [_fact("fact-1")]
    store = _FakeStore(existing_ids={"fact-1"})
    indexer = HistoricalFactsIndexer(
        loader=_FakeLoader(facts),
        embedding_generator=_FakeEmbeddingGenerator(),
        vector_store=store,
    )
    result = indexer.index_historical_facts()
    assert result.total_records == 1
    assert result.indexed_records == 0
    assert result.skipped_records == 1
    assert len(store.upsert_calls) == 0
