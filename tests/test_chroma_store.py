"""Unit tests for ChromaDB store service."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import Settings
from database import ChromaSportsStore
from models import SportsFact


class _FakeCollection:
    def __init__(self) -> None:
        self.last_upsert = None
        self.last_query = None

    def upsert(self, *, ids, documents, embeddings, metadatas) -> None:
        self.last_upsert = {
            "ids": ids,
            "documents": documents,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }

    def query(self, *, query_embeddings, n_results, where=None):
        self.last_query = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "where": where,
        }
        return {
            "ids": [["f1"]],
            "documents": [["Fact text"]],
            "metadatas": [[{"sport": "Football", "difficulty": "easy", "source": "src"}]],
            "distances": [[0.2]],
        }

    def get(self, include):
        return {"ids": ["f1", "f2"]}


class _FakeClient:
    def __init__(self, path: str) -> None:
        self.path = path
        self.collection = _FakeCollection()
        self.last_collection_name = None

    def get_or_create_collection(self, *, name: str, metadata: dict):
        self.last_collection_name = name
        return self.collection


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


def test_upsert_facts_length_mismatch_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("database.chroma_store.chromadb.PersistentClient", _FakeClient)
    store = ChromaSportsStore(settings=Settings(root_dir=tmp_path))
    with pytest.raises(ValueError, match="equal lengths"):
        store.upsert_facts(facts=[_sample_fact()], documents=[], embeddings=[])


def test_upsert_facts_success(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("database.chroma_store.chromadb.PersistentClient", _FakeClient)
    store = ChromaSportsStore(settings=Settings(root_dir=tmp_path))
    doc = "Sport: Football\nFact: Uruguay won..."
    emb = [0.1, 0.2, 0.3]
    store.upsert_facts(facts=[_sample_fact()], documents=[doc], embeddings=[emb])
    collection = store._get_collection()
    assert collection.last_upsert["ids"] == ["football-1930-first-world-cup"]
    assert collection.last_upsert["documents"] == [doc]
    assert collection.last_upsert["embeddings"] == [emb]


def test_query_facts_applies_filters_and_normalizes_response(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("database.chroma_store.chromadb.PersistentClient", _FakeClient)
    store = ChromaSportsStore(settings=Settings(root_dir=tmp_path))
    chunks = store.query_facts(query_embedding=[0.1, 0.2], top_k=3, sport="Football", difficulty="easy")
    assert len(chunks) == 1
    assert chunks[0].id == "f1"
    assert chunks[0].sport == "Football"
    assert chunks[0].difficulty == "easy"
    assert chunks[0].score > 0.0
    collection = store._get_collection()
    assert collection.last_query["where"] == {
        "$and": [{"sport": "Football"}, {"difficulty": "easy"}]
    }


def test_get_existing_ids_returns_set(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("database.chroma_store.chromadb.PersistentClient", _FakeClient)
    store = ChromaSportsStore(settings=Settings(root_dir=tmp_path))
    assert store.get_existing_ids() == {"f1", "f2"}
