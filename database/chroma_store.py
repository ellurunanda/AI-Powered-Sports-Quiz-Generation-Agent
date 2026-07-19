"""ChromaDB persistence and retrieval service."""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from config import Settings, get_settings
from models import RetrievalChunk, SportsFact
from utils import get_logger

LOGGER = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


class ChromaSportsStore:
    """Persistent ChromaDB-backed storage for historical sports facts."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.vector_db_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.settings.vector_db_dir))
        self._collection: Collection | None = None

    def _get_collection(self) -> Collection:
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def upsert_facts(
        self,
        *,
        facts: list[SportsFact],
        documents: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Insert or update fact records with vectors and metadata."""

        if not (len(facts) == len(documents) == len(embeddings)):
            msg = "facts, documents, and embeddings must have equal lengths."
            raise ValueError(msg)
        if not facts:
            return

        collection = self._get_collection()
        ids = [fact.id for fact in facts]
        metadatas = [
            {
                "sport": fact.sport,
                "difficulty": fact.difficulty,
                "source": fact.source,
                "year": fact.year,
                "era": fact.era,
                "title": fact.title,
            }
            for fact in facts
        ]

        try:
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            msg = "Failed to upsert vectors into ChromaDB."
            LOGGER.error(msg)
            raise VectorStoreError(msg) from exc

        LOGGER.info("Upserted %d fact vectors into ChromaDB collection '%s'", len(ids), self.settings.chroma_collection_name)

    def get_existing_ids(self) -> set[str]:
        """Fetch currently indexed IDs to support incremental ingestion."""

        collection = self._get_collection()
        try:
            response = collection.get(include=[])
        except Exception as exc:
            msg = "Failed to fetch existing IDs from ChromaDB."
            LOGGER.error(msg)
            raise VectorStoreError(msg) from exc

        ids = response.get("ids", [])
        return set(ids)

    def query_facts(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        sport: str | None = None,
        difficulty: str | None = None,
    ) -> list[RetrievalChunk]:
        """Retrieve nearest historical fact chunks with optional metadata filters."""

        if top_k < 1:
            msg = "top_k must be >= 1."
            raise ValueError(msg)

        where_clause = self._build_where_clause(sport=sport, difficulty=difficulty)
        collection = self._get_collection()

        try:
            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None,
            )
        except Exception as exc:
            msg = "Failed to query ChromaDB."
            LOGGER.error(msg)
            raise VectorStoreError(msg) from exc

        return self._normalize_query_response(response)

    def _build_where_clause(self, *, sport: str | None, difficulty: str | None) -> dict[str, Any]:
        normalized_sport = sport.strip() if sport else None
        normalized_difficulty = difficulty.strip().lower() if difficulty else None

        if normalized_sport and normalized_difficulty:
            return {
                "$and": [
                    {"sport": normalized_sport},
                    {"difficulty": normalized_difficulty},
                ]
            }
        if normalized_sport:
            return {"sport": normalized_sport}
        if normalized_difficulty:
            return {"difficulty": normalized_difficulty}
        return {}

    def _normalize_query_response(self, response: dict[str, Any]) -> list[RetrievalChunk]:
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        chunks: list[RetrievalChunk] = []
        for idx, chunk_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            distance = distances[idx] if idx < len(distances) else 1.0
            score = 1.0 / (1.0 + float(distance))

            chunks.append(
                RetrievalChunk(
                    id=chunk_id,
                    sport=str(metadata.get("sport", "unknown")),
                    difficulty=str(metadata.get("difficulty", "unknown")),
                    source=str(metadata.get("source", "unknown")),
                    text=documents[idx] if idx < len(documents) else "unknown",
                    score=score,
                    metadata=metadata,
                )
            )

        return chunks
