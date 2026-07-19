"""Pipeline for populating ChromaDB with historical facts embeddings."""

from __future__ import annotations

from dataclasses import dataclass

from database import ChromaSportsStore
from services import EmbeddingGenerator, SportsFactsLoader
from utils import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class IndexingResult:
    """Summary of one indexing run."""

    total_records: int
    indexed_records: int
    skipped_records: int


class HistoricalFactsIndexer:
    """Coordinates loading, embedding, and storing historical facts."""

    def __init__(
        self,
        *,
        loader: SportsFactsLoader | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        vector_store: ChromaSportsStore | None = None,
    ) -> None:
        self.loader = loader or SportsFactsLoader()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.vector_store = vector_store or ChromaSportsStore()

    def index_historical_facts(self, *, force_reindex: bool = False) -> IndexingResult:
        """Index dataset facts into ChromaDB with incremental behavior."""

        facts = self.loader.load_facts()
        if not facts:
            LOGGER.warning("No historical facts available for indexing.")
            return IndexingResult(total_records=0, indexed_records=0, skipped_records=0)

        existing_ids = self.vector_store.get_existing_ids() if not force_reindex else set()
        pending_facts = [fact for fact in facts if fact.id not in existing_ids]
        skipped_count = len(facts) - len(pending_facts)

        if not pending_facts:
            LOGGER.info("No new facts to index. Existing index is up to date.")
            return IndexingResult(
                total_records=len(facts),
                indexed_records=0,
                skipped_records=skipped_count,
            )

        documents = [self.embedding_generator.compose_fact_document(fact) for fact in pending_facts]
        embeddings = self.embedding_generator.embed_texts(documents)
        self.vector_store.upsert_facts(
            facts=pending_facts,
            documents=documents,
            embeddings=embeddings,
        )

        LOGGER.info("Indexed %d new facts (%d skipped)", len(pending_facts), skipped_count)
        return IndexingResult(
            total_records=len(facts),
            indexed_records=len(pending_facts),
            skipped_records=skipped_count,
        )

