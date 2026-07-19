"""Typed retrieval output model for RAG context assembly."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalChunk(BaseModel):
    """One retrieved context chunk returned from vector search."""

    id: str = Field(min_length=1)
    sport: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)
    score: float
    metadata: dict

