"""Domain model for normalized news search results."""

from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class NewsResult(BaseModel):
    """Normalized web news item used in RAG context."""

    title: str = Field(min_length=3)
    snippet: str = Field(min_length=5)
    url: HttpUrl
    source: str = "DuckDuckGo"

