"""Unit tests for DuckDuckGo news search service."""

from __future__ import annotations

import pytest

from config import Settings
from services.news_search import DuckDuckGoNewsSearch


def test_search_latest_news_validates_inputs() -> None:
    service = DuckDuckGoNewsSearch(settings=Settings())
    with pytest.raises(ValueError, match="sport cannot be empty"):
        service.search_latest_news(sport="   ")
    with pytest.raises(ValueError, match="limit must be >= 1"):
        service.search_latest_news(sport="Football", limit=0)


def test_search_latest_news_normalizes_results(monkeypatch) -> None:
    service = DuckDuckGoNewsSearch(settings=Settings(top_k_news=3))

    def _fake_search(*, query: str, max_results: int):
        assert "latest news updates" in query
        assert max_results == 2
        return [
            {"title": "A", "body": "Too short", "href": "https://example.com/a"},
            {
                "title": "Football transfer roundup",
                "body": "Major updates from top European clubs.",
                "href": "https://example.com/b",
            },
            {"title": "Missing URL", "body": "Will be skipped", "href": ""},
        ]

    monkeypatch.setattr(service, "_search", _fake_search)
    results = service.search_latest_news(sport="Football", limit=2)
    assert len(results) == 1
    assert results[0].title == "Football transfer roundup"
    assert str(results[0].url) == "https://example.com/b"


def test_search_latest_news_provider_failure_returns_empty_list(monkeypatch) -> None:
    service = DuckDuckGoNewsSearch(settings=Settings())

    def _failing_search(*, query: str, max_results: int):
        raise RuntimeError("network down")

    monkeypatch.setattr(service, "_search", _failing_search)
    results = service.search_latest_news(sport="Cricket", limit=3)
    assert results == []


def test_search_latest_news_uses_cache(monkeypatch) -> None:
    service = DuckDuckGoNewsSearch(settings=Settings(top_k_news=2, news_cache_ttl_seconds=600))
    calls = {"count": 0}

    def _fake_search(*, query: str, max_results: int):
        calls["count"] += 1
        return [
            {
                "title": "Cached headline",
                "body": "A meaningful sports update for caching.",
                "href": "https://example.com/cache",
            }
        ]

    monkeypatch.setattr(service, "_search", _fake_search)
    first = service.search_latest_news(sport="Football", limit=1)
    second = service.search_latest_news(sport="Football", limit=1)
    assert len(first) == 1
    assert len(second) == 1
    assert calls["count"] == 1


def test_search_latest_news_uses_stale_cache_on_failure(monkeypatch) -> None:
    service = DuckDuckGoNewsSearch(settings=Settings(top_k_news=2, news_cache_ttl_seconds=0))

    def _success_search(*, query: str, max_results: int):
        return [
            {
                "title": "Cached headline",
                "body": "A meaningful sports update for caching.",
                "href": "https://example.com/cache",
            }
        ]

    monkeypatch.setattr(service, "_search", _success_search)
    cached = service.search_latest_news(sport="Football", limit=1)
    assert len(cached) == 1

    def _failing_search(*, query: str, max_results: int):
        raise RuntimeError("202 ratelimit")

    monkeypatch.setattr(service, "_search", _failing_search)
    stale = service.search_latest_news(sport="Football", limit=1)
    assert len(stale) == 1
