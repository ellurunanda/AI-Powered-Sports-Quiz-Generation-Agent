"""DuckDuckGo search integration for latest sports news retrieval."""

from __future__ import annotations

import time

from config import Settings, get_settings
from models import NewsResult
from utils import get_logger

LOGGER = get_logger(__name__)
DDGS = None


class NewsSearchError(Exception):
    """Raised when sports news retrieval fails."""


class DuckDuckGoNewsSearch:
    """Fetches latest sports news snippets from DuckDuckGo."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._cache: dict[tuple[str, int], tuple[float, list[NewsResult]]] = {}

    def search_latest_news(self, *, sport: str, limit: int | None = None) -> list[NewsResult]:
        """Search recent sport-specific news and normalize results."""

        requested_limit = limit if limit is not None else self.settings.top_k_news
        if requested_limit < 1:
            msg = "limit must be >= 1."
            raise ValueError(msg)
        if not sport.strip():
            msg = "sport cannot be empty."
            raise ValueError(msg)

        query = f"{sport.strip()} latest news updates"
        LOGGER.info("Fetching latest news for sport='%s' with limit=%d", sport, requested_limit)
        cache_key = (sport.strip().lower(), requested_limit)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            raw_results = self._search_with_retries(query=query, max_results=requested_limit)
        except Exception as exc:
            stale = self._get_cached(cache_key, allow_stale=True)
            if stale is not None:
                LOGGER.warning("DuckDuckGo search failed; using stale cached news context.")
                return stale
            LOGGER.warning("DuckDuckGo search failed after retries; continuing without news context.")
            LOGGER.debug("DuckDuckGo failure details: %s", exc)
            return []

        normalized_results: list[NewsResult] = []
        for item in raw_results:
            title = str(item.get("title", "") or item.get("headline", "")).strip()
            snippet = str(item.get("body", "") or item.get("snippet", "") or item.get("content", "")).strip()
            url = str(item.get("href", "") or item.get("url", "") or item.get("link", "")).strip()
            if not (title and snippet and url):
                continue
            try:
                normalized_results.append(
                    NewsResult(
                        title=title,
                        snippet=snippet,
                        url=url,
                    )
                )
            except Exception:
                continue

        final_results = normalized_results[:requested_limit]
        self._cache[cache_key] = (time.time(), final_results)
        return final_results

    def _search(self, *, query: str, max_results: int) -> list[dict]:
        ddgs_cls = self._get_ddgs_class()
        with ddgs_cls() as ddgs:
            # Try dedicated news endpoint first; fall back to general text results.
            try:
                news_results = list(
                    ddgs.news(
                        query,
                        region=self.settings.duckduckgo_region,
                        timelimit="w",
                        max_results=max_results,
                    )
                )
                if news_results:
                    return news_results
            except Exception:  # noqa: BLE001
                pass

            return list(
                ddgs.text(
                    query,
                    region=self.settings.duckduckgo_region,
                    max_results=max_results,
                )
            )

    def _get_ddgs_class(self):
        global DDGS
        if DDGS is None:
            from duckduckgo_search import DDGS as _DDGS

            DDGS = _DDGS
        return DDGS

    def _search_with_retries(self, *, query: str, max_results: int) -> list[dict]:
        max_attempts = 4
        delay_seconds = 1.0
        last_exception: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return self._search(query=query, max_results=max_results)
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                if attempt >= max_attempts:
                    break
                if self._is_rate_limited(exc):
                    LOGGER.info("DuckDuckGo rate-limited (attempt %d/%d). Retrying...", attempt, max_attempts)
                else:
                    LOGGER.info("DuckDuckGo transient failure (attempt %d/%d). Retrying...", attempt, max_attempts)
                time.sleep(delay_seconds * attempt)

        raise NewsSearchError("DuckDuckGo news search failed.") from last_exception

    def _get_cached(self, cache_key: tuple[str, int], *, allow_stale: bool = False) -> list[NewsResult] | None:
        cached_entry = self._cache.get(cache_key)
        if cached_entry is None:
            return None
        timestamp, payload = cached_entry
        if allow_stale:
            return payload
        if self.settings.news_cache_ttl_seconds <= 0:
            return None
        if time.time() - timestamp > self.settings.news_cache_ttl_seconds:
            return None
        return payload

    def _is_rate_limited(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "ratelimit" in message or "rate limit" in message or "202" in message
