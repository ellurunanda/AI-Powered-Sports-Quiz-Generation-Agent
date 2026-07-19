"""Unit tests for historical data loading service."""

import json
from pathlib import Path

import pytest

from config import Settings
from services import DataLoadError, SportsFactsLoader


def test_load_facts_success() -> None:
    loader = SportsFactsLoader()
    facts = loader.load_facts()
    assert len(facts) >= 20
    assert all(fact.id for fact in facts)


def test_query_facts_by_sport_and_difficulty() -> None:
    loader = SportsFactsLoader()
    results = loader.query_facts(sport="Football", difficulty="easy", limit=2)
    assert len(results) <= 2
    assert all(item.sport.lower() == "football" for item in results)
    assert all(item.difficulty == "easy" for item in results)


def test_query_facts_with_invalid_limit() -> None:
    loader = SportsFactsLoader()
    with pytest.raises(ValueError):
        loader.query_facts(limit=0)


def test_loader_raises_when_dataset_missing(tmp_path: Path) -> None:
    settings = Settings(root_dir=tmp_path)
    loader = SportsFactsLoader(settings=settings)
    with pytest.raises(DataLoadError, match="Dataset file not found"):
        loader.load_facts()


def test_loader_raises_on_invalid_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "historical_sports_facts.json").write_text("{not-json}", encoding="utf-8")
    settings = Settings(root_dir=tmp_path)
    loader = SportsFactsLoader(settings=settings)
    with pytest.raises(DataLoadError, match="Dataset JSON is invalid"):
        loader.load_facts()


def test_loader_raises_on_invalid_schema(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = [{"id": "x1", "sport": "Football"}]
    (data_dir / "historical_sports_facts.json").write_text(json.dumps(payload), encoding="utf-8")
    settings = Settings(root_dir=tmp_path)
    loader = SportsFactsLoader(settings=settings)
    with pytest.raises(DataLoadError, match="schema validation failed"):
        loader.load_facts()

