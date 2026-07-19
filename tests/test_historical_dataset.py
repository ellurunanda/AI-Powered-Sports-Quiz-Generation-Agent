"""Validation tests for the historical sports facts seed dataset."""

import json
from pathlib import Path


DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "historical_sports_facts.json"
REQUIRED_KEYS = {"id", "sport", "difficulty", "title", "fact", "year", "era", "source", "tags"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def _load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def test_dataset_file_exists() -> None:
    assert DATASET_PATH.exists()


def test_dataset_has_records() -> None:
    records = _load_dataset()
    assert len(records) >= 20


def test_dataset_schema_and_values() -> None:
    records = _load_dataset()
    ids: set[str] = set()
    for record in records:
        assert REQUIRED_KEYS.issubset(record.keys())
        assert record["id"] not in ids
        ids.add(record["id"])
        assert record["difficulty"] in VALID_DIFFICULTIES
        assert isinstance(record["year"], int)
        assert 1800 <= record["year"] <= 2100
        assert isinstance(record["tags"], list)
        assert len(record["tags"]) >= 1
        assert isinstance(record["fact"], str) and len(record["fact"]) > 30


def test_dataset_contains_multiple_sports() -> None:
    records = _load_dataset()
    sports = {record["sport"] for record in records}
    assert len(sports) >= 6

