"""Historical sports dataset loading and filtering service."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from config import Settings, get_settings
from models import DifficultyLevel, SportsFact
from utils import get_logger

LOGGER = get_logger(__name__)


class DataLoadError(Exception):
    """Raised when loading or parsing dataset records fails."""


class SportsFactsLoader:
    """Loads and filters sports facts from the seed dataset."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dataset_path = self.settings.data_dir / "historical_sports_facts.json"

    def load_facts(self) -> list[SportsFact]:
        """Load all facts from the dataset file as validated domain objects."""

        if not self.dataset_path.exists():
            msg = f"Dataset file not found: {self.dataset_path}"
            LOGGER.error(msg)
            raise DataLoadError(msg)

        try:
            payload = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            msg = f"Dataset JSON is invalid: {self.dataset_path}"
            LOGGER.error(msg)
            raise DataLoadError(msg) from exc

        if not isinstance(payload, list):
            msg = "Dataset root must be a list of fact records."
            LOGGER.error(msg)
            raise DataLoadError(msg)

        facts: list[SportsFact] = []
        try:
            for raw_item in payload:
                facts.append(SportsFact.model_validate(raw_item))
        except ValidationError as exc:
            msg = "Dataset record schema validation failed."
            LOGGER.error(msg)
            raise DataLoadError(msg) from exc

        LOGGER.info("Loaded %d historical sports facts", len(facts))
        return facts

    def query_facts(
        self,
        *,
        sport: str | None = None,
        difficulty: DifficultyLevel | None = None,
        limit: int | None = None,
    ) -> list[SportsFact]:
        """Filter facts by sport and difficulty with optional result limit."""

        facts = self.load_facts()
        normalized_sport = sport.strip().lower() if sport else None
        normalized_difficulty = difficulty.strip().lower() if difficulty else None

        filtered = [
            fact
            for fact in facts
            if (normalized_sport is None or fact.sport.lower() == normalized_sport)
            and (normalized_difficulty is None or fact.difficulty == normalized_difficulty)
        ]

        if limit is None:
            return filtered
        if limit < 1:
            msg = "limit must be >= 1 when provided."
            raise ValueError(msg)
        return filtered[:limit]

