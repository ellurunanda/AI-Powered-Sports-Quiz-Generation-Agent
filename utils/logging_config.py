"""Centralized logging configuration."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import Settings, get_settings

_LOGGING_CONFIGURED = False


def configure_logging(settings: Settings | None = None) -> None:
    """Configure root logging handlers for console and file output."""

    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    resolved_settings = settings or get_settings()
    log_dir = resolved_settings.logs_dir
    _ensure_log_directory(log_dir)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, resolved_settings.log_level))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(getattr(logging, resolved_settings.log_level))
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=log_dir / "app.log",
        maxBytes=1_048_576,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, resolved_settings.log_level))
    file_handler.setFormatter(formatter)

    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("duckduckgo_search").setLevel(logging.WARNING)
    logging.getLogger("duckduckgo_search.DDGS").setLevel(logging.WARNING)
    logging.getLogger("primp").setLevel(logging.ERROR)
    logging.getLogger("primp.impersonate").setLevel(logging.ERROR)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a module-level logger by name."""

    return logging.getLogger(name)


def _ensure_log_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _reset_logging_state_for_tests() -> None:
    """Reset logger state for deterministic test execution."""

    global _LOGGING_CONFIGURED
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    _LOGGING_CONFIGURED = False
