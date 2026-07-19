"""Unit tests for logging setup."""

import logging
from pathlib import Path

from config import Settings
from utils.logging_config import (
    _reset_logging_state_for_tests,
    configure_logging,
    get_logger,
)


def _flush_handlers() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_configure_logging_creates_log_file(tmp_path: Path) -> None:
    _reset_logging_state_for_tests()
    settings = Settings(root_dir=tmp_path, log_level="INFO")

    configure_logging(settings)
    logger = get_logger("tests.logging")
    logger.info("logging-ready")
    _flush_handlers()

    log_file = tmp_path / "logs" / "app.log"
    assert log_file.exists()
    assert "logging-ready" in log_file.read_text(encoding="utf-8")


def test_configure_logging_is_idempotent(tmp_path: Path) -> None:
    _reset_logging_state_for_tests()
    settings = Settings(root_dir=tmp_path, log_level="DEBUG")

    configure_logging(settings)
    configure_logging(settings)

    handlers = logging.getLogger().handlers
    assert len(handlers) == 2
    assert any(handler.__class__.__name__ == "RotatingFileHandler" for handler in handlers)

