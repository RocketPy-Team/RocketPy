"""Unit tests for rocketpy.utils module."""

import logging

import pytest

import rocketpy


@pytest.fixture(autouse=True)
def reset_rocketpy_logger():
    """Reset the rocketpy logger to its original state after each test."""
    logger = logging.getLogger("rocketpy")
    original_level = logger.level
    original_handlers = logger.handlers[:]
    yield
    logger.handlers = original_handlers
    logger.setLevel(original_level)


def test_enable_logging_adds_stream_handler():
    """enable_logging() must attach a StreamHandler to the rocketpy logger."""
    rocketpy.utils.enable_logging(level="INFO")

    logger = logging.getLogger("rocketpy")
    stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(stream_handlers) >= 1


def test_enable_logging_sets_correct_level():
    """enable_logging() must set the requested level on the rocketpy logger."""
    rocketpy.utils.enable_logging(level="DEBUG")
    assert logging.getLogger("rocketpy").level == logging.DEBUG

    rocketpy.utils.enable_logging(level="WARNING")
    assert logging.getLogger("rocketpy").level == logging.WARNING


def test_enable_logging_no_duplicate_handlers():
    """Calling enable_logging() twice must not duplicate StreamHandlers."""
    rocketpy.utils.enable_logging(level="INFO")
    rocketpy.utils.enable_logging(level="INFO")

    logger = logging.getLogger("rocketpy")
    stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(stream_handlers) == 1


def test_enable_logging_replaces_handler_on_level_change():
    """Calling enable_logging() with a new level must replace the old handler."""
    rocketpy.utils.enable_logging(level="WARNING")
    rocketpy.utils.enable_logging(level="DEBUG")

    logger = logging.getLogger("rocketpy")
    stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(stream_handlers) == 1
    assert logger.level == logging.DEBUG


def test_enable_logging_invalid_level_raises():
    """enable_logging() must raise ValueError for an unrecognised level string."""
    with pytest.raises(ValueError, match="Invalid logging level"):
        rocketpy.utils.enable_logging(level="INVALID")


def test_enable_logging_messages_are_captured(caplog):
    """After enable_logging(), internal rocketpy log messages must be visible."""
    rocketpy.utils.enable_logging(level="DEBUG")

    with caplog.at_level(logging.DEBUG, logger="rocketpy"):
        logger = logging.getLogger("rocketpy.simulation.flight")
        logger.info("test message from flight")

    assert "test message from flight" in caplog.text
