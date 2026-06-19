"""Unit tests for RocketPy's logging configuration."""

import logging

import pytest

from rocketpy import enable_logging, logger, set_log_level
from rocketpy.simulation.helpers.flight_phase import _FlightPhases


@pytest.fixture(autouse=True)
def _restore_logger_state():
    """Snapshot and restore the rocketpy logger so tests don't leak state."""
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    yield
    logger.handlers[:] = saved_handlers
    logger.setLevel(saved_level)


def test_logger_has_null_handler_by_default():
    """The library logger ships with a NullHandler so it stays silent."""
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)


def test_set_log_level_changes_level():
    set_log_level("DEBUG")
    assert logger.level == logging.DEBUG
    set_log_level(logging.WARNING)
    assert logger.level == logging.WARNING


def test_set_log_level_is_case_insensitive():
    set_log_level("debug")
    assert logger.level == logging.DEBUG
    set_log_level("Info")
    assert logger.level == logging.INFO


def test_enable_logging_is_idempotent_and_sets_level():
    before = len(logger.handlers)
    enable_logging("INFO")
    after_first = len(logger.handlers)
    enable_logging("DEBUG")
    after_second = len(logger.handlers)

    # Exactly one console handler is added, and not duplicated on re-call.
    assert after_first == before + 1
    assert after_second == after_first
    assert logger.level == logging.DEBUG
    assert any(
        getattr(h, "name", None) == "rocketpy_console_handler" for h in logger.handlers
    )


def test_phase_collision_emits_warning(caplog):
    """A flight-phase time collision is reported through the logger."""
    phases = _FlightPhases(
        t_initial=0.0,
        initial_derivative=lambda *args, **kwargs: None,
        max_time=10.0,
    )
    with caplog.at_level(logging.WARNING, logger="rocketpy"):
        phases.display_warning("together", "preceding")

    messages = [record.getMessage() for record in caplog.records]
    assert any("multiple events being triggered simultaneously" in m for m in messages)
