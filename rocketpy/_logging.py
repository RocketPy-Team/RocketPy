"""Centralized logging configuration for RocketPy.

RocketPy follows the standard library-author convention: it logs through a
single ``"rocketpy"`` logger that has a :class:`logging.NullHandler` attached,
so the library stays silent unless the application explicitly configures
logging. Use :func:`enable_logging` for a quick, ready-to-use console output or
:func:`set_log_level` to adjust verbosity when you have already configured your
own handlers.
"""

import logging

__all__ = ["logger", "set_log_level", "enable_logging"]

#: The RocketPy library logger. Attach handlers to this (or any descendant such
#: as ``logging.getLogger("rocketpy.simulation")``) to capture RocketPy output.
logger = logging.getLogger("rocketpy")

# Library default: stay silent unless the application configures a handler.
logger.addHandler(logging.NullHandler())


def set_log_level(level):
    """Set the verbosity level of the RocketPy logger.

    Parameters
    ----------
    level : int or str
        A standard :mod:`logging` level, given either as a constant
        (``logging.INFO``) or its name. Level names are case-insensitive, so
        ``"info"``, ``"INFO"`` and ``"Debug"`` are all accepted.
    """
    if isinstance(level, str):
        level = level.upper()
    logger.setLevel(level)


def enable_logging(level="INFO"):
    """Attach a console handler to the RocketPy logger and set its level.

    This is a convenience for interactive use (notebooks, scripts) that want to
    see RocketPy's progress without configuring :mod:`logging` themselves. It is
    idempotent: calling it repeatedly will not add duplicate handlers.

    Parameters
    ----------
    level : int or str, optional
        The logging level to use. Default is ``"INFO"``.

    Returns
    -------
    logging.Logger
        The configured RocketPy logger.
    """
    handler_name = "rocketpy_console_handler"
    existing = [h for h in logger.handlers if getattr(h, "name", None) == handler_name]
    if not existing:
        handler = logging.StreamHandler()
        handler.name = handler_name
        handler.setFormatter(
            logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    set_log_level(level)
    return logger
