import logging


def enable_logging(level="WARNING"):
    """Enable RocketPy logging output to the console.

    Attaches a StreamHandler to RocketPy's root logger so that internal
    runtime events (simulation progress, warnings, errors) are printed to
    the terminal. By default, only WARNING and above are shown.

    Parameters
    ----------
    level : str, optional
        The minimum logging level to display. Options are "DEBUG", "INFO",
        "WARNING", "ERROR", and "CRITICAL". Default is "WARNING".

    Examples
    --------
    Show only warnings and errors (default):

    >>> import rocketpy
    >>> rocketpy.utils.enable_logging()

    Show all internal runtime messages, including simulation progress:

    >>> import rocketpy
    >>> rocketpy.utils.enable_logging(level="DEBUG")

    Show confirmations like "Simulation completed" and "File saved":

    >>> import rocketpy
    >>> rocketpy.utils.enable_logging(level="INFO")
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: '{level}'")

    logger = logging.getLogger("rocketpy")

    # Remove any existing StreamHandlers to avoid duplicate messages
    logger.handlers = [
        h for h in logger.handlers if not isinstance(h, logging.StreamHandler)
    ]

    logger.setLevel(numeric_level)

    handler = logging.StreamHandler()
    handler.setLevel(numeric_level)
    handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))

    logger.addHandler(handler)
