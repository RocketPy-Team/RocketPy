Logging
=======

RocketPy uses Python's built-in `logging <https://docs.python.org/3/library/logging.html>`_
module to report internal runtime events, such as simulation progress,
warnings and errors. By design, RocketPy is **silent by default**: unless
you configure a handler, no log records are printed to the console. This
follows the best practices recommended for Python libraries and keeps
RocketPy from cluttering the output of applications that embed it.

.. note::
    This only affects internal runtime events. Output that you explicitly
    request, such as ``flight.info()`` or ``rocket.all_info()``, continues
    to print directly to the terminal regardless of the logging
    configuration described here.

Enabling logging
-----------------

The easiest way to see RocketPy's internal log messages is the
:func:`rocketpy.utils.enable_logging` helper, which attaches a console
handler to RocketPy's logger hierarchy:

.. jupyter-execute::

    import rocketpy

    rocketpy.utils.enable_logging(level="INFO")

Once enabled, operations such as saving a file or completing a simulation
will emit messages to the console, for example:

.. code-block:: text

    INFO | rocketpy.simulation.flight | Simulation completed successfully.

Log levels
----------

The ``level`` parameter controls the minimum severity of messages that are
shown:

- ``DEBUG``: high-frequency, low-level detail (e.g. solver iterations),
  useful when troubleshooting a simulation.
- ``INFO``: confirmations of completed operations (e.g. simulation
  completed, file saved).
- ``WARNING`` (default): unexpected but recoverable situations (e.g. a
  missing motor, an automatically corrected geometry parameter).
- ``ERROR``: a feature is broken or an optional dependency is missing.

.. jupyter-execute::

    # Show every internal runtime message, including solver ticks
    rocketpy.utils.enable_logging(level="DEBUG")

Filtering by module
--------------------

Because each RocketPy module exposes its own logger (e.g.
``rocketpy.simulation.flight``, ``rocketpy.environment.environment``), you
can rely on the standard ``logging`` module to filter or redirect messages
from specific modules, without using :func:`rocketpy.utils.enable_logging`
at all:

.. code-block:: python

    import logging

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))

    # Only show logs coming from the Flight simulation module
    flight_logger = logging.getLogger("rocketpy.simulation.flight")
    flight_logger.addHandler(handler)
    flight_logger.setLevel(logging.INFO)
