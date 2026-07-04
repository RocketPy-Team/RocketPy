Exceptions and Warnings
=======================

RocketPy raises a small hierarchy of custom exceptions and warnings so that
error handling can be more specific than catching plain ``ValueError`` or
``UserWarning``. All exceptions inherit from :class:`~rocketpy.exceptions.RocketPyError`,
which makes it possible to catch any RocketPy-specific error with a single
``except`` clause while still deriving from the relevant built-in type
(e.g. :class:`~rocketpy.exceptions.InvalidParameterError` is also a
``ValueError``).

.. automodule:: rocketpy.exceptions
   :members:
   :show-inheritance:
