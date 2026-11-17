"""Custom exceptions and warnings for RocketPy."""

# TODO: progressively adopt these custom exceptions across the codebase.
# Many modules still ``raise ValueError``/``TypeError`` directly; migrate those
# to the appropriate ``RocketPyError`` subclass (adding new exception types here
# as needed) so users can reliably catch ``RocketPyError`` and its subclasses.


class RocketPyError(Exception):
    """Base class for all RocketPy exceptions."""


class InvalidParameterError(RocketPyError, ValueError):
    """Raised when a constructor parameter has an invalid value (e.g. negative
    radius or mass)."""


class InvalidInertiaError(RocketPyError, ValueError):
    """Raised when the inertia tuple/list does not have the expected length."""


class StochasticTypeError(RocketPyError, TypeError, AssertionError):
    """Raised when a stochastic input has the wrong type or format (e.g. a
    string where a number or tuple is expected).

    Also an ``AssertionError``, which these checks used to raise, so existing
    ``except AssertionError`` clauses keep working.
    """


class StochasticValueError(RocketPyError, ValueError, AssertionError):
    """Raised when a stochastic input has the right type but an invalid value
    (e.g. a tuple of the wrong length or an out-of-range ensemble member).

    Also an ``AssertionError``, which these checks used to raise, so existing
    ``except AssertionError`` clauses keep working.
    """


class UnstableRocketWarning(UserWarning):
    """Issued when the rocket's static margin is negative at motor ignition,
    indicating an aerodynamically unstable configuration.

    Not issued when the rocket has any ``GenericSurface`` aerodynamic
    surfaces, since their lift coefficient derivative is not accounted for
    in the center of pressure calculation, making the static margin
    unreliable for this check in that case.
    """
