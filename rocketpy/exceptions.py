"""Custom exceptions and warnings for RocketPy."""

# TODO: progressively adopt these custom exceptions across the codebase.
# Many modules still ``raise ValueError``/``TypeError`` directly; migrate those
# to the appropriate ``RocketPyError`` subclass (adding new exception types here
# as needed) so users can reliably catch ``RocketPyError`` and its subclasses.


class RocketPyError(Exception):
    """Base class for all RocketPy exceptions."""


class InvalidParameterError(RocketPyError, ValueError):
    """Raised when a function, method, or constructor parameter receives an
    invalid value or violates geometric/physical constraints (e.g., negative
    mass, out-of-bounds coordinates, or unsupported options)."""


class InvalidInertiaError(RocketPyError, ValueError):
    """Raised when the inertia tuple/list does not have the expected length."""


class UnstableRocketWarning(UserWarning):
    """Issued when the rocket's static margin is negative at motor ignition,
    indicating an aerodynamically unstable configuration.

    Not issued when the rocket has any ``GenericSurface`` aerodynamic
    surfaces, since their lift coefficient derivative is not accounted for
    in the center of pressure calculation, making the static margin
    unreliable for this check in that case.
    """
