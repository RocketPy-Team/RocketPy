"""Custom exceptions and warnings for RocketPy."""


class RocketPyError(Exception):
    """Base class for all RocketPy exceptions."""


class InvalidParameterError(RocketPyError, ValueError):
    """Raised when a constructor parameter has an invalid value (e.g. negative
    radius or mass)."""


class InvalidInertiaError(RocketPyError, ValueError):
    """Raised when the inertia tuple/list does not have the expected length."""


class UnstableRocketWarning(UserWarning):
    """Issued when the rocket's static margin is negative at motor ignition,
    indicating an aerodynamically unstable configuration."""
