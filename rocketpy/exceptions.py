"""
Custom exceptions for RocketPy library.
"""


class RocketPyException(Exception):
    """All RocketPy custom exceptions should inherit from this class."""


class UnstableRocketError(RocketPyException):
    """Raised when the rocket jas negative static margin."""

    def __init__(self, stability_margin):
        self.stability_margin = stability_margin
        self.message = (
            "Rocket is unstable. Initial Static Margin is "
            f"{stability_margin} calibers, this can lead to eternal loop simulation."
        )

    def __str__(self):
        return self.message
