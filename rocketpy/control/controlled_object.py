from typing import Protocol, runtime_checkable


@runtime_checkable
class ControlledObject(Protocol):
    """Structural type for objects that a :class:`Controller` can drive.

    Any object exposing this interface can be handed to a controller as a
    controlled object: the controller records its ``control_state`` after
    every execution (see ``Controller.control_history``) and restores it to
    ``initial_control_state`` at the start of each simulation.

    The interface is intentionally not tied to aerodynamic surfaces so that
    non-surface actuators (e.g. a future thrust-vector-control gimbal on a
    motor mount) can conform to it. ``ControllableGenericSurface`` (and thus
    ``AirBrakes``) is the reference implementation.

    Notes
    -----
    Controllers duck-type on this interface (they check for the presence of
    ``control_state`` and ``_reset``) rather than requiring inheritance;
    this class exists for documentation, typing, and optional ``isinstance``
    checks.
    """

    control_variables: list
    """Ordered names of the control axes."""

    control_state: dict
    """Current value of each control variable."""

    initial_control_state: dict
    """Control values to restore at the start of each simulation."""

    def set_control(self, name, value):
        """Set the current value of a control variable (applying clamping)."""

    def get_control(self, name):
        """Return the current value of a control variable."""

    def _reset(self):
        """Restore ``control_state`` to ``initial_control_state``."""
