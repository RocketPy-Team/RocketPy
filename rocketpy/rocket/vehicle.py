"""Common vehicle contract used by atmospheric and orbital simulations."""

from __future__ import annotations

from abc import ABC

from rocketpy.mathutils.frame_vector import FrameVector


class Vehicle(ABC):
    """Minimal parent for objects that can be integrated by :class:`Flight`.

    The class intentionally owns no rocket-specific geometry or aerodynamics.
    It only standardizes mass and orbital interaction hooks so lightweight
    spacecraft and traditional rockets can share the same simulation engine.
    """

    _is_point_mass = False

    @staticmethod
    def _evaluate_property(value, epoch, state, direction):
        if not isinstance(direction, FrameVector):
            raise TypeError("Interaction directions must be FrameVector instances.")
        return float(value(epoch, state, direction) if callable(value) else value)

    def mass_at(self, time: float) -> float:
        """Return total vehicle mass at elapsed vehicle time."""
        mass = self.total_mass
        return float(
            mass.get_value_opt(time) if hasattr(mass, "get_value_opt") else mass(time)
        )

    def warn_if_unstable(self):
        """Compatibility hook for launch vehicles; generic vehicles do nothing."""
        return None
