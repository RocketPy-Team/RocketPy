"""Lightweight spacecraft vehicle."""

from __future__ import annotations

import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.mathutils.vector_matrix import Matrix
from rocketpy.motors.empty_motor import EmptyMotor
from rocketpy.rocket.components import Components
from rocketpy.rocket.vehicle import Vehicle


class Spacecraft(Vehicle):
    """A minimal vehicle for coast/orbital phases.

    Parameters are deliberately limited to properties used by the shared
    Earth-centered dynamics. A spacecraft can still be replaced by a full
    :class:`Rocket` whenever propulsion or launch aerodynamics are needed.
    """

    def __init__(
        self,
        mass,
        area=0.0,
        inertia=(1.0, 1.0, 1.0),
        drag_coefficient=None,
        drag_area=None,
        radiation_coefficient=0.0,
        radiation_area=None,
        name="Spacecraft",
    ):
        if not callable(mass) and float(mass) <= 0:
            raise ValueError("Spacecraft mass must be positive.")
        if float(area) < 0:
            raise ValueError("Spacecraft area must be non-negative.")
        inertia = (*inertia, 0.0, 0.0, 0.0) if len(inertia) == 3 else tuple(inertia)
        if len(inertia) != 6:
            raise ValueError("inertia must contain 3 or 6 components.")

        self.name = name
        self.mass = mass
        self.total_mass = Function(mass, "Time (s)", "Total Mass (kg)")
        self.motor = EmptyMotor()
        self.area = float(area)
        self.radius = np.sqrt(self.area / np.pi) if self.area else 0.0
        self.orbital_drag_coefficient = drag_coefficient
        if drag_area is not None and not callable(drag_area):
            if float(drag_area) < 0:
                raise ValueError("Spacecraft drag area must be non-negative.")
        if radiation_area is not None and not callable(radiation_area):
            if float(radiation_area) < 0:
                raise ValueError("Spacecraft radiation area must be non-negative.")
        self.orbital_projected_area = self.area if drag_area is None else drag_area
        self.radiation_coefficient = radiation_coefficient
        self.radiation_projected_area = (
            self.area if radiation_area is None else radiation_area
        )
        self._inertia = tuple(float(value) for value in inertia)

        # The shared Flight event machinery expects these collections. Keeping
        # them empty avoids teaching Flight about individual vehicle subclasses.
        self.parachutes = []
        self._controllers = []
        self.air_brakes = []
        self.sensors = Components()
        self.sensors_by_name = {}
        self.rail_buttons = Components()

    def get_inertia_tensor_at_time(self, time):
        del time
        i11, i22, i33, i12, i13, i23 = self._inertia
        return Matrix([[i11, i12, i13], [i12, i22, i23], [i13, i23, i33]])

    def evaluate_orbital_drag_coefficient(self, epoch, state, direction):
        if self.orbital_drag_coefficient is None:
            return 0.0
        return self._evaluate_property(
            self.orbital_drag_coefficient, epoch, state, direction
        )

    def evaluate_orbital_drag_area(self, epoch, state, direction):
        return self._evaluate_property(
            self.orbital_projected_area, epoch, state, direction
        )

    def evaluate_radiation_coefficient(self, epoch, state, direction):
        return self._evaluate_property(
            self.radiation_coefficient, epoch, state, direction
        )

    def evaluate_radiation_area(self, epoch, state, direction):
        return self._evaluate_property(
            self.radiation_projected_area, epoch, state, direction
        )
