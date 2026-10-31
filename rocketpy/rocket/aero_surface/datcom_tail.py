"""Ogive geometry for whole-vehicle DATCOM, without component aerodynamics."""

import numpy as np
from scipy.integrate import quad

from rocketpy.mathutils import Function
from rocketpy.rocket.aero_surface.tail import Tail


class DatcomTail(Tail):
    """A tangent-ogive afterbody usable only with DATCOM aerodynamics.

    The inherited component lift and CP are disabled placeholders. Native
    DATCOM supplies the whole vehicle's forces, moments and pressure center.
    ``kind`` is read-only; construct a new surface to change its profile.
    """

    @property
    def kind(self):
        return "ogive"

    def radius_at(self, x):
        """Radius (m) at distance x (m) aft of the upstream cylinder joint."""
        x = np.asarray(x)
        change = self.bottom_radius - self.top_radius
        if change == 0:
            return np.full_like(x, self.top_radius, dtype=float)
        rho = (self.length**2 + change**2) / (2 * abs(change))
        sagitta = x**2 / (rho + np.sqrt(np.maximum(0, rho**2 - x**2)))
        return self.top_radius + np.sign(change) * sagitta

    def evaluate_geometrical_parameters(self):
        change = self.bottom_radius - self.top_radius
        if self.length <= 0 or abs(change) > self.length:
            raise ValueError(
                "Ogive tail needs positive length and radius change <= length"
            )
        if change == 0:
            self.slant_length = self.length
            self.surface_area = 2 * np.pi * self.top_radius * self.length
        else:
            rho = (self.length**2 + change**2) / (2 * abs(change))
            angle = np.arcsin(self.length / rho)
            self.slant_length = rho * angle
            self.surface_area = quad(
                lambda theta: 2 * np.pi * self.radius_at(rho * np.sin(theta)) * rho,
                0,
                angle,
            )[0]
        self.evaluate_shape()

    def evaluate_shape(self):
        x = np.linspace(0, self.length, 100)
        self.shape_vec = [x, self.radius_at(x)]

    def evaluate_lift_coefficient(self):
        self.clalpha = Function(0, "Mach", "DATCOM-only component lift slope")
        self.cl = Function(
            lambda alpha, mach: 0, ["Alpha (rad)", "Mach"], "DATCOM-only component lift"
        )

    def evaluate_center_of_pressure(self):
        self.cpx = self.cpy = self.cpz = 0
        self.cp = (0, 0, 0)

    def compute_forces_and_moments(self, *args, **kwargs):
        raise ValueError("Ogive tails require whole-vehicle DATCOM aerodynamics")
