"""DATCOM damping and load coupling without modifying ordinary flight models."""

import math
from functools import wraps

from rocketpy.mathutils.vector_matrix import Matrix, Vector

from .datcom_uncertainty import DatcomUncertaintyMixin


def datcom_derivative(method):
    """Wrap an RHS with a local implicit incidence-rate solve when DATCOM is active."""

    @wraps(method)
    def evaluate(self, time, state, post_processing=False):
        if self.rocket._datcom_model is None or getattr(
            self, "_datcom_evaluating", False
        ):
            return method(self, time, state, post_processing)

        def derivative(t, u, post):
            return method(self, t, u, post)

        return self._evaluate_datcom_derivative(
            derivative, time, state, post_processing
        )

    return evaluate


class DatcomFlightMixin(DatcomUncertaintyMixin):
    """Time-dependent loads and deterministic unsteady aerodynamic coupling."""

    def _datcom_surface_loads(self, surface, time, *args):
        """Supply time and the instantaneous incidence rate to DATCOM only."""
        if surface is self.rocket._datcom_model:
            model = getattr(self, "_datcom_load_model", surface)
            loads = model.compute_forces_and_moments_at_time(
                time, getattr(self, "_datcom_alpha_dot", 0.0), *args
            )
            if any(
                brake.override_rocket_drag and brake.deployment_level > 0
                for brake in self.rocket.air_brakes
            ):
                # The air-brake loop already supplied the overriding axial load.
                # Keep DATCOM's lateral forces and intrinsic moments.
                return (*loads[:2], 0.0, *loads[3:])
            return loads
        return surface.compute_forces_and_moments(*args)

    def _surface_cp_at_time(self, surface, time):
        """Aerodynamic reference relative to dry CG in body axes, in meters."""
        if surface is self.rocket._datcom_model:
            return surface.reference_offset(time)
        return self.rocket.surfaces_cp_to_cdm[surface]

    def _evaluate_datcom_derivative(self, derivative, time, state, post_processing):
        """Solve the affine CNAD/CMAD acceleration coupling within one RHS call.

        The rate is never differenced across adaptive ODE evaluations. Probe
        evaluations do not append post-processing records. Scratch state belongs
        to this Flight rather than its shared rocket, and is cleared on errors.
        """
        self._datcom_evaluating = True
        self._datcom_alpha_dot = 0.0
        try:
            base = derivative(time, state, False)
            intercept = self._datcom_incidence_rate(time, state, base)
            self._datcom_alpha_dot = 1.0
            probe = derivative(time, state, False)
            response = self._datcom_incidence_rate(time, state, probe) - intercept
            denominator = 1 - response
            if not math.isfinite(denominator) or abs(denominator) < 1e-10:
                raise RuntimeError("Singular DATCOM alpha-dot aerodynamic coupling")
            self._datcom_alpha_dot = intercept / denominator
            if not math.isfinite(self._datcom_alpha_dot):
                raise RuntimeError("Nonfinite DATCOM alpha-dot aerodynamic coupling")
            return derivative(time, state, post_processing)
        finally:
            self._datcom_alpha_dot = 0.0
            self._datcom_evaluating = False

    def _datcom_incidence_rate(self, time, state, derivative):
        """Local alpha rate (rad/s), including rotation, acceleration and shear."""
        rotation = Matrix.transformation(state[6:10])
        inverse = rotation.transpose
        omega = Vector(state[10:13])
        omega_dot = Vector(derivative[10:13])
        reference = self.rocket._datcom_model.reference_offset(time)
        z = state[2] + (rotation @ reference).z
        wind = Vector([self.env.wind_velocity_x(z), self.env.wind_velocity_y(z), 0])
        relative = inverse @ (Vector(state[3:6]) - wind)
        velocity = relative + (omega ^ reference)
        speed = abs(velocity)
        axial = abs(velocity.z)
        denominator = axial**2 + velocity.y**2
        if speed < 1 or denominator < 1e-16:
            return 0.0
        z_dot = state[5] + (rotation @ (omega ^ reference)).z
        wind_dot = Vector(
            [
                self.env.wind_velocity_x.differentiate(z, dx=0.1) * z_dot,
                self.env.wind_velocity_y.differentiate(z, dx=0.1) * z_dot,
                0,
            ]
        )
        acceleration = (
            inverse @ (Vector(derivative[3:6]) - wind_dot)
            - (omega ^ relative)
            + (omega_dot ^ reference)
        )
        axial_dot = math.copysign(1.0, velocity.z) * acceleration.z
        return (axial * acceleration.y - velocity.y * axial_dot) / denominator

    def _datcom_rail_force(self, time, velocity, speed, mach, rho, reynolds, fallback):
        """Constrained axial load uses body-axis CA rather than stability-axis CD."""
        if self.rocket._datcom_model is None:
            return fallback
        model = getattr(self, "_datcom_load_model", self.rocket._datcom_model)
        loads = model.compute_forces_and_moments_at_time(
            time,
            0.0,
            velocity,
            speed,
            mach,
            rho,
            Vector.zeros(),
            Vector.zeros(),
            reynolds,
        )
        return loads[2]

    def _datcom_moment_at_cm(self, time, forces, moments):
        """Transfer a physical dry-CG moment to the current CG for legacy EOM."""
        if self.rocket._datcom_model is None:
            return moments
        cm_to_dry = Vector([0, 0, self.rocket.com_to_cdm_function(time)])
        return Vector(moments) + (cm_to_dry ^ Vector(forces))

    def _datcom_angular_acceleration(self, inertia, T20, T21, r_cm, forces, moments):
        """Transfer aerodynamic torque explicitly within generalized EOM conventions."""
        if self.rocket._datcom_model is None:
            return inertia.inverse @ (T21 + (T20 ^ r_cm))
        aero_force, aero_moment = Vector(forces), Vector(moments)
        moment_at_cm = aero_moment + (r_cm ^ aero_force)
        return inertia.inverse @ (
            T21 - aero_moment + ((T20 - aero_force) ^ r_cm) + moment_at_cm
        )
