"""Equation-level checks for Flight.u_dot_generalized.

These drive the real right-hand side with a stubbed rigid body so that every
term except the one under test is exactly zero, which makes the expected
answer a closed-form Newton-Euler result rather than a recorded trajectory.
"""

import numpy as np
import pytest

from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.rocket.rocket import Rocket
from rocketpy.simulation.flight import Flight

MASS = 40.0
COM_TO_CDM = 0.30
NOZZLE_TO_CDM = 1.25
INERTIA = [[60.0, 0.0, 0.0], [0.0, 60.0, 0.0], [0.0, 0.0, 2.0]]


class _Scalar:
    """Constant Function stand-in with a settable first derivative."""

    def __init__(self, value, derivative=0.0):
        self.value = float(value)
        self.derivative = float(derivative)

    def get_value_opt(self, _t):
        return self.value

    def differentiate_complex_step(self, _t):
        return self.derivative

    def differentiate(self, _t, order=1):
        return self.derivative if order == 1 else 0.0


class _Surface:
    """Aerodynamic surface returning a prescribed force and moment."""

    reference_length = 1.0

    def __init__(self, forces_and_moments):
        self.forces_and_moments = forces_and_moments

    def compute_forces_and_moments(self, *_args, **_kwargs):
        return self.forces_and_moments


class _Motor:
    burn_start_time = 0.0
    burn_out_time = 0.0
    nozzle_radius = 0.05
    thrust = _Scalar(0.0)

    def pressure_thrust(self, _pressure):
        return 0.0


class _Rocket:
    area = 1.0
    radius = 0.5
    cp_eccentricity_x = cp_eccentricity_y = 0.0
    thrust_eccentricity_x = thrust_eccentricity_y = 0.0
    air_brakes = []

    def __init__(
        self,
        mass,
        com_to_cdm,
        inertia,
        cp,
        forces_and_moments,
        mass_flow_rate=0.0,
        mass_flow_rate_dot=0.0,
        nozzle_to_cdm=NOZZLE_TO_CDM,
    ):
        self.motor = _Motor()
        self.total_mass = _Scalar(mass)
        self.total_mass_flow_rate = _Scalar(mass_flow_rate, mass_flow_rate_dot)
        self.com_to_cdm_function = _Scalar(com_to_cdm)
        self.nozzle_to_cdm = nozzle_to_cdm
        self.nozzle_gyration_tensor = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self._inertia = Matrix(inertia)
        surface = _Surface(forces_and_moments)
        self.aerodynamic_surfaces = [(surface, None)]
        self.surfaces_cp_to_cdm = {surface: Vector(cp)}

    def get_inertia_tensor_at_time(self, _t):
        return self._inertia

    def get_inertia_tensor_derivative_at_time(self, _t):
        return Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    def power_off_drag_7d(self, *_args):
        return 0.0

    def power_on_drag_7d(self, *_args):
        return 0.0


class _Environment:
    earth_rotation_vector = [0.0, 0.0, 0.0]
    density = _Scalar(0.0)
    wind_velocity_x = _Scalar(0.0)
    wind_velocity_y = _Scalar(0.0)
    speed_of_sound = _Scalar(340.0)
    dynamic_viscosity = _Scalar(1.8e-5)
    pressure = _Scalar(0.0)
    gravity = _Scalar(0.0)


def _derivatives(**kwargs):
    """Return (linear acceleration, angular acceleration) in the body frame."""
    flight = Flight.__new__(Flight)
    flight.rocket = _Rocket(**kwargs)
    flight.env = _Environment()
    state = [0.0, 0.0, 1000.0, 0.0, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    u_dot = flight.u_dot_generalized(0.0, state)
    return np.array(u_dot[3:6]), np.array(u_dot[10:13])


def _center_of_mass_inertia(mass, com_to_cdm, inertia):
    """Inertia about the true center of mass, by the parallel axis theorem."""
    lever = np.array([0.0, 0.0, -com_to_cdm])
    return np.array(inertia) - mass * (
        lever @ lever * np.eye(3) - np.outer(lever, lever)
    )


def _skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


@pytest.mark.parametrize("cp", [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.4]])
def test_angular_acceleration_uses_the_lever_arm_from_the_center_of_mass(cp):
    """A force applied at cp must turn the rocket about its center of mass."""
    force = np.array([100.0, 0.0, 0.0])
    moment = np.cross(np.array(cp), force)
    _, angular = _derivatives(
        mass=MASS,
        com_to_cdm=COM_TO_CDM,
        inertia=INERTIA,
        cp=cp,
        forces_and_moments=(*force, *moment),
    )
    inertia_cm = _center_of_mass_inertia(MASS, COM_TO_CDM, INERTIA)
    lever = np.array(cp) - np.array([0.0, 0.0, -COM_TO_CDM])
    expected = np.linalg.solve(inertia_cm, np.cross(lever, force))
    np.testing.assert_allclose(angular, expected, rtol=1e-12, atol=1e-12)


def test_axial_force_through_the_axis_produces_no_rotation():
    """Control: an on-axis force can never generate an angular acceleration."""
    _, angular = _derivatives(
        mass=MASS,
        com_to_cdm=COM_TO_CDM,
        inertia=INERTIA,
        cp=[0.0, 0.0, -1.0],
        forces_and_moments=(0.0, 0.0, 500.0, 0.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(angular, np.zeros(3), atol=1e-12)


def test_reference_origin_invariance():
    """The same rocket described about its CDM or about its CM must agree."""
    cp = np.array([0.0, 0.0, -1.0])
    force = np.array([100.0, 0.0, 0.0])
    cdm_to_cm = np.array([0.0, 0.0, -COM_TO_CDM])
    inertia_cm = _center_of_mass_inertia(MASS, COM_TO_CDM, INERTIA)

    about_cdm = {
        "mass": MASS,
        "com_to_cdm": COM_TO_CDM,
        "inertia": INERTIA,
        "cp": list(cp),
        "forces_and_moments": (*force, *np.cross(cp, force)),
    }
    cp_from_cm = cp - cdm_to_cm
    about_cm = {
        "mass": MASS,
        "com_to_cdm": 0.0,
        "inertia": inertia_cm.tolist(),
        "cp": list(cp_from_cm),
        "forces_and_moments": (*force, *np.cross(cp_from_cm, force)),
        "nozzle_to_cdm": NOZZLE_TO_CDM - COM_TO_CDM,
    }
    linear_cdm, angular_cdm = _derivatives(**about_cdm)
    linear_cm, angular_cm = _derivatives(**about_cm)

    np.testing.assert_allclose(angular_cdm, angular_cm, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        linear_cdm + np.cross(angular_cdm, cdm_to_cm),
        linear_cm,
        rtol=1e-12,
        atol=1e-12,
    )


def test_satisfies_the_documented_block_system():
    """The returned pair must solve M*v + [m r]^T*w = T20 and I*w + [m r]*v = T21."""
    cp = [0.0, 0.0, -1.0]
    force = np.array([100.0, -40.0, 250.0])
    moment = np.cross(np.array(cp), force)
    linear, angular = _derivatives(
        mass=MASS,
        com_to_cdm=COM_TO_CDM,
        inertia=INERTIA,
        cp=cp,
        forces_and_moments=(*force, *moment),
    )
    coupling = MASS * _skew(np.array([0.0, 0.0, -COM_TO_CDM]))
    np.testing.assert_allclose(
        MASS * linear + coupling.T @ angular, force, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        np.array(INERTIA) @ angular + coupling @ linear, moment, rtol=1e-11, atol=1e-11
    )


def test_nozzle_offset_enters_as_the_center_of_mass_to_nozzle_vector():
    """The mddot*(r_NOZ - r_CM) term must push toward the nozzle, that is aft."""
    mass_flow_rate_dot = 3.0
    linear, angular = _derivatives(
        mass=MASS,
        com_to_cdm=COM_TO_CDM,
        inertia=INERTIA,
        cp=[0.0, 0.0, 0.0],
        forces_and_moments=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        mass_flow_rate=0.0,
        mass_flow_rate_dot=mass_flow_rate_dot,
    )
    expected_z = -mass_flow_rate_dot * (NOZZLE_TO_CDM - COM_TO_CDM) / MASS
    np.testing.assert_allclose(angular, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(linear, [0.0, 0.0, expected_z], rtol=1e-12, atol=1e-12)


class _NozzleStub:
    def __init__(self, nozzle_radius, nozzle_to_cdm):
        self.nozzle_to_cdm = nozzle_to_cdm
        self.motor = type("motor", (), {"nozzle_radius": nozzle_radius})


@pytest.mark.parametrize("nozzle_to_cdm", [0.0, 0.75, 1.35])
def test_nozzle_gyration_tensor_matches_the_disk_second_moment(nozzle_to_cdm):
    """S is the exit disk second moment per unit area about the CDM."""
    radius = 0.033
    tensor = Rocket.evaluate_nozzle_gyration_tensor(_NozzleStub(radius, nozzle_to_cdm))
    lateral = radius**2 / 4 + nozzle_to_cdm**2
    expected = np.diag([lateral, lateral, radius**2 / 2])
    np.testing.assert_allclose(np.array(tensor), expected, rtol=1e-12, atol=1e-15)


def test_nozzle_gyration_tensor_reduces_to_a_point_mass_flux():
    """A vanishing exit radius leaves only the offset, diag(d^2, d^2, 0)."""
    offset = 1.35
    tensor = Rocket.evaluate_nozzle_gyration_tensor(_NozzleStub(0.0, offset))
    np.testing.assert_allclose(
        np.array(tensor), np.diag([offset**2, offset**2, 0.0]), atol=1e-15
    )
