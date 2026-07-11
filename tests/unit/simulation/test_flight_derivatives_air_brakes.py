"""Unit tests for the air-brakes handling in the flight derivatives, driven
without constructing a Flight: body-drag suppression under
override_rocket_drag and force equivalence of the surface-loop path with the
legacy drag-only formula."""

from types import SimpleNamespace

import pytest

from rocketpy import Function
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.simulation.helpers.flight_derivatives import _aerodynamic_drag_force

DRAG_CURVE = "data/rockets/calisto/air_brakes_cd.csv"

RHO = 1.10
SPEED = 170.0
MACH = 0.5
REYNOLDS = 1e6
OMEGA = (0.0, 0.0, 0.0)
TIME_AFTER_BURNOUT = 1000.0


def body_drag(rocket):
    flight = SimpleNamespace(rocket=rocket)
    return _aerodynamic_drag_force(
        flight, TIME_AFTER_BURNOUT, RHO, SPEED, 0, 0, MACH, REYNOLDS, OMEGA
    )


@pytest.fixture
def rocket_with_air_brakes(calisto_motorless):
    air_brakes = calisto_motorless.add_air_brakes(
        drag_coefficient_curve=DRAG_CURVE,
        controller_function=lambda **kwargs: None,
        sampling_rate=10,
    )
    return calisto_motorless, air_brakes


class TestBodyDragSuppression:
    def test_body_drag_unchanged_when_retracted(self, rocket_with_air_brakes):
        rocket, air_brakes = rocket_with_air_brakes
        air_brakes.deployment_level = 0
        reference = body_drag(rocket)
        assert reference < 0

        air_brakes.override_rocket_drag = True
        assert body_drag(rocket) == reference

    def test_body_drag_unchanged_when_deployed_without_override(
        self, rocket_with_air_brakes
    ):
        rocket, air_brakes = rocket_with_air_brakes
        air_brakes.deployment_level = 0
        reference = body_drag(rocket)
        air_brakes.deployment_level = 0.8
        assert body_drag(rocket) == reference

    def test_body_drag_suppressed_when_deployed_with_override(
        self, rocket_with_air_brakes
    ):
        rocket, air_brakes = rocket_with_air_brakes
        air_brakes.override_rocket_drag = True
        air_brakes.deployment_level = 0.8
        assert body_drag(rocket) == 0.0


class TestSurfaceLoopForceEquivalence:
    def compute_surface_forces(self, rocket, air_brakes):
        cp_to_cdm = rocket.surfaces_cp_to_cdm[air_brakes]
        return air_brakes.compute_forces_and_moments(
            Vector([0, 0, -SPEED]),
            SPEED,
            MACH,
            RHO,
            cp_to_cdm,
            Vector([0, 0, 0]),
            Function(lambda z: RHO),
            Function(lambda z: 1.7e-5),
            1000.0,
        )

    def test_axial_force_matches_legacy_formula(self, rocket_with_air_brakes):
        rocket, air_brakes = rocket_with_air_brakes
        air_brakes.deployment_level = 0.5
        fx, fy, fz, m1, m2, m3 = self.compute_surface_forces(rocket, air_brakes)

        cd = air_brakes.drag_coefficient.get_value_opt(0.5, MACH)
        legacy_force = -0.5 * RHO * SPEED**2 * air_brakes.reference_area * cd
        assert fz == pytest.approx(legacy_force)
        # zero incidence, pinned cp: no lateral force, no moments
        assert (fx, fy, m1, m2, m3) == (0, 0, 0, 0, 0)

    def test_zero_force_when_retracted(self, rocket_with_air_brakes):
        rocket, air_brakes = rocket_with_air_brakes
        air_brakes.deployment_level = 0
        forces = self.compute_surface_forces(rocket, air_brakes)
        assert tuple(forces) == (0, 0, 0, 0, 0, 0)
