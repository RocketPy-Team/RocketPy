"""Regressions for the boundary and compatibility review of PR 1214."""

import numpy as np
import pytest

from rocketpy import (
    CylindricalTank,
    EmptyMotor,
    Flight,
    Fluid,
    Function,
    HybridMotor,
    MassFlowRateBasedTank,
    RingClusterMotor,
)


@pytest.fixture
def near_boundary_rocket(calisto_motorless, request):
    """Build a hybrid whose tank and motor endpoint differ only by rounding."""
    endpoint, direction = request.param
    times = [1.0, 11.0]
    times[endpoint] = np.nextafter(times[endpoint], direction)
    tank = MassFlowRateBasedTank(
        name="review tank",
        geometry=CylindricalTank(0.1, 1),
        flux_time=times,
        liquid=Fluid("liquid", 1000),
        gas=Fluid("gas", 1),
        initial_liquid_mass=3,
        initial_gas_mass=0.01,
        liquid_mass_flow_rate_in=0,
        liquid_mass_flow_rate_out=0.1,
        gas_mass_flow_rate_in=0,
        gas_mass_flow_rate_out=0,
    )
    motor = HybridMotor(
        thrust_source=lambda t: 2000 - 100 * (t - 1),
        burn_time=(1, 11),
        dry_mass=8,
        dry_inertia=(4, 4, 0.1),
        center_of_dry_mass_position=0,
        grain_density=1700,
        grain_number=4,
        grain_initial_height=0.1,
        grain_separation=0,
        grain_initial_inner_radius=0.04,
        grain_outer_radius=0.1,
        grains_center_of_mass_position=-0.1,
        nozzle_position=-0.4,
        nozzle_radius=0.07,
    )
    motor.add_tank(tank, position=0.3)
    calisto_motorless.add_motor(motor, position=-1.373)
    return calisto_motorless


@pytest.mark.parametrize(
    "near_boundary_rocket",
    [(0, -np.inf), (0, np.inf), (1, -np.inf), (1, np.inf)],
    indirect=True,
)
def test_com_to_cdm_rates_near_equal_boundaries(near_boundary_rocket):
    """One-ULP tank/motor differences must not abort derivative evaluation."""
    for boundary in near_boundary_rocket.motor.burn_time:
        for time in (
            np.nextafter(boundary, -np.inf),
            boundary,
            np.nextafter(boundary, np.inf),
        ):
            assert np.isfinite(near_boundary_rocket._get_com_to_cdm_rates(time)).all()


@pytest.mark.parametrize(
    "near_boundary_rocket",
    [(0, -np.inf), (0, np.inf), (1, -np.inf), (1, np.inf)],
    indirect=True,
)
def test_flight_near_equal_boundaries(near_boundary_rocket, example_plain_env):
    """A real airborne flight must finish after crossing ignition and burnout."""
    state = [0, 0, 100, 0, 0, 30, 1, 0, 0, 0, 0, 0, 0]
    flight = Flight(
        rocket=near_boundary_rocket,
        environment=example_plain_env,
        rail_length=1,
        initial_solution=[0.99, *state],
        max_time=11.5,
        verbose=False,
    )
    assert flight.solution[-1][0] == pytest.approx(11.5)
    assert np.isfinite(flight.solution).all()
    # Roughly 2 kN of thrust and at least 22 kg dry mass; this rejects the
    # former numerical spikes while allowing the normal powered trajectory.
    assert flight.max_acceleration < 100


def test_com_to_cdm_function_keeps_sampled_source(calisto):
    """Public sampled arrays and interpolation retain the pre-PR behavior."""
    expected = (
        -1
        * (
            (
                calisto.center_of_propellant_position
                - calisto.center_of_dry_mass_position
            )
            * calisto._csys
        )
        * calisto.motor.propellant_mass
        / calisto.total_mass
    )
    actual = calisto.com_to_cdm_function
    assert actual.get_source_type() == expected.get_source_type()
    np.testing.assert_array_equal(actual.x_array, expected.x_array)
    np.testing.assert_array_equal(actual.source, expected.source)


def test_com_to_cdm_boundaries_custom_motor_refresh(calisto_motorless):
    """Use a custom motor's declared callable knots and refresh on reattachment."""

    class CallableMotor(EmptyMotor):
        """An input with a breakpoint that is not stored in an x_array."""

        knot = 0.4

        def get_mass_property_time_knots(self):
            return super().get_mass_property_time_knots() + [self.knot]

    motor = CallableMotor()
    rocket = calisto_motorless
    rocket.add_motor(motor, position=0)
    assert 0.4 in rocket._com_to_cdm_rate_boundaries
    motor.knot = 0.6
    rocket.add_motor(motor, position=0)
    assert 0.6 in rocket._com_to_cdm_rate_boundaries
    assert 0.4 not in rocket._com_to_cdm_rate_boundaries


def test_com_to_cdm_boundaries_cluster(calisto_motorless, hybrid_motor):
    """A cluster reports the component's tank endpoints and internal data knots."""
    cluster = RingClusterMotor(hybrid_motor, number=3, radius=0.1)
    rocket = calisto_motorless
    rocket.add_motor(cluster, position=0)
    knots = rocket._com_to_cdm_rate_boundaries.ravel()
    tank = hybrid_motor.positioned_tanks[0]["tank"]
    assert all(time in knots for time in tank.flux_time)
    assert any(time in knots for time in tank.fluid_mass.x_array[1:-1])


@pytest.mark.parametrize("time", [0.0, 1e-12, 2e-12])
def test_com_to_cdm_rates_coincident_callable_knots(
    calisto_motorless, monkeypatch, time
):
    """A merged cluster chooses the right branch, not the short artificial gap."""
    rocket = calisto_motorless
    rocket.dry_mass = 6.0
    rocket.center_of_dry_mass_position = 0.0
    rocket.center_of_propellant_position = Function(-3.0)
    rocket.motor.propellant_mass = Function(
        lambda t: 4.0 if t.real < 2e-12 else 4.0 - 2 * (t - 2e-12)
    )
    rocket.total_mass = Function(lambda t: 6 + rocket.motor.propellant_mass(t))
    monkeypatch.setattr(
        rocket.motor, "get_mass_property_time_knots", lambda: [0, 2e-12, 1]
    )
    rocket.evaluate_com_to_cdm_function()
    actual = rocket._get_com_to_cdm_rates(time)
    assert actual == pytest.approx((1.2, -0.36, -0.144), rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("time", [0.0, 1e-9, 2e-9, 1e6])
def test_com_to_cdm_samples_distinct_at_small_spacing(
    calisto_motorless, monkeypatch, time
):
    """Even an undersized requested dx gives distinct samples in one interval."""
    rocket = calisto_motorless
    monkeypatch.setattr(
        rocket.motor, "get_mass_property_time_knots", lambda: [0, 2e-9, 2]
    )
    rocket.evaluate_com_to_cdm_function()
    anchor, _, times, _ = rocket._com_to_cdm_samples(time, 1e-320)
    assert len(set((anchor, *times))) == 3
    assert all(np.isfinite((anchor, *times)))
