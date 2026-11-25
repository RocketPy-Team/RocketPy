"""Airborne ignition/burnout checks for the sampled CM-offset approach."""

import numpy as np
import pytest

from rocketpy import CylindricalTank, Flight, Fluid, HybridMotor, MassFlowRateBasedTank


@pytest.fixture
def near_boundary_rocket(calisto_motorless, request):
    """Build a hybrid whose tank and motor endpoint differ by one ULP."""
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
def test_flight_near_equal_boundaries(near_boundary_rocket, example_plain_env):
    """Cross both endpoints in flight without the former acceleration spikes."""
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
    # About 2 kN thrust and at least 22 kg dry mass in this fixture: this
    # allows its normal flight but rejects the former thousands of m/s².
    assert flight.max_acceleration < 100
    for time in near_boundary_rocket.motor.burn_time:
        rhs = flight.u_dot_generalized(time, state)
        assert np.linalg.norm(rhs[3:6]) < 100
