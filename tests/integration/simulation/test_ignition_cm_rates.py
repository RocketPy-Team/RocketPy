import numpy as np
import pytest

from rocketpy import Flight, HybridMotor


@pytest.mark.parametrize("step", [1e-6, 2e-6, 4e-6])
def test_ignition_rhs_uses_branch_local_cm_rates(
    calisto_hybrid_modded, example_plain_env, monkeypatch, step
):
    """Use production CM rates in the real flight RHS at ignition."""
    rocket = calisto_hybrid_modded
    ignition = rocket.motor.burn_start_time
    end = float(rocket.motor.solid.propellant_mass.x_array[1])
    offset = rocket.com_to_cdm_function
    mass = rocket.total_mass

    # Here, mass and offset * mass each vary linearly. Differentiating their
    # ratio gives reference rates without using the production rate helper.
    width = end - ignition
    mass_start = mass(ignition)
    mass_slope = (mass(end) - mass_start) / width
    moment_start = offset(ignition) * mass_start
    moment_slope = (offset(end) * mass(end) - moment_start) / width
    expected_rate = (
        moment_slope / mass_start - moment_start * mass_slope / mass_start**2
    )
    expected_second = (
        -2 * moment_slope * mass_slope / mass_start**2
        + 2 * moment_start * mass_slope**2 / mass_start**3
    )

    state = [0, 0, 100, 0, 0, 30, 1, 0, 0, 0, 0.1, 0.2, 0.3]
    flight = Flight(
        rocket=rocket,
        environment=example_plain_env,
        rail_length=1,
        initial_solution=[ignition + 0.01, *state],
        max_time=ignition + 0.0101,
        max_time_step=1e-4,
        verbose=False,
    )

    reference_calls = []

    def reference_rates(t):
        reference_calls.append(t)
        return offset(t), expected_rate, expected_second

    # Reference: substitute independently calculated analytical rates.
    with monkeypatch.context() as patch:
        patch.setattr(
            rocket,
            "_get_com_to_cdm_rates",
            reference_rates,
        )
        expected = np.asarray(flight.u_dot_generalized(ignition, state), dtype=float)

    # If Flight ignored the replacement, both paths could call the old code
    # and falsely agree. Check that it really requested these reference rates.
    assert reference_calls == [ignition]

    # Exercise the real production method at several sample spacings.
    production_rates = rocket._get_com_to_cdm_rates
    production_calls = []

    def checked_production_rates(t):
        production_calls.append(t)
        return production_rates(t, dx=step)

    with monkeypatch.context() as patch:
        patch.setattr(
            rocket,
            "_get_com_to_cdm_rates",
            checked_production_rates,
        )
        actual = np.asarray(flight.u_dot_generalized(ignition, state), dtype=float)

    assert production_calls == [ignition]
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("max_time_step", [1e-3, 5e-4])
def test_airborne_flight_crosses_delayed_ignition(
    calisto_motorless, oxidizer_tank, example_plain_env, max_time_step
):
    """Integrate through ignition off the rail without an acceleration spike."""
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
    motor.add_tank(oxidizer_tank, position=0.3)
    rocket = calisto_motorless
    rocket.add_motor(motor, position=-1.373)
    state = [0, 0, 100, 0, 0, 30, 1, 0, 0, 0, 0, 0, 0]

    flight = Flight(
        rocket=rocket,
        environment=example_plain_env,
        rail_length=1,
        initial_solution=[0.99, *state],
        max_time=1.02,
        max_time_step=max_time_step,
        verbose=False,
    )
    solution = np.asarray(flight.solution, dtype=float)

    assert np.all(np.isfinite(solution))
    assert np.any(np.isclose(solution[:, 0], 1.0, rtol=0, atol=1e-12))
    assert solution[-1, 0] == pytest.approx(1.02)
    assert solution[-1, 6] > state[5]
    # Thrust is about 2 kN on a roughly 51 kg rocket. A 100 m/s² bound
    # allows normal dynamics but rejects the former ~37000 m/s² spike.
    assert flight.max_acceleration < 100
