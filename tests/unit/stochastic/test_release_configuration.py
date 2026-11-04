"""Regression checks for stochastic configurations used by Monte Carlo."""

import numpy as np
import pytest

from rocketpy import Flight
from rocketpy.stochastic import StochasticFlight, StochasticSolidMotor


def test_stochastic_flight_accepts_a_previous_flight(flight_calisto_robust):
    """A continuation must use the last state of the supplied flight."""
    nominal = Flight(
        flight_calisto_robust.rocket,
        flight_calisto_robust.env,
        rail_length=5.2,
        max_time=1,
    )
    stochastic = StochasticFlight(
        nominal, initial_solution=nominal, max_time=nominal.t_final + 1
    )

    continued = stochastic.create_object()

    assert continued.solution[0] == pytest.approx(nominal.solution[-1])


@pytest.mark.parametrize("bonded", [False, True])
def test_stochastic_solid_motor_preserves_grain_bonding(cesaroni_m1670, bonded):
    """A deterministic grain configuration must survive stochastic wrapping."""
    cesaroni_m1670.grains_bonded = bonded
    stochastic = StochasticSolidMotor(cesaroni_m1670)

    sampled = stochastic.create_object()

    assert sampled.grains_bonded is bonded
    assert sampled.center_of_propellant_mass(1) == pytest.approx(
        cesaroni_m1670.center_of_propellant_mass(1)
    )


def test_stochastic_flight_keeps_normal_rail_launch(flight_calisto_robust):
    """The nominal flight's generated initial state must not bypass the rail."""
    stochastic = StochasticFlight(flight_calisto_robust)

    sampled = stochastic.create_object()

    assert sampled.out_of_rail_time > sampled.t_initial
    assert np.linalg.norm(sampled.out_of_rail_state[3:6]) > 0


def test_stochastic_flight_preserves_nominal_explicit_state(flight_calisto_robust):
    """Wrapping a continuation must retain its initial free-flight state."""
    state = list(min(flight_calisto_robust.solution, key=lambda row: abs(row[0] - 1)))
    nominal = Flight(
        flight_calisto_robust.rocket,
        flight_calisto_robust.env,
        rail_length=5.2,
        initial_solution=state,
        max_time=state[0] + 1,
    )

    sampled = StochasticFlight(nominal).create_object()

    assert sampled.solution[0] == pytest.approx(state)
