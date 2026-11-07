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


def test_stochastic_flight_keeps_nominal_terminate_on_apogee(flight_calisto_robust):
    """An omitted ``terminate_on_apogee`` must not reset the nominal flight's."""
    nominal = Flight(
        flight_calisto_robust.rocket,
        flight_calisto_robust.env,
        rail_length=5.2,
        terminate_on_apogee=True,
        max_time=1,
    )

    kept = StochasticFlight(nominal)
    overridden = StochasticFlight(nominal, terminate_on_apogee=False)

    assert kept.create_object().terminate_on_apogee is True
    assert overridden.create_object().terminate_on_apogee is False


def test_stochastic_flight_keeps_nominal_tip_off_phase(flight_calisto_robust):
    """The opt-in single rail button phase must survive stochastic wrapping."""
    nominal = Flight(
        flight_calisto_robust.rocket,
        flight_calisto_robust.env,
        rail_length=5.2,
        use_udot_rail2=True,
        max_time=1,
    )

    sampled = StochasticFlight(nominal).create_object()

    assert sampled.use_udot_rail2 is True


@pytest.mark.parametrize("scalar", [np.float32, np.float64, np.int64])
def test_stochastic_flight_accepts_numpy_scalars_in_initial_solution(
    flight_calisto_robust, scalar
):
    """A state sliced out of a NumPy array is a state, not a type error."""
    state = [scalar(value) for value in flight_calisto_robust.solution[0]]

    stochastic = StochasticFlight(flight_calisto_robust, initial_solution=state)

    assert stochastic.initial_solution == state


@pytest.mark.parametrize("flag", [True, np.bool_(True)])
def test_stochastic_flight_rejects_booleans_in_initial_solution(
    flight_calisto_robust, flag
):
    """A boolean compares equal to a number and is still not one."""
    state = list(flight_calisto_robust.solution[0])
    state[3] = flag

    with pytest.raises(AssertionError, match="tuple of numbers"):
        StochasticFlight(flight_calisto_robust, initial_solution=state)


def test_stochastic_flight_rejects_a_non_boolean_terminate_on_apogee(
    flight_calisto_robust,
):
    """``terminate_on_apogee`` has to be a real boolean when it is given."""
    with pytest.raises(AssertionError, match="terminate_on_apogee"):
        StochasticFlight(flight_calisto_robust, terminate_on_apogee=1)
