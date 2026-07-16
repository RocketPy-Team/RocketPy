"""Tests for the internal _Dynamics wrapper and its bound form.

These tests use stub flights and stub derivatives, so they never run a Flight.
"""

import pytest

from rocketpy.simulation.helpers.dynamics import (
    DYNAMICS_REGISTRY,
    PARACHUTE_DYNAMICS,
    RAIL_DYNAMICS,
    SIX_DOF_DYNAMICS,
    _Dynamics,
)
from rocketpy.simulation.solution import (
    CANONICAL_SCHEMA,
    PARACHUTE_3T_SCHEMA,
    StateSchema,
)


class StubFlight:
    """Minimal stand-in for a Flight, recording derivative calls."""

    def __init__(self):
        self.calls = []
        self.atol = 6 * [1e-3] + 4 * [1e-6] + 3 * [1e-3]


def stub_derivative(flight, t, u, post_processing=False):
    flight.calls.append((t, list(u), post_processing))
    return [value * 2 for value in u]


def test_registry_keys_and_schemas():
    assert set(DYNAMICS_REGISTRY) == {
        "rail",
        "solid_propulsion",
        "six_dof",
        "three_dof",
        "parachute",
    }
    assert SIX_DOF_DYNAMICS.schema is CANONICAL_SCHEMA
    assert RAIL_DYNAMICS.schema is CANONICAL_SCHEMA
    # The parachute descent integrates only position and velocity.
    assert PARACHUTE_DYNAMICS.schema is PARACHUTE_3T_SCHEMA
    assert PARACHUTE_DYNAMICS.derived_names == ("ax", "ay", "az", "R1", "R2", "R3")


def test_bound_dynamics_calls_free_function():
    spec = _Dynamics("stub", stub_derivative, CANONICAL_SCHEMA, ("ax",))
    flight = StubFlight()
    bound = spec.bind(flight)
    result = bound(1.5, [1.0, 2.0, 3.0], post_processing=True)
    assert result == [2.0, 4.0, 6.0]
    assert flight.calls == [(1.5, [1.0, 2.0, 3.0], True)]
    assert bound.schema is CANONICAL_SCHEMA
    assert bound.key == "stub"
    assert bound.__name__ == "stub_derivative"


def test_bound_dynamics_default_initial_state():
    spec = _Dynamics("chute", stub_derivative, PARACHUTE_3T_SCHEMA, ("ax",))
    flight = StubFlight()
    bound = spec.bind(flight)
    canonical = list(range(13))
    # default seeding picks the schema's variables out of the canonical state
    assert bound.initial_state(0.0, canonical) == [0, 1, 2, 3, 4, 5]


def test_bound_dynamics_custom_initial_state():
    def seed(flight, t, canonical_state):
        return [canonical_state[2]]  # only altitude

    schema = StateSchema(("z",))
    spec = _Dynamics("z_only", stub_derivative, schema, ("ax",), initial_state=seed)
    bound = spec.bind(StubFlight())
    assert bound.initial_state(0.0, list(range(13))) == [2]


def test_bound_dynamics_select_atol():
    flight = StubFlight()
    canonical_bound = SIX_DOF_DYNAMICS.bind(flight)
    assert canonical_bound.select_atol(flight.atol) == flight.atol
    chute_spec = _Dynamics("chute", stub_derivative, PARACHUTE_3T_SCHEMA, ("ax",))
    chute_bound = chute_spec.bind(flight)
    assert chute_bound.select_atol(flight.atol) == [1e-3] * 6


def test_scalar_atol_passthrough():
    bound = SIX_DOF_DYNAMICS.bind(StubFlight())
    assert bound.select_atol(1e-5) == 1e-5


def test_bad_atol_length_raises():
    chute_spec = _Dynamics("chute", stub_derivative, PARACHUTE_3T_SCHEMA, ("ax",))
    bound = chute_spec.bind(StubFlight())
    with pytest.raises(ValueError):
        bound.select_atol([1e-3, 1e-3, 1e-3])
