"""Bundles that pair a flight-phase derivative with its state description.

A flight phase is driven by a derivative function (the equations of motion for
that phase) and integrates a particular set of state variables. ``_Dynamics``
ties those two together, plus the list of derived quantities the phase reports
(accelerations, aerodynamic forces and moments, net thrust) and the rule for
seeding the phase from the state that ended the previous one.

This is an internal building block. It is shaped so that a public "custom
dynamics" extension point can be added later without reworking the pipeline,
but it is not part of the public API yet.
"""

from ..solution import CANONICAL_SCHEMA, PARACHUTE_3T_SCHEMA
from .flight_derivatives import (
    u_dot,
    u_dot_generalized,
    u_dot_generalized_3dof,
    u_dot_parachute,
    udot_rail1,
)

# Derived quantities each kind of phase reports, in the order its derivative
# writes them to the post-processing buffer.
FULL_DERIVED_NAMES = (
    "ax",
    "ay",
    "az",
    "alpha1",
    "alpha2",
    "alpha3",
    "R1",
    "R2",
    "R3",
    "M1",
    "M2",
    "M3",
    "net_thrust",
)

# A parachute descent reports only translational accelerations and the drag
# force components; angular quantities are not integrated.
PARACHUTE_DERIVED_NAMES = ("ax", "ay", "az", "R1", "R2", "R3")


class _Dynamics:
    """Pairs a phase's derivative with its state schema and outputs.

    Parameters
    ----------
    key : str
        Stable identifier stored in saved files and used to look the dynamics
        back up on load.
    derivative : callable
        The free function ``f(flight, t, u, post_processing=False)`` computing
        the state derivative for this phase.
    schema : StateSchema
        The state variables this phase integrates.
    derived_names : sequence of str
        Names of the derived quantities the phase reports, in the order the
        derivative writes them.
    initial_state : callable, optional
        Rule ``f(flight, t, canonical_state) -> list`` that seeds this phase's
        raw state from the full canonical state that ended the previous phase.
        Defaults to picking this schema's variables out of the canonical state.
    name : str, optional
        Human-readable label. Defaults to ``key``.
    """

    def __init__(
        self, key, derivative, schema, derived_names, initial_state=None, name=None
    ):
        self.key = key
        self.derivative = derivative
        self.schema = schema
        self.derived_names = tuple(derived_names)
        self._initial_state = initial_state
        self.name = name or key

    def bind(self, flight):
        """Return a callable bound to ``flight`` for use as a phase derivative."""
        return _BoundDynamics(self, flight)

    def initial_state(self, flight, t, canonical_state):
        """Seed this phase's raw state from a canonical state."""
        if self._initial_state is not None:
            return self._initial_state(flight, t, canonical_state)
        return self.schema.subset_from_canonical(canonical_state)

    def __repr__(self):
        return f"_Dynamics(key={self.key!r}, schema={self.schema!r})"


class _BoundDynamics:
    """A :class:`_Dynamics` bound to a specific flight.

    It is callable with the same signature the solver and post-processing
    expect, ``(t, u, post_processing=False)``, so it can be used anywhere the
    old bound-lambda derivatives were used, while still exposing the schema and
    seeding rule.
    """

    __slots__ = ("spec", "flight", "__name__")

    def __init__(self, spec, flight):
        self.spec = spec
        self.flight = flight
        self.__name__ = getattr(spec.derivative, "__name__", spec.key)

    def __call__(self, t, u, post_processing=False):
        return self.spec.derivative(self.flight, t, u, post_processing)

    @property
    def schema(self):
        return self.spec.schema

    @property
    def derived_names(self):
        return self.spec.derived_names

    @property
    def key(self):
        return self.spec.key

    def initial_state(self, t, canonical_state):
        """Seed this phase's raw state from a canonical state."""
        return self.spec.initial_state(self.flight, t, canonical_state)

    def select_atol(self, atol):
        """Map the flight's absolute tolerance onto this phase's variables."""
        return self.spec.schema.select_atol(atol)

    def __repr__(self):
        return f"_BoundDynamics(key={self.spec.key!r})"


RAIL_DYNAMICS = _Dynamics("rail", udot_rail1, CANONICAL_SCHEMA, FULL_DERIVED_NAMES)
SOLID_PROPULSION_DYNAMICS = _Dynamics(
    "solid_propulsion", u_dot, CANONICAL_SCHEMA, FULL_DERIVED_NAMES
)
SIX_DOF_DYNAMICS = _Dynamics(
    "six_dof", u_dot_generalized, CANONICAL_SCHEMA, FULL_DERIVED_NAMES
)
THREE_DOF_DYNAMICS = _Dynamics(
    "three_dof", u_dot_generalized_3dof, CANONICAL_SCHEMA, FULL_DERIVED_NAMES
)
# The parachute descent integrates only position and velocity; attitude is held
# fixed at its value when the parachute deployed.
PARACHUTE_DYNAMICS = _Dynamics(
    "parachute", u_dot_parachute, PARACHUTE_3T_SCHEMA, PARACHUTE_DERIVED_NAMES
)

DYNAMICS_REGISTRY = {
    dynamics.key: dynamics
    for dynamics in (
        RAIL_DYNAMICS,
        SOLID_PROPULSION_DYNAMICS,
        SIX_DOF_DYNAMICS,
        THREE_DOF_DYNAMICS,
        PARACHUTE_DYNAMICS,
    )
}
