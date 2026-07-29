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

from functools import partial

from ..solution import (
    CANONICAL_SCHEMA,
    PARACHUTE_3T_SCHEMA,
    DerivedQuantity,
    register_derived_quantity,
)
from .flight_derivatives import (
    u_dot,
    u_dot_generalized,
    u_dot_generalized_3dof,
    u_dot_parachute,
    udot_rail1,
)

# RocketPy's built-in derived quantities. Every one of them is genuinely zero
# in a phase that does not compute it (a parachute descent, for example, has no
# aerodynamic moments and no thrust), so they all use the "zero" fallback and
# stay defined across the whole flight. The labels and interpolation match the
# matching Flight attributes (``flight.ax``, ``flight.M1``, ...), so reading a
# quantity either way gives the same Function.
for _name, _label, _unit, _interpolation in (
    ("ax", "Ax", "m/s²", "spline"),
    ("ay", "Ay", "m/s²", "spline"),
    ("az", "Az", "m/s²", "spline"),
    ("alpha1", "α1", "rad/s²", "spline"),
    ("alpha2", "α2", "rad/s²", "spline"),
    ("alpha3", "α3", "rad/s²", "spline"),
    ("R1", "R1", "N", "spline"),
    ("R2", "R2", "N", "spline"),
    ("R3", "R3", "N", "spline"),
    ("M1", "M1", "Nm", "linear"),
    ("M2", "M2", "Nm", "linear"),
    ("M3", "M3", "Nm", "linear"),
    ("net_thrust", "Net Thrust", "N", "linear"),
):
    register_derived_quantity(
        DerivedQuantity(
            _name,
            label=_label,
            unit=_unit,
            absent="zero",
            interpolation=_interpolation,
        )
    )
del _name, _label, _unit, _interpolation

# Derived quantities each kind of phase reports, in the order its derivative
# returns them.
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
        Stable identifier for this kind of phase, stored in saved files.
    derivative : callable
        The free function ``f(flight, t, u, post_processing=False)`` computing
        the state derivative for this phase. With ``post_processing`` it
        returns the phase's derived quantities instead, in ``derived_names``
        order (or as a dictionary of quantity name to value, for a phase that
        reports only some of them).
    schema : StateSchema
        The state variables this phase integrates.
    derived_names : sequence of str
        Ordered names of the derived quantities the phase reports, in the order
        its derivative returns them.
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

    def bind(self, flight, **derivative_kwargs):
        """Return a callable bound to ``flight`` for use as a phase derivative.

        Any extra keyword arguments are fixed onto the derivative. This is how a
        phase carries a parameter that is only known once it begins, such as
        which parachute is descending.
        """
        derivative = self.derivative
        if derivative_kwargs:
            derivative = partial(derivative, **derivative_kwargs)
        return _BoundDynamics(self, flight, derivative)

    def initial_state(self, flight, t, canonical_state):
        """Seed this phase's raw state from a canonical state."""
        if self._initial_state is not None:
            return self._initial_state(flight, t, canonical_state)
        return self.schema.subset_from_canonical(canonical_state)

    def __repr__(self):
        return f"_Dynamics(key={self.key!r}, schema={self.schema!r})"


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
PARACHUTE_DYNAMICS = _Dynamics(
    "parachute", u_dot_parachute, PARACHUTE_3T_SCHEMA, PARACHUTE_DERIVED_NAMES
)


class _BoundDynamics:
    """A :class:`_Dynamics` bound to a specific flight.

    Calling it returns the state derivative, which is what the ODE solver
    needs. :meth:`derived_at` returns the phase's derived quantities, which is
    what post-processing needs. The state schema and the seeding rule stay
    available through the properties below.
    """

    __slots__ = ("dynamics", "flight", "_derivative", "__name__")

    def __init__(self, dynamics, flight, derivative=None):
        self.dynamics = dynamics
        self.flight = flight
        # The derivative with any phase-specific arguments already fixed onto
        # it; falls back to the plain one when the phase needs none.
        self._derivative = dynamics.derivative if derivative is None else derivative
        self.__name__ = getattr(dynamics.derivative, "__name__", dynamics.key)

    def __call__(self, t, u):
        """Return the state derivative at ``t``, as the ODE solver needs it."""
        return self._derivative(self.flight, t, u)

    def derived_at(self, t, u):
        """Return the quantities this phase reports at ``t``.

        Returns
        -------
        list or dict
            The quantities in :attr:`derived_names` order, or a dictionary of
            quantity name to value for a phase that reports only some of them.
            Pass it to ``PhaseSolution.record_derived`` to store it.
        """
        return self._derivative(self.flight, t, u, True)

    def rebind(self, **derivative_kwargs):
        """Return a copy for the same flight with extra derivative arguments."""
        return self.dynamics.bind(self.flight, **derivative_kwargs)

    @property
    def schema(self):
        return self.dynamics.schema

    @property
    def derived_names(self):
        return self.dynamics.derived_names

    @property
    def key(self):
        return self.dynamics.key

    def initial_state(self, t, canonical_state):
        """Seed this phase's raw state from a canonical state."""
        return self.dynamics.initial_state(self.flight, t, canonical_state)

    def select_atol(self, atol):
        """Map the flight's absolute tolerance onto this phase's variables."""
        return self.dynamics.schema.select_atol(atol)

    def __repr__(self):
        return f"_BoundDynamics(key={self.dynamics.key!r})"
