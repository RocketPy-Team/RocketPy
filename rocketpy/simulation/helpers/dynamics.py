import numpy as np

from .flight_derivatives import (
    u_dot,
    u_dot_generalized,
    u_dot_generalized_3dof,
    u_dot_parachute,
    udot_rail1,
)

CANONICAL_STATE_NAMES = (
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "e0",
    "e1",
    "e2",
    "e3",
    "w1",
    "w2",
    "w3",
)

"""Names of the 13 states in RocketPy's full flight state, in order."""
CANONICAL_INDEX = {name: index for index, name in enumerate(CANONICAL_STATE_NAMES)}

# The accelerations, linear and angular.
ACCELERATION_VARS = ("ax", "ay", "az", "alpha1", "alpha2", "alpha3")

# Variables each kind of phase reports besides its states, in the order its
# post-processing pass returns them.
FULL_POST_PROCESS_VARS = (
    *ACCELERATION_VARS,
    "R1",
    "R2",
    "R3",
    "M1",
    "M2",
    "M3",
    "net_thrust",
)

# A parachute descent reports only translational accelerations and the drag force
# components; angular quantities are not integrated.
PARACHUTE_POST_PROCESS_VARS = ("ax", "ay", "az", "R1", "R2", "R3")


class _PhaseDynamics:
    """The states, equations of motion and outputs of one kind of flight phase.

    A phase may integrate fewer states than the 13 canonical ones, more, or
    states of its own (a parafoil heading, sloshing modes). Events, sensors and
    outputs always read the full canonical state, so every phase must be able to
    report it. By default a canonical state the phase does not integrate is held
    at the value it had when the phase began; a phase that rebuilds some of them
    from its own states (an attitude from a heading, say) gives a
    ``to_canonical`` function instead.

    Notes
    -----
    RocketPy defines one of these per kind of flight phase it knows how to fly:
    ``RAIL_DYNAMICS``, ``SOLID_PROPULSION_DYNAMICS``, ``SIX_DOF_DYNAMICS``,
    ``THREE_DOF_DYNAMICS`` and ``PARACHUTE_DYNAMICS``.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        name,
        derivative,
        states,
        post_process_vars=(),
        to_canonical=None,
        to_canonical_dot=None,
        initial_state=None,
    ):
        """Describe one kind of flight phase.

        Parameters
        ----------
        name : str
            Name of this kind of phase, such as ``"parachute"``. It is written
            out with the phase when a flight is saved.
        derivative : callable or None
            The function ``f(flight, t, u)`` giving the time derivative of each
            state in ``states``, in that same order. ``u`` is the phase's own
            state. ``None`` for a phase read back from a saved flight: its
            stored states can still be read by name, but it cannot be flown or
            post-processed again, since a derivative is code and is not saved.

            A phase that reports post-process variables writes
            ``f(flight, t, u, post_processing=False)`` instead, and when the
            flag is true returns those variables in ``post_process_vars`` order.
            A phase that reports nothing never sees the flag and can leave it
            out.
        states : sequence of str
            Ordered names of the states this phase integrates. Position ``i`` in
            this list is the position of that state inside every state row of
            the phase.

            A phase can integrate fewer states than the canonical thirteen, and
            it can add states of its own. An extra state is read by name like
            any other, through ``solution["heading"]`` or ``solution.at(t)``,
            but is not part of the canonical state, so events and sensors do not
            see it. Give an ``initial_state`` rule to seed it.
        post_process_vars : sequence of str, optional
            Ordered names of the variables this phase reports besides its
            states, such as ``("ax", "ay", "az")``. The order must match what
            the post-processing pass returns. Default is no variables.
        to_canonical : callable, optional
            The function ``f(values) -> list`` that returns the 13 canonical
            states, in canonical order, for one moment of this phase.
            ``values`` is a dictionary of name to value holding this phase's
            own states plus the value every other canonical state had at the
            start of the phase, so the function only has to work out the states
            it rebuilds and copy the rest::

                def parafoil_to_canonical(values):
                    half = values["heading"] / 2
                    values["e0"], values["e3"] = math.cos(half), math.sin(half)
                    return [values[name] for name in CANONICAL_STATE_NAMES]

            Default is None: the phase's own canonical states are used, and
            every other canonical state keeps its start-of-phase value.
        to_canonical_dot : callable, optional
            The function ``f(values, values_dot) -> list`` that returns the
            time derivative of the 13 canonical states, in canonical order.
            ``values`` is as for ``to_canonical`` and ``values_dot`` holds the
            time derivative of each state this phase integrates, by name.

            Give it whenever ``to_canonical`` rebuilds a state, since that state
            changes as the phase runs. Sensors and controllers read this
            derivative by position, so a state left out reads as not changing.
            Default is None: the phase's own derivatives are used, and every
            other canonical state has derivative zero.
        initial_state : callable, optional
            Rule ``f(flight, t, canonical_state) -> list`` that seeds this
            phase's state from the full canonical state that ended the previous
            phase. It must return exactly one value per state in ``states``.
            Defaults to picking this phase's states out of the canonical state,
            which works whenever every state it integrates is a canonical one.

        """
        self.name = name
        self.derivative = derivative
        self.states = tuple(states)
        self.state_index = {name: i for i, name in enumerate(self.states)}
        self.width = len(self.states)
        self.is_canonical = self.states == CANONICAL_STATE_NAMES
        self.post_process_vars = tuple(post_process_vars)
        self.post_process_index = {
            name: i for i, name in enumerate(self.post_process_vars)
        }
        self.to_canonical = to_canonical
        self.to_canonical_dot = to_canonical_dot
        self._initial_state = initial_state
        # (canonical slot, own index) for each canonical state this phase
        # integrates: the default way to fill the canonical state.
        self._own_fills = tuple(
            (slot, self.state_index[name])
            for slot, name in enumerate(CANONICAL_STATE_NAMES)
            if name in self.state_index
        )

    def __repr__(self):
        return f"_PhaseDynamics(name={self.name!r}, states={self.states!r})"

    # -- Canonical view ----------------------------------------------------

    def state_values(self, state, start_canonical):
        """Return the name-to-value mapping of one moment of this phase.

        Holds this phase's own states, plus the value every other canonical state
        had at the start of the phase.

        Parameters
        ----------
        state : sequence of float
            The phase's state (without time), in this phase's order.
        start_canonical : sequence of float or None
            The canonical state captured at the start of the phase.

        Returns
        -------
        dict
            State name to value.
        """
        values = (
            dict(zip(CANONICAL_STATE_NAMES, start_canonical))
            if start_canonical is not None
            else {}
        )
        values.update(zip(self.states, state))
        return values

    def canonicalize(self, state, start_canonical):
        """Return the full 13-value canonical state for one of this phase's states.

        Parameters
        ----------
        state : sequence of float
            The phase's state (without time), in this phase's order.
        start_canonical : sequence of float or None
            The canonical state captured at the start of the phase, used to fill
            in any state this phase does not integrate.

        Returns
        -------
        list or sequence of float
            The 13-value canonical state. When this phase integrates all of them
            in canonical order, ``state`` is returned unchanged (no copy).
        """
        if self.is_canonical:
            return state
        if self.to_canonical is not None:
            return list(self.to_canonical(self.state_values(state, start_canonical)))
        result = list(start_canonical)
        for slot, index in self._own_fills:
            result[slot] = state[index]
        return result

    def canonicalize_derivative(self, state_dot, state=None, start_canonical=None):
        """Return the 13-value canonical derivative for one of this phase's.

        This is the ``state_dot`` sensors, controllers and event triggers read.

        Parameters
        ----------
        state_dot : sequence of float
            Time derivative of the phase's state, in this phase's order.
        state : sequence of float, optional
            The phase's state at the same instant. Only needed by a phase with a
            ``to_canonical_dot`` function, which reads the state values.
        start_canonical : sequence of float or None, optional
            The canonical state captured at the start of the phase, for the same
            reason.

        Returns
        -------
        list or sequence of float
            The 13-value canonical derivative. When this phase integrates all of
            them in canonical order, ``state_dot`` is returned unchanged.

        Raises
        ------
        ValueError
            If this phase has a ``to_canonical_dot`` function but ``state`` was
            not given.
        """
        if self.is_canonical:
            return state_dot
        if self.to_canonical_dot is not None:
            if state is None:
                raise ValueError(
                    f"Flight phase '{self.name}' rebuilds canonical states from "
                    f"its own, so computing the canonical derivative needs the "
                    f"phase's state at the same time. Pass state."
                )
            return list(
                self.to_canonical_dot(
                    self.state_values(state, start_canonical),
                    dict(zip(self.states, state_dot)),
                )
            )
        result = [0.0] * len(CANONICAL_STATE_NAMES)
        for slot, index in self._own_fills:
            result[slot] = state_dot[index]
        return result

    # -- Seeding and solver settings ----------------------------------------

    def initial_state(self, flight, t, canonical_state):
        """Seed this phase's state from the canonical state ending the previous one.

        Parameters
        ----------
        flight : Flight
            The flight being simulated.
        t : float
            The time the phase begins, in seconds.
        canonical_state : sequence of float
            The full 13-value canonical state.

        Returns
        -------
        list of float
            The phase's state, in this phase's order.
        """
        if self._initial_state is not None:
            return self._initial_state(flight, t, canonical_state)
        if self.is_canonical:
            return list(canonical_state)
        return [canonical_state[CANONICAL_INDEX[name]] for name in self.states]

    def select_atol(self, atol):
        """Map an absolute-tolerance setting onto this phase's states.

        Parameters
        ----------
        atol : float or sequence of float
            A single tolerance for all states, or one tolerance per canonical
            state (13 values).

        Returns
        -------
        float or list of float
            A single number is returned unchanged. A 13-value canonical vector is
            reduced to this phase's states (states that are not canonical use the
            largest supplied tolerance).

        Raises
        ------
        ValueError
            If a per-state tolerance is given whose length is neither 13 nor the
            number of states this phase integrates.
        """
        if not np.iterable(atol):
            return atol
        atol = list(atol)
        if self.is_canonical or len(atol) == self.width:
            return atol
        if len(atol) == len(CANONICAL_STATE_NAMES):
            fallback = max(atol)
            return [
                atol[CANONICAL_INDEX[name]] if name in CANONICAL_INDEX else fallback
                for name in self.states
            ]
        raise ValueError(
            f"atol vector has length {len(atol)}, which matches neither the "
            f"canonical state (13) nor this flight phase ({self.width})."
        )

    # -- Post-process variables -------------------------------------------

    def post_process_values(self, values):
        """Check that ``values`` has one entry per post-process variable.

        Parameters
        ----------
        values : sequence of float
            The values in ``post_process_vars`` order, as the derivative
            returns them with ``post_processing=True``.

        Returns
        -------
        list of float
            The same values, as a list.

        Raises
        ------
        ValueError
            If the number of values is not the number of variables this phase
            computes.
        """
        values = list(values)
        if len(values) != len(self.post_process_vars):
            raise ValueError(
                f"Flight phase '{self.name}' computes "
                f"{len(self.post_process_vars)} post-process variables "
                f"({', '.join(self.post_process_vars)}), but {len(values)} "
                f"values were given. They must be in that same order."
            )
        return values

    def post_process_at(self, flight, t, u, **kwargs):
        """Return this phase's post-process variables at time ``t``.

        Any extra keyword arguments are the phase-specific ones fixed onto the
        phase when it began, such as which parachute is descending.

        Returns
        -------
        list
            The values in ``post_process_vars`` order. Empty for a phase that
            reports no variables, whose derivative is never asked for them.
        """
        if not self.post_process_vars:
            return []
        return self.derivative(flight, t, u, post_processing=True, **kwargs)

    # -- Binding ----------------------------------------------------------

    def bind(self, flight, **derivative_kwargs):
        """Return a callable bound to ``flight`` for use as a phase derivative.

        Any extra keyword arguments are fixed onto the derivative. This is how a
        phase carries a parameter that is only known once it begins, such as
        which parachute is descending.
        """
        return _BoundDynamics(self, flight, derivative_kwargs)


class _BoundDynamics:
    """A :class:`_PhaseDynamics` bound to a specific flight.

    Calling it returns the state derivative, which is what the ODE solver needs.
    :meth:`post_process_at` and :meth:`initial_state` are the two other
    operations that need the flight. Everything about the phase's states stays
    reachable through :attr:`dynamics`.
    """

    __slots__ = ("dynamics", "flight", "kwargs", "__name__")

    def __init__(self, dynamics, flight, kwargs=None):
        """Bind a phase's dynamics to the flight being simulated.

        Parameters
        ----------
        dynamics : _PhaseDynamics
            The phase being flown.
        flight : Flight
            The flight whose rocket and environment the equations of motion read.
        kwargs : dict, optional
            Extra arguments fixed onto the derivative and the post-processing
            pass, for a parameter only known once the phase begins (such as which
            parachute is descending). Default is none.
        """
        self.dynamics = dynamics
        self.flight = flight
        self.kwargs = dict(kwargs or {})
        self.__name__ = getattr(dynamics.derivative, "__name__", dynamics.name)

    def __call__(self, t, u):
        """Return the state derivative at ``t``, as the ODE solver needs it."""
        return self.dynamics.derivative(self.flight, t, u, **self.kwargs)

    def post_process_at(self, t, u):
        """Return this phase's post-process variables at ``t``."""
        return self.dynamics.post_process_at(self.flight, t, u, **self.kwargs)

    def initial_state(self, t, canonical_state):
        """Seed this phase's state from a canonical state."""
        return self.dynamics.initial_state(self.flight, t, canonical_state)

    def __repr__(self):
        return f"_BoundDynamics(name={self.dynamics.name!r})"


RAIL_DYNAMICS = _PhaseDynamics(
    "rail", udot_rail1, CANONICAL_STATE_NAMES, FULL_POST_PROCESS_VARS
)
SOLID_PROPULSION_DYNAMICS = _PhaseDynamics(
    "solid_propulsion", u_dot, CANONICAL_STATE_NAMES, FULL_POST_PROCESS_VARS
)
SIX_DOF_DYNAMICS = _PhaseDynamics(
    "six_dof", u_dot_generalized, CANONICAL_STATE_NAMES, FULL_POST_PROCESS_VARS
)
THREE_DOF_DYNAMICS = _PhaseDynamics(
    "three_dof", u_dot_generalized_3dof, CANONICAL_STATE_NAMES, FULL_POST_PROCESS_VARS
)
PARACHUTE_DYNAMICS = _PhaseDynamics(
    "parachute",
    u_dot_parachute,
    CANONICAL_STATE_NAMES,
    PARACHUTE_POST_PROCESS_VARS,
)

# Every kind of phase RocketPy ships, by name, so a saved flight gets back the
# dynamics each of its phases was flown with. Add new dynamics here.
BUILT_IN_DYNAMICS = {
    dynamics.name: dynamics
    for dynamics in (
        RAIL_DYNAMICS,
        SOLID_PROPULSION_DYNAMICS,
        SIX_DOF_DYNAMICS,
        THREE_DOF_DYNAMICS,
        PARACHUTE_DYNAMICS,
    )
}
