"""What a flight phase is: the states it integrates and the equations that move them.

A flight is integrated in phases (rail, powered ascent, coast, parachute descent,
...). A :class:`_PhaseDynamics` describes one such phase: the states it integrates,
the derivative function that moves them, the variables it reports on the side
(accelerations, aerodynamic forces and moments, net thrust), and how to seed the
phase from the state that ended the previous one.

The full 13-state vector ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]`` is
the *canonical* state: every phase can report it, filling in the states it does not
integrate. Plots, outputs and user callbacks all read that.

Choosing a phase's states
-------------------------

A phase does not have to integrate all thirteen. It can declare a shorter list,
and the states it leaves out are then either held at the value they had when the
phase began or rebuilt by a rule the phase supplies. It can also declare states
of its own on top of the canonical ones, such as a parafoil heading or an air
brake position. That flexibility is real and supported — but **prefer the full
canonical state whenever the phase can supply it**, including states whose
derivative is simply zero.

The reason is that a shorter state is cheap to integrate and expensive to
*observe*. Whenever an event trigger, a sensor or a controller runs, it is handed
the full canonical state, so a phase that integrates fewer states has to rebuild
it on the spot. Events are checked far more often than steps are stored: a
parachute descent with 100 Hz triggers checks its events roughly 130 times per
stored step, so the rebuild is paid 130 times over for each step it saved.

Measured on the ``getting_started`` flight, a parachute descent integrating six
states instead of thirteen saved 320 of 259,958 derivative evaluations (0.1%) and
cost about 2.3% of total run time in reconstruction. That is why the built-in
descent below integrates the whole canonical state and simply reports zero for
the attitude and angular rates.

Declare a shorter state when the states left out genuinely cannot be produced —
not to save integration work.

This is an internal building block. Phase dynamics are defined in RocketPy's own
code; a public "custom dynamics" extension point may be added later.
"""

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

# Variables each kind of phase reports besides its states, in the order its
# post-processing pass returns them.
FULL_POST_PROCESS_VARS = (
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

# A parachute descent reports only translational accelerations and the drag force
# components; angular quantities are not integrated.
PARACHUTE_POST_PROCESS_VARS = ("ax", "ay", "az", "R1", "R2", "R3")


def _phase_cannot_be_flown(flight, t, u, **kwargs):
    """Stand in for the equations of motion of a phase read back from a file."""
    raise NotImplementedError(
        "This flight phase was read back from a saved flight, so the equations "
        "of motion it was flown with are not available: a derivative is code, "
        "and code cannot be saved. Its stored states are still exact and can "
        "still be read by name, but it cannot be flown or post-processed again."
    )


class _PhaseDynamics:
    """The states, equations of motion and outputs of one kind of flight phase.

    A phase that integrates fewer than the 13 canonical states can still report
    the full canonical state for plots, outputs and user callbacks. For each
    canonical state the phase does not integrate, the value is filled in one of
    two ways:

    - the phase's ``reconstruct`` function, if the state is one it rebuilds from
      the phase's own states (for example, a parafoil phase could compute the
      attitude quaternion from a heading it integrates); or
    - the state's value at the start of the phase, held constant. This reproduces
      the behaviour of phases whose unintegrated states simply do not change.

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
        reconstructed_states=(),
        reconstruct=None,
        initial_state=None,
    ):
        """Describe one kind of flight phase.

        Parameters
        ----------
        name : str
            Name of this kind of phase, such as ``"parachute"``. It is written
            out when a flight is saved and looked up again when one is loaded,
            so two kinds of phase must not share a name.
        derivative : callable
            The function ``f(flight, t, u)`` giving the time derivative of each
            state in ``states``, in that same order. ``u`` is the phase's own
            state.

            A phase that reports post-process variables writes
            ``f(flight, t, u, post_processing=False)`` instead, and when the
            flag is true returns those variables in ``post_process_vars`` order,
            or as a name-to-value dictionary. A phase that reports nothing never
            sees the flag and can leave it out.
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
        reconstructed_states : sequence of str, optional
            Canonical states this phase does not integrate but rebuilds from the
            ones it does, such as an attitude quaternion built from a heading.
            Needs ``reconstruct``. Any canonical state named neither here nor in
            ``states`` is held at the value it had when the phase began,
            unchanged for the rest of the phase. Default is none, so every state
            the phase does not integrate is held constant.
        reconstruct : callable, optional
            The function ``f(values, values_dot=None)`` that rebuilds the states
            named in ``reconstructed_states``, returning one value per name, in
            that order.

            ``values`` is a dictionary of name to value, holding this phase's
            states plus the value of every other canonical state had at the 
            start of the phase.

            It is called a second way to get the time derivative of each of 
            those states: ``values_dot`` then holds the time derivative of each 
            state this phase integrates, by name, and the function applies the
            chain rule. A reconstructed state changes as the phase runs, so its
            derivative cannot be taken as zero the way a held state's can::

                reconstructed_states=("e0", "e3"),
                reconstruct=lambda values, values_dot=None: (
                    (math.cos(values["heading"] / 2),
                     math.sin(values["heading"] / 2))
                    if values_dot is None
                    else (
                        -0.5 * math.sin(values["heading"] / 2)
                        * values_dot["heading"],
                        0.5 * math.cos(values["heading"] / 2)
                        * values_dot["heading"],
                    )
                )

            Sensors and controllers read the canonical derivative by position,
            so a wrong value here is reported as though it were measured. 
            Default is None.
        initial_state : callable, optional
            Rule ``f(flight, t, canonical_state) -> list`` that seeds this
            phase's state from the full canonical state that ended the previous
            phase. It must return exactly one value per state in ``states``.
            Defaults to picking this phase's states out of the canonical state,
            which works whenever every state it integrates is a canonical one.

        Raises
        ------
        TypeError
            If ``derivative`` is not callable. A phase whose equations of motion
            are unavailable, such as one read back from a saved flight, is built
            with :meth:`states_only` instead.
        ValueError
            If a name appears both in ``states`` and in ``reconstructed_states``
            (a phase either integrates a state or rebuilds it, not both); if
            ``reconstructed_states`` names something that is not a canonical
            state; or if only one of ``reconstructed_states`` and ``reconstruct``
            is given.

        See Also
        --------
        _PhaseDynamics.states_only : Build a phase that reports states but cannot
            be integrated.
        """
        self.name = name
        self.derivative = derivative
        self.states = tuple(states)
        self.state_index = {name: i for i, name in enumerate(self.states)}
        self.width = len(self.states)
        self.is_canonical = self.states == CANONICAL_STATE_NAMES
        self.reconstructed_states = tuple(reconstructed_states)
        self.reconstructed_index = {
            name: i for i, name in enumerate(self.reconstructed_states)
        }
        self.reconstruct = reconstruct
        self.post_process_vars = tuple(post_process_vars)
        self.post_process_index = {
            name: i for i, name in enumerate(self.post_process_vars)
        }
        self._initial_state = initial_state

        self._validate()

        # Resolve how to fill each canonical slot: from one of this phase's own
        # states, from the reconstruction function, or by holding the value
        # captured at the start of the phase.
        own_fills = []
        freeze_fills = []
        for slot, canonical_name in enumerate(CANONICAL_STATE_NAMES):
            if canonical_name in self.state_index:
                own_fills.append((slot, self.state_index[canonical_name]))
            elif canonical_name not in self.reconstructed_index:
                freeze_fills.append(slot)
        self._own_fills = tuple(own_fills)
        self._freeze_fills = tuple(freeze_fills)
        # Where each value the reconstruction function returns belongs, in the
        # order it returns them.
        self._reconstructed_slots = tuple(
            CANONICAL_INDEX[name] for name in self.reconstructed_states
        )

        # Canonical states held at their value from the start of the phase.
        self.frozen_states = tuple(
            CANONICAL_STATE_NAMES[slot] for slot in self._freeze_fills
        )

    def _validate(self):
        """Reject a declaration that contradicts itself or is half-finished."""
        if not callable(self.derivative):
            raise TypeError(
                f"The derivative of flight phase '{self.name}' is "
                f"{type(self.derivative).__name__}, not a function. Every phase "
                f"needs the equations that move its states. Use "
                f"_PhaseDynamics.states_only() for a phase that only has to report "
                f"states it already holds, such as one read back from a saved "
                f"flight."
            )
        both = set(self.states) & set(self.reconstructed_states)
        if both:
            raise ValueError(
                f"{', '.join(sorted(both))} appear(s) both in states and in "
                f"reconstructed_states. A phase either integrates a state or "
                f"reconstructs it, not both."
            )
        not_canonical = set(self.reconstructed_states) - set(CANONICAL_STATE_NAMES)
        if not_canonical:
            raise ValueError(
                f"reconstructed_states names {', '.join(sorted(not_canonical))}, "
                f"which is not a canonical state. Only these can be reconstructed: "
                f"{', '.join(CANONICAL_STATE_NAMES)}."
            )
        if self.reconstructed_states and not callable(self.reconstruct):
            raise ValueError(
                f"This flight phase reconstructs "
                f"{', '.join(self.reconstructed_states)}, but no reconstruct "
                f"function was given to rebuild them."
            )
        if self.reconstruct is not None and not self.reconstructed_states:
            raise ValueError(
                "A reconstruct function was given, but reconstructed_states is "
                "empty, so there is nothing for it to rebuild. Name the canonical "
                "states it returns."
            )

    @classmethod
    def states_only(cls, name, states):
        """Build a phase that can report its states but cannot be flown.

        A derivative is code, so it cannot be written to a file. A flight read
        back from one therefore has no equations of motion for its phases, and
        gets this instead: every stored state is still exact and still readable
        by name, but integrating or post-processing the phase again raises
        ``NotImplementedError``.

        Parameters
        ----------
        name : str
            Name of the kind of phase this was, as stored in the file.
        states : sequence of str
            Ordered names of the states the phase integrated.
        """
        return cls(name, _phase_cannot_be_flown, states)

    def __repr__(self):
        return f"_PhaseDynamics(name={self.name!r}, states={self.states!r})"

    def state_index_of(self, name):
        """Return the position of ``name`` in this phase's state row.

        Raises
        ------
        KeyError
            If ``name`` is not one of the states this phase integrates.
        """
        try:
            return self.state_index[name]
        except KeyError as error:
            raise KeyError(
                f"State '{name}' is not integrated by this flight phase. "
                f"Available states: {self.states}."
            ) from error

    def canonicalize(self, state, frozen):
        """Return the full 13-value canonical state for one of this phase's states.

        Parameters
        ----------
        state : sequence of float
            The phase's state (without time), in this phase's order.
        frozen : sequence of float or None
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
        # Frozen states keep the value they had at the start of the phase, so
        # starting from that state fills all of them at once. Only the states
        # this phase supplies are then written over it.
        if self._freeze_fills:
            result = list(frozen)
        else:
            result = [0.0] * len(CANONICAL_STATE_NAMES)
        for slot, index in self._own_fills:
            result[slot] = state[index]
        if self._reconstructed_slots:
            rebuilt = self.reconstruct(self.reconstruction_inputs(state, frozen))
            if len(rebuilt) != len(self._reconstructed_slots):
                self._reject_reconstruction(rebuilt)
            for slot, value in zip(self._reconstructed_slots, rebuilt):
                result[slot] = value
        return result

    def canonicalize_derivative(self, state_dot, state=None, frozen=None):
        """Return the 13-value canonical derivative for one of this phase's.

        This is the ``state_dot`` sensors, controllers and event triggers read.
        They read it by position, so every slot has to mean what it says:

        - a state this phase integrates contributes its own derivative;
        - a state held at its start-of-phase value contributes zero, because it
          really does not change during the phase;
        - a reconstructed state contributes what the phase's ``reconstruct``
          function computes for it, since it does change.

        Parameters
        ----------
        state_dot : sequence of float
            Time derivative of the phase's state, in this phase's order.
        state : sequence of float, optional
            The phase's state at the same instant. Only needed by a phase that
            reconstructs a state, whose rule reads the state values.
        frozen : sequence of float or None, optional
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
            If this phase reconstructs a state but ``state`` was not given, so
            the reconstruction rules have nothing to read.
        """
        if self.is_canonical:
            return state_dot
        result = [0.0] * len(CANONICAL_STATE_NAMES)
        for slot, index in self._own_fills:
            result[slot] = state_dot[index]
        if self._reconstructed_slots:
            if state is None:
                raise ValueError(
                    f"This flight phase reconstructs "
                    f"{', '.join(self.reconstructed_states)}, so computing the "
                    f"canonical derivative needs the phase's state at the same "
                    f"time. Pass state."
                )
            rebuilt = self.reconstruct(
                self.reconstruction_inputs(state, frozen),
                dict(zip(self.states, state_dot)),
            )
            if len(rebuilt) != len(self._reconstructed_slots):
                self._reject_reconstruction(rebuilt)
            for slot, value in zip(self._reconstructed_slots, rebuilt):
                result[slot] = value
        return result

    def _reject_reconstruction(self, rebuilt):
        """Explain a reconstruction that does not answer for every state named."""
        raise ValueError(
            f"The reconstruct function of flight phase '{self.name}' returned "
            f"{len(rebuilt)} values, but reconstructed_states names "
            f"{len(self.reconstructed_states)} "
            f"({', '.join(self.reconstructed_states)}). It must return one value "
            f"per name, in that order, both for the states and for their "
            f"derivatives."
        )

    def reconstruction_inputs(self, state, frozen):
        """Return the name-to-value mapping a reconstruction function reads.

        Holds this phase's own states, plus the value every other canonical state
        had at the start of the phase.

        Parameters
        ----------
        state : sequence of float
            The phase's state (without time), in this phase's order.
        frozen : sequence of float or None
            The canonical state captured at the start of the phase.

        Returns
        -------
        dict
            State name to value.
        """
        values = dict(zip(CANONICAL_STATE_NAMES, frozen)) if frozen is not None else {}
        values.update(zip(self.states, state))
        return values

    def state_from_canonical(self, canonical_state):
        """Pick this phase's states out of a full canonical state.

        Used to seed a new flight phase from the state that ended the previous
        one. Every state this phase integrates must be a canonical state for this
        to work; a phase with states that are not canonical (such as a heading)
        supplies its own ``initial_state`` rule instead.

        Parameters
        ----------
        canonical_state : sequence of float
            The full 13-value canonical state.

        Returns
        -------
        list of float
            The phase's state, in this phase's order.
        """
        if self.is_canonical:
            return list(canonical_state)
        return [canonical_state[CANONICAL_INDEX[name]] for name in self.states]

    def select_atol(self, atol):
        """Map an absolute-tolerance setting onto this phase's states.

        The flight's ``atol`` is either a single number or a 13-value vector in
        canonical order. This returns the tolerance the solver should use for a
        phase that integrates only this phase's states.

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
        if self.is_canonical:
            return atol
        if len(atol) == len(CANONICAL_STATE_NAMES):
            fallback = max(atol)
            return [
                atol[CANONICAL_INDEX[name]] if name in CANONICAL_INDEX else fallback
                for name in self.states
            ]
        if len(atol) == self.width:
            return atol
        raise ValueError(
            f"atol vector has length {len(atol)}, which matches neither the "
            f"canonical state (13) nor this flight phase ({self.width})."
        )

    def initial_state(self, flight, t, canonical_state):
        """Seed this phase's state from a canonical state."""
        if self._initial_state is not None:
            return self._initial_state(flight, t, canonical_state)
        return self.state_from_canonical(canonical_state)

    # -- Post-process variables -------------------------------------------

    def post_process_index_of(self, name):
        """Return the position of ``name`` in this phase's post-process row.

        Raises
        ------
        KeyError
            If this phase does not compute the variable.
        """
        try:
            return self.post_process_index[name]
        except KeyError as error:
            raise KeyError(
                f"Post-process variable '{name}' is not computed by this flight "
                f"phase. It computes: {', '.join(self.post_process_vars)}."
            ) from error

    def post_process_values(self, values):
        """Return ``values`` in this phase's ``post_process_vars`` order.

        The time is not part of the result. Post-process values are stored
        alongside the state row they were computed from, so they already share
        that row's time.

        Parameters
        ----------
        values : sequence of float or dict
            Either the values in ``post_process_vars`` order, or a dictionary of
            name to value. A dictionary may leave variables out, which are then
            recorded as zero.

        Returns
        -------
        list of float
            One value per variable this phase computes, in
            ``post_process_vars`` order.

        Raises
        ------
        ValueError
            If a sequence is given whose length is not the number of variables
            this phase computes.
        KeyError
            If a dictionary names a variable this phase does not compute.
        """
        width = len(self.post_process_vars)
        if isinstance(values, dict):
            row = [0.0] * width
            for name, value in values.items():
                row[self.post_process_index_of(name)] = value
            return row
        values = list(values)
        if len(values) != width:
            raise ValueError(
                f"This flight phase computes {width} post-process variables "
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
        list or dict
            The values in ``post_process_vars`` order, or a dictionary of name to
            value for a phase that reports only some of them. Pass it to
            :meth:`post_process_row` to lay it out for storage. Empty for a phase
            that reports no variables, whose derivative is never asked for them.
        """
        if not self.post_process_vars:
            return {}
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
    :meth:`post_process_at` returns the phase's post-process variables, which is
    what post-processing needs. Everything about the phase's states stays
    reachable through :attr:`dynamics`.
    """

    __slots__ = ("dynamics", "flight", "_kwargs", "__name__")

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
        # Phase-specific arguments fixed onto the derivative, such as which
        # parachute is descending.
        self._kwargs = dict(kwargs or {})
        self.__name__ = getattr(dynamics.derivative, "__name__", dynamics.name)

    def __call__(self, t, u):
        """Return the state derivative at ``t``, as the ODE solver needs it."""
        return self.dynamics.derivative(self.flight, t, u, **self._kwargs)

    def post_process_at(self, t, u):
        """Return this phase's post-process variables at ``t``.

        Returns
        -------
        list or dict
            Whatever the phase reports; pass it to
            ``_PhaseDynamics.post_process_row`` to lay it out for storage.
        """
        return self.dynamics.post_process_at(self.flight, t, u, **self._kwargs)

    def bind(self, flight=None, **derivative_kwargs):
        """Return the same phase bound again, with different fixed arguments.

        Mirrors :meth:`_PhaseDynamics.bind` so that both forms can be bound the
        same way, which is what lets an event name either one. The arguments
        replace whatever this one was bound with rather than adding to them.

        Parameters
        ----------
        flight : Flight, optional
            The flight to bind to. Defaults to the flight this one is already
            bound to, which is what re-binding during a simulation wants.
        **derivative_kwargs
            Values fixed onto the derivative, such as which parachute is
            descending.
        """
        return self.dynamics.bind(
            self.flight if flight is None else flight, **derivative_kwargs
        )

    @property
    def name(self):
        return self.dynamics.name

    @property
    def post_process_vars(self):
        return self.dynamics.post_process_vars

    def initial_state(self, t, canonical_state):
        """Seed this phase's state from a canonical state."""
        return self.dynamics.initial_state(self.flight, t, canonical_state)

    def select_atol(self, atol):
        """Map the flight's absolute tolerance onto this phase's states."""
        return self.dynamics.select_atol(atol)

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
    # Only position and velocity move under a parachute, but the descent still
    # carries the whole canonical state. See "Choosing a phase's states" in this
    # module's docstring for why that is the faster choice here.
    CANONICAL_STATE_NAMES,
    PARACHUTE_POST_PROCESS_VARS,
)

# Looking a saved phase's name up here is what gives it its states back when a
# flight is loaded, since a derivative is code and cannot be written to a file.
_DYNAMICS_BY_NAME = {
    dynamics.name: dynamics
    for dynamics in (
        RAIL_DYNAMICS,
        SOLID_PROPULSION_DYNAMICS,
        SIX_DOF_DYNAMICS,
        THREE_DOF_DYNAMICS,
        PARACHUTE_DYNAMICS,
    )
}


def dynamics_for_name(name, state_names):
    """Return the :class:`_PhaseDynamics` a saved flight phase was flown with.

    A phase whose name is missing or unrecognized (an older saved flight, or one
    written by a newer version of RocketPy) gets a states-only ``_PhaseDynamics``
    instead: its stored states are still exact and can still be read by name, but
    it cannot be integrated or post-processed again. The same happens when the
    name is known but the states saved with it are not the ones that phase
    integrates today, which is how a flight saved by an older version keeps
    reading correctly after a phase's states change.

    Parameters
    ----------
    name : str or None
        The phase's ``dynamics`` name as stored in the file.
    state_names : sequence of str
        The phase's state names as stored in the file. Used to check that a
        recognized name still describes the same states, and as the fallback.
    """
    dynamics = _DYNAMICS_BY_NAME.get(name)
    if dynamics is not None and dynamics.states == tuple(state_names):
        return dynamics
    return _PhaseDynamics.states_only(name or "unknown", state_names)
