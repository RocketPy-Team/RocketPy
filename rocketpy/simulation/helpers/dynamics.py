import inspect
import math

import numpy as np

from ...control._helpers import initial_controls, not_finite_error
from ..events.views import ControlsView
from .conditions import FlightConditions
from .flight_derivatives import (
    u_dot,
    u_dot_generalized,
    u_dot_generalized_3dof,
    u_dot_parachute,
    udot_rail1,
)

# The names of the 13 canonical states are defined with the state view, a
# module with no imports of its own, and reached from here by everyone else
from .state import CANONICAL_INDEX, CANONICAL_STATE_NAMES

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
    states of its own that are not among the 13. Events, sensors and
    outputs always read the full canonical state, so every phase must be able to
    report it. By default a canonical state the phase does not integrate is held
    at the value it had when the phase began; a phase that rebuilds some of them
    from its own states (an attitude from an angle of its own, say) gives a
    ``to_canonical`` function instead.

    Notes
    -----
    RocketPy defines one description per kind of flight phase it knows how to
    fly: ``RAIL_DYNAMICS``, ``SOLID_PROPULSION_DYNAMICS``, ``SIX_DOF_DYNAMICS``,
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
        reads_controls=True,
    ):
        """Describe one kind of flight phase.

        Parameters
        ----------
        name : str
            Name of this kind of phase, such as ``"parachute"``.
        derivative : callable or None
            The function ``f(flight, t, u, controls)`` giving the time
            derivative of each state in ``states``, in that same order. ``u``
            is the phase's own state and ``controls`` the flight's control
            inputs, a dictionary from name to value, such as
            ``controls["canards.deflection"]``.
            With ``None`` the phase's states can still be read by name, but the
            phase cannot be flown or post-processed.

            A phase that reports post-process variables writes
            ``f(flight, t, u, controls, post_processing=False)`` instead, and when the
            flag is true returns those variables in ``post_process_vars`` order.
            A phase that reports nothing never sees the flag and can leave it
            out.

            A derivative that drives the rocket's effectors takes
            ``conditions=None`` as well: the :class:`FlightConditions` of the
            moment, which it hands to the effector sums of
            :mod:`rocketpy.simulation.helpers.flight_derivatives`. It is given
            whenever
            the rocket has effectors, to every derivative that names the
            parameter or takes ``**kwargs``.
        states : sequence of str
            Ordered names of the states this phase integrates. Position ``i`` in
            this list is the position of that state inside every state row of
            the phase.

            A phase can integrate fewer states than the canonical thirteen, and
            it can add states of its own. An extra state is read by name like
            any other, through ``solution["my_state"]`` or ``solution.at(t)``,
            and by events as ``context.state.my_state``. It is not part of the
            canonical state, so sensors do not see it. It starts from the
            value of the state of the same name in the previous phase; give
            an ``initial_state`` rule to seed it otherwise. A state cannot be
            called ``names`` or ``values``, which ``context.state`` uses
            itself.
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
            it rebuilds and copy the rest:

            .. code-block:: python

                def my_to_canonical(values):
                    half = values["my_angle"] / 2
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
            Rule ``f(flight, t, canonical_state, previous) -> list`` that seeds
            this phase's state from the full canonical state that ended the
            previous phase, and from ``previous``, every state of the previous
            phase by name (an empty dictionary at the start of the flight). It
            must return exactly one value per state in ``states``.
            Defaults to picking this phase's states out of the canonical state,
            which works whenever every state it integrates is a canonical one.
        reads_controls : bool, optional
            Whether ``derivative`` reads the control inputs. Give ``False`` for
            a phase where the rocket's controllable parts do nothing, as under
            a parachute: a control input written during the phase is still
            stored, but the solver is not stopped at the moment it is written,
            which makes the phase faster to fly. Default is ``True``.
        """
        self.name = name
        self.reads_controls = bool(reads_controls)
        self.derivative = derivative
        self.states = tuple(states)
        # ``context.state`` has these two names itself
        taken = [state for state in self.states if state in ("names", "values")]
        if taken:
            raise ValueError(
                f"The flight phase '{name}' cannot have a state called "
                f"{' or '.join(repr(state) for state in taken)}: "
                "`context.state.names` and `context.state.values` already "
                "mean the names and the values of the state. Give it another name."
            )
        self.state_index = {name: i for i, name in enumerate(self.states)}
        self.width = len(self.states)
        self.is_canonical = self.states == CANONICAL_STATE_NAMES
        # The 13 canonical states first, in order, then states of its own
        self._canonical_first = (
            self.states[: len(CANONICAL_STATE_NAMES)] == CANONICAL_STATE_NAMES
        )
        self.post_process_vars = tuple(post_process_vars)
        self.post_process_index = {
            name: i for i, name in enumerate(self.post_process_vars)
        }
        self.to_canonical = to_canonical
        self.to_canonical_dot = to_canonical_dot
        self._initial_state = initial_state
        # Set by _with_controllables on the extended dynamics it builds for one
        # rocket: the dynamics it was built from. Dynamics that were not
        # extended are their own base.
        self._base = self
        # Whether the derivative takes the canonical state the phase began
        # with, as ``start_canonical``: only the one _with_controllables builds for
        # a phase that is not canonical does, to hand the canonical state to
        # the rocket's controllables. _BoundDynamics.started_from gives it.
        self._takes_start = False
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
            in any state this phase does not integrate. ``None`` only for a
            phase that integrates every canonical state or rebuilds them all.

        Returns
        -------
        list or sequence of float
            The 13-value canonical state. When this phase integrates all of them
            in canonical order, ``state`` is returned unchanged (no copy).

        Raises
        ------
        ValueError
            If this phase holds some canonical states at their start-of-phase
            value and ``start_canonical`` was not given.
        """
        if self.is_canonical:
            return state
        if self.to_canonical is not None:
            return list(self.to_canonical(self.state_values(state, start_canonical)))
        if self._canonical_first:
            return state[: len(CANONICAL_STATE_NAMES)]
        if start_canonical is None:
            if len(self._own_fills) < len(CANONICAL_STATE_NAMES):
                raise ValueError(
                    f"Flight phase '{self.name}' does not integrate every "
                    "canonical state, so its canonical state needs the state "
                    "the phase began with. Pass start_canonical."
                )
            return [state[index] for _, index in self._own_fills]
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
        if self._canonical_first:
            return state_dot[: len(CANONICAL_STATE_NAMES)]
        result = [0.0] * len(CANONICAL_STATE_NAMES)
        for slot, index in self._own_fills:
            result[slot] = state_dot[index]
        return result

    # -- Seeding and solver settings ----------------------------------------

    def initial_state(self, flight, t, canonical_state, previous=None):
        """Seed this phase's state from the state ending the previous one.

        Parameters
        ----------
        flight : Flight
            The flight being simulated.
        t : float
            The time the phase begins, in seconds.
        canonical_state : sequence of float
            The full 13-value canonical state.
        previous : dict, optional
            Every state of the previous phase by name, canonical or not. A
            state that is not canonical carries over from it when this phase
            has one of the same name. Default ``None``, at the start of the
            flight.

        Returns
        -------
        list of float
            The phase's state, in this phase's order.
        """
        previous = previous or {}
        if self._initial_state is not None:
            return self._initial_state(flight, t, canonical_state, previous)
        if self.is_canonical:
            return list(canonical_state)
        values = {**previous, **dict(zip(CANONICAL_STATE_NAMES, canonical_state))}
        return [values[name] for name in self.states]

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
            A single number is returned unchanged. A 13-value canonical vector
            is matched to this phase's states by name (states that are not
            canonical use the largest supplied tolerance), even when the phase
            also integrates 13 states. A vector with one value per state of
            this phase is returned as given.

        Raises
        ------
        ValueError
            If a per-state tolerance is given whose length is neither 13 nor the
            number of states this phase integrates.
        """
        if not np.iterable(atol):
            return atol
        atol = list(atol)
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

    def post_process_at(self, flight, t, u, controls, **kwargs):
        """Return this phase's post-process variables at time ``t``.

        ``u`` is the phase's state and ``controls`` the control inputs at that
        time.
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
        return self.derivative(flight, t, u, controls, post_processing=True, **kwargs)

    # -- Binding ----------------------------------------------------------

    def bind(self, flight, **derivative_kwargs):
        """Return these dynamics bound to ``flight``, as a :class:`_BoundDynamics`
        the solver can call.

        Any extra keyword arguments are fixed onto the derivative. This is how a
        flight phase carries a parameter that is only known once it begins,
        such as which parachute is descending. The rocket's controllables and
        continuous controllers are folded in first (see
        :func:`_with_controllables`), so the bound dynamics may have more states
        than these.
        """
        rocket = getattr(flight, "rocket", None)
        dynamics = self if rocket is None else _with_controllables(self, rocket)
        return _BoundDynamics(dynamics, flight, derivative_kwargs)


class _BoundDynamics:
    """A :class:`_PhaseDynamics` bound to a specific flight.

    Calling it returns the state derivative, which is what the ODE solver needs.
    :meth:`post_process_at` and :meth:`initial_state` are the two other
    operations that need the flight. Everything about the phase's states stays
    reachable through :attr:`dynamics`.
    """

    __slots__ = (
        "dynamics",
        "flight",
        "kwargs",
        "start_canonical",
        "_kwargs",
        "__name__",
    )

    def __init__(self, dynamics, flight, kwargs=None, start_canonical=None):
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
        start_canonical : sequence of float, optional
            The canonical state the phase began with. Kept only by dynamics
            whose derivative takes it, those extended for a rocket from a
            phase that is not canonical, which fill the canonical states the
            phase does not integrate from it. Default is none: see
            :meth:`started_from`.
        """
        self.dynamics = dynamics
        self.flight = flight
        self.kwargs = dict(kwargs or {})
        # pylint: disable-next=protected-access
        self.start_canonical = start_canonical if dynamics._takes_start else None
        # What the derivative is called with: the phase's arguments, and the
        # start when kept
        self._kwargs = (
            self.kwargs
            if self.start_canonical is None
            else {**self.kwargs, "start_canonical": self.start_canonical}
        )
        self.__name__ = getattr(dynamics.derivative, "__name__", dynamics.name)

    def started_from(self, start_canonical):
        """Return these dynamics for a phase that began with the canonical
        state ``start_canonical``: a copy that carries it when the derivative
        takes it, else these dynamics themselves.

        A flight calls this as each phase begins, so a phase that does not
        integrate every canonical state knows the values to hold the others
        at, in every call the solver or the post-processing makes.
        """
        if not self.dynamics._takes_start:  # pylint: disable=protected-access
            return self
        return _BoundDynamics(self.dynamics, self.flight, self.kwargs, start_canonical)

    def __call__(self, t, u):
        """Return the state derivative at ``t``, as the ODE solver needs it,
        with the flight's control inputs as they are now."""
        return self.dynamics.derivative(
            self.flight, t, u, self.flight._controls, **self._kwargs
        )

    def derivative_at(self, t, u, controls):
        """Return the state derivative at ``t`` with the control inputs
        ``controls``."""
        return self.dynamics.derivative(self.flight, t, u, controls, **self._kwargs)

    def base_derivative(self, t, u, controls, post_processing=False, conditions=None):
        """Return what the phase's own equations give, without the states of
        the rocket's controllables the flight adds to them (see
        :func:`_with_controllables`).

        For another derivative that reuses these equations from inside its own
        call: ``u`` is then the phase's own states, ``controls`` already holds
        the values the controllables act with and ``conditions`` is what that
        derivative was handed, passed on when given. With ``post_processing``
        the post-process variables are returned instead of the derivative.
        """
        kwargs = (
            self.kwargs
            if conditions is None
            else {**self.kwargs, "conditions": conditions}
        )
        return self.dynamics._base.derivative(  # pylint: disable=protected-access
            self.flight, t, u, controls, post_processing=post_processing, **kwargs
        )

    def post_process_at(self, t, u, controls=None):
        """Return this phase's post-process variables at ``t``.

        ``controls`` are the control inputs in force at ``t``, such as the values
        stored with a row of the solution. ``None`` uses the flight's control
        inputs as they are now.
        """
        if controls is None:
            controls = getattr(self.flight, "_controls", None)
        return self.dynamics.post_process_at(
            self.flight, t, u, controls, **self._kwargs
        )

    def initial_state(self, t, canonical_state, previous=None):
        """Seed this phase's state from a canonical state, and the previous
        phase's states by name."""
        return self.dynamics.initial_state(self.flight, t, canonical_state, previous)

    def __repr__(self):
        return f"_BoundDynamics(name={self.dynamics.name!r})"


def _with_controllables(dynamics, rocket):  # pylint: disable=too-many-statements
    """Return a new ``_PhaseDynamics`` that also integrates the rocket's
    controllables, or ``dynamics`` itself when there is nothing to add.

    The new dynamics appends to the phase's states the states of the rocket's
    continuous controllers (an integrator), then those of its other
    controllables (a servo angle). On every solver call its derivative:

    1. runs the continuous controllers' laws in turn, which write the
       control inputs, and keeps what each wrote inside the limits, so the
       next law reads the limited values;
    2. lets each controllable turn the values written to its controls
       into the ones it acts with;
    3. calls the phase's derivative with those values, and appends the
       derivatives of the added states to what it returns.

    The laws and the controllables are told about the flight through a
    :class:`FlightConditions`, which reads the 13 canonical states. A
    phase with other states, more or fewer, has them worked out from its
    own at every call (see :meth:`_PhaseDynamics.canonicalize`), so the
    conditions are right in any phase. The same conditions are handed to the phase's derivative
    as ``conditions`` when it takes them, for the rocket's effectors; a
    rocket with effectors is extended for that alone.

    When there is a continuous controller, the control inputs its law
    wrote are also stored with every row, as ``controls.<key>.<control>``,
    for ``Flight.controls``.

    Parameters
    ----------
    dynamics : _PhaseDynamics
        The phase's states and equations of motion.
    rocket : Rocket
        The rocket whose controllables are folded in.

    Returns
    -------
    _PhaseDynamics
        The extended dynamics, or ``dynamics``.
    """
    if dynamics._base is not dynamics:
        return dynamics
    controls, control_names, limits = initial_controls(rocket.controllables)
    # The names of the added states: distinct, and checked to be, when
    # each controllable was added to the rocket
    names = []

    def place(controllable):
        # Add the controllable's states after the ones already there and
        # return where they sit in the phase's state
        start = dynamics.width + len(names)
        names.extend(controllable._state_names())
        return slice(start, dynamics.width + len(names))

    # The continuous controllers, then the parts with states or an
    # ``actual_controls`` of their own, each with where its states sit.
    # Everything the derivative needs is worked out here, once: the
    # derivative runs at every evaluation of the equations of motion, so
    # it only loops over what is ready.
    laws = tuple(
        (controller, place(controller)) for controller in rocket._continuous_controllers
    )
    acting = tuple(
        (part, place(part), control_names.get(part, ()))
        for part in rocket._parts_with_states
    )
    # Whether the phase's equations accept ``conditions``, by naming it
    # or by taking ``**kwargs``
    try:
        parameters = inspect.signature(dynamics.derivative).parameters.values()
        takes_conditions = any(
            p.name == "conditions" or p.kind is p.VAR_KEYWORD for p in parameters
        )
    except (TypeError, ValueError):
        takes_conditions = False
    # The effectors are driven by the equations that take the conditions,
    # which are built here
    has_effectors = takes_conditions and bool(rocket.effectors.get_components())
    if not (laws or acting or has_effectors):
        return dynamics
    # The controllers whose states are integrated
    integrated = tuple(law for law in laws if law[0].states)
    # Each added state with its position in the phase's state
    published = tuple(zip(names, range(dynamics.width, dynamics.width + len(names))))
    # With a law, what the laws wrote is stored with every row
    recorded = tuple(controls) if laws else ()
    # The control inputs with limits: a law's write is kept inside them
    limited = tuple(
        (name, *bounds) for name, bounds in limits.items() if bounds is not None
    )
    # The ones without: a law's write is only checked to be a finite number
    unlimited = tuple(name for name, bounds in limits.items() if bounds is None)
    is_finite = math.isfinite
    # The laws read and write the control inputs as a sampled controller
    # does, ``controls.canards.fin_1``. One view, pointed at each call's
    # values: building one per call costs more than everything else here
    view = ControlsView({}, rocket.controllables)
    width, equations = dynamics.width, dynamics.derivative
    state_names = (*dynamics.states, *names)
    is_canonical, canonicalize = dynamics.is_canonical, dynamics.canonicalize

    def written_by(controller):
        return f"the continuous controller {controller.name!r}"

    def derivative(
        flight,
        t,
        u,
        controls,
        post_processing=False,
        start_canonical=None,
        **kwargs,
    ):
        own = u[:width]
        # The 13 canonical states, which the conditions read. A canonical
        # state this phase does not integrate comes from the state the
        # phase began with, which the bound dynamics pass
        canonical = own if is_canonical else canonicalize(own, start_canonical)
        # What the controllables act with: the control inputs, and each
        # added state by name, which is how the phase's own equations read
        # a controllable's state (they are only given the phase's states)
        actual = dict(controls)
        for name, index in published:
            actual[name] = u[index]
        conditions = FlightConditions(
            flight.rocket, flight.env, t, canonical, state_names, u
        )
        if takes_conditions:
            kwargs["conditions"] = conditions
        rates = []
        if laws:
            view._point_at(actual)
            for controller, piece in laws:
                controller.law(u[piece], view, conditions)
                # Kept inside the limits before the next law reads them.
                # ``limited_value``, written out: this runs at every call
                for name, low, high in limited:
                    value = actual[name]
                    if value != controls[name]:
                        value = float(value)
                        # A NaN is the only value that differs from itself. An
                        # infinite one goes to the limit below
                        if value != value:  # pylint: disable=comparison-with-itself
                            raise not_finite_error(name, value, written_by(controller))
                        actual[name] = (
                            low if value < low else high if value > high else value
                        )
                for name in unlimited:
                    value = actual[name]
                    if value != controls[name] and not is_finite(value):
                        raise not_finite_error(name, value, written_by(controller))
            for controller, piece in integrated:
                rates.extend(controller.state_derivative(u[piece], view, conditions))
        # The control inputs as the laws left them, before the
        # controllables turn them into what they act with
        written = [actual[name] for name in recorded] if post_processing else ()
        for controllable, piece, inputs in acting:
            states = u[piece]
            commands = [actual[name] for name in inputs]
            for name, value in zip(
                inputs, controllable.actual_controls(states, commands, conditions)
            ):
                actual[name] = value
            rates.extend(controllable.state_derivative(states, commands, conditions))
        if post_processing:
            values = (
                equations(flight, t, own, actual, post_processing=True, **kwargs)
                if dynamics.post_process_vars
                else ()
            )
            return [*values, *written]
        return [*equations(flight, t, own, actual, **kwargs), *rates]

    derivative.__name__ = getattr(dynamics.derivative, "__name__", dynamics.name)

    def initial_state(flight, t, canonical_state, previous):
        # The phase's states, then each added one: from the previous
        # phase when it had it, else from its controllable
        starts = [
            start
            for controllable, *_ in (*laws, *acting)
            for start in controllable.initial_states()
        ]
        return [
            *dynamics.initial_state(flight, t, canonical_state, previous),
            *(previous.get(name, start) for name, start in zip(names, starts)),
        ]

    extended = _PhaseDynamics(
        dynamics.name,
        derivative,
        state_names,
        (*dynamics.post_process_vars, *(f"controls.{name}" for name in recorded)),
        dynamics.to_canonical,
        dynamics.to_canonical_dot,
        initial_state,
        # A state added here follows the control inputs even when the phase's
        # own equations read none of them
        reads_controls=dynamics.reads_controls or bool(acting or integrated),
    )
    extended._base = dynamics
    extended._takes_start = not is_canonical
    return extended


RAIL_DYNAMICS = _PhaseDynamics(
    "rail",
    udot_rail1,
    CANONICAL_STATE_NAMES,
    FULL_POST_PROCESS_VARS,
)
SOLID_PROPULSION_DYNAMICS = _PhaseDynamics(
    "solid_propulsion",
    u_dot,
    CANONICAL_STATE_NAMES,
    FULL_POST_PROCESS_VARS,
)
SIX_DOF_DYNAMICS = _PhaseDynamics(
    "six_dof",
    u_dot_generalized,
    CANONICAL_STATE_NAMES,
    FULL_POST_PROCESS_VARS,
)
THREE_DOF_DYNAMICS = _PhaseDynamics(
    "three_dof",
    u_dot_generalized_3dof,
    CANONICAL_STATE_NAMES,
    FULL_POST_PROCESS_VARS,
)
PARACHUTE_DYNAMICS = _PhaseDynamics(
    "parachute",
    u_dot_parachute,
    CANONICAL_STATE_NAMES,
    PARACHUTE_POST_PROCESS_VARS,
    # The air brakes, the control fins and the effectors do nothing under a
    # parachute, so a controller that keeps writing to them costs no solver stop
    reads_controls=False,
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
