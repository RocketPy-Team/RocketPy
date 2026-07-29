"""Container for the state history produced by a flight simulation.

A flight is integrated in phases (rail, powered ascent, coast, parachute
descent, ...), and each phase integrates only the states it needs. A parachute
descent, for example, integrates position and velocity, holding the attitude
fixed. Whatever a phase stores, the whole flight still reads as one list of
rows ``[t, *state]`` and answers queries by state name.

The full 13-state vector ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]``
is the *canonical* state: every phase can report it, filling in the states it
does not integrate. Plots, outputs and user callbacks all read that.

The pieces are:

- :class:`StateSchema` — the ordered names of the states a phase
  integrates, plus how to rebuild the full canonical state from them.
- :class:`PhaseSolution` — the rows integrated during one flight phase,
  together with that phase's schema and the quantities it reports.
- :class:`Solution` — the whole flight: a sequence of :class:`PhaseSolution`
  objects.
"""

import warnings

import numpy as np

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


def _nearest_time_index(times, t, atol, where):
    """Return the position of the stored time closest to ``t``.

    Warns when the closest stored time is further from ``t`` than ``atol``,
    naming ``where`` it was looked for.
    """
    index = int(np.argmin(np.abs(times - t)))
    if abs(times[index] - t) > atol:
        warnings.warn(
            f"Time {t} not found in {where}. Closest time is "
            f"{times[index]}. Using closest time.",
            UserWarning,
        )
    return index


class StateSchema:
    """Describes the states a flight phase integrates.

    A schema is an ordered list of state names. Position ``i`` in the name
    list is the position of that state inside every raw state row of the
    phase. The canonical schema (the full 13-value state) is the reference
    everything else is compared against.

    A phase that integrates fewer states can still report the full
    canonical state for plots, outputs and user callbacks. For each canonical
    state the phase does not integrate, the value is filled in one of two
    ways:

    - a *reconstruction* function, if one is provided for that state, that
      computes it from the phase's own states (for example, a
      parafoil phase could compute the attitude quaternion from a heading it
      integrates); or
    - the state's value at the start of the phase, held constant (the
      "freeze" fallback). This reproduces the behaviour of phases whose
      unintegrated states simply do not change.
    """

    def __init__(self, names, reconstructors=None):
        """Initialize a state schema.

        Parameters
        ----------
        names : sequence of str
            Ordered names of the states the phase integrates.
        reconstructors : dict, optional
            Maps a canonical state name to a function ``f(state_by_name) ->
            float`` that rebuilds it from the phase's own states.
            ``state_by_name`` is a dictionary of state name to value, holding
            this phase's states plus the value every other canonical
            state had at the start of the phase, so a function can read
            whatever it needs by name::

                reconstructors={
                    "e0": lambda state_by_name: math.cos(state_by_name["heading"] / 2)
                }

            Any canonical state without a reconstruction function uses the
            freeze fallback instead: it is held at the value it had when the
            phase began, unchanged for the rest of the phase.
        """
        self.names = tuple(names)
        self.index = {name: i for i, name in enumerate(self.names)}
        self.reconstructors = dict(reconstructors) if reconstructors else {}
        self.width = len(self.names)
        self.is_canonical = self.names == CANONICAL_STATE_NAMES

        # Resolve, how to fill each canonical slot: from one of this
        # schema's own states, from a reconstruction function, or by
        # freezing the value captured at the start of the phase.
        self._own_fills = []
        self._hook_fills = []
        self._freeze_fills = []
        for slot, canonical_name in enumerate(CANONICAL_STATE_NAMES):
            if canonical_name in self.index:
                self._own_fills.append((slot, self.index[canonical_name]))
            elif canonical_name in self.reconstructors:
                self._hook_fills.append((slot, self.reconstructors[canonical_name]))
            else:
                self._freeze_fills.append(slot)
        self._own_fills = tuple(self._own_fills)
        self._hook_fills = tuple(self._hook_fills)
        self._freeze_fills = tuple(self._freeze_fills)

        # Canonical states held at their value from the start of the phase.
        self.frozen_names = tuple(
            CANONICAL_STATE_NAMES[slot] for slot in self._freeze_fills
        )

    def __len__(self):
        return self.width

    def __iter__(self):
        return iter(self.names)

    def __contains__(self, name):
        return name in self.index

    def __repr__(self):
        return f"StateSchema({self.names!r})"

    def index_of(self, name):
        """Return the position of ``name`` in this schema's raw state row.

        Raises
        ------
        KeyError
            If ``name`` is not one of the states this phase integrates.
        """
        try:
            return self.index[name]
        except KeyError as error:
            raise KeyError(
                f"State '{name}' is not integrated by this flight phase. "
                f"Available states: {self.names}."
            ) from error

    def canonicalize(self, state, frozen):
        """Return the full 13-value canonical state for a raw state row.

        Parameters
        ----------
        state : sequence of float
            The phase's raw state (without time), in this schema's order.
        frozen : sequence of float or None
            The canonical state captured at the start of the phase, used to
            fill in any state this phase does not integrate.

        Returns
        -------
        list or sequence of float
            The 13-value canonical state. When this schema is already the
            canonical one, ``state`` is returned unchanged (no copy).
        """
        if self.is_canonical:
            return state
        # Frozen states keep the value they had at the start of the phase,
        # so starting from that state fills all of them at once. Only the
        # states this phase supplies are then written over it.
        if self._freeze_fills:
            result = list(frozen)
        else:
            result = [0.0] * len(CANONICAL_STATE_NAMES)
        for slot, index in self._own_fills:
            result[slot] = state[index]
        if self._hook_fills:
            state_by_name = self.reconstruction_inputs(state, frozen)
            for slot, reconstruct in self._hook_fills:
                result[slot] = reconstruct(state_by_name)
        return result

    def canonicalize_derivative(self, state_dot):
        """Return the canonical-state derivative for a raw state derivative.

        States this phase does not integrate are treated as unchanging, so
        their canonical time derivative is zero. This matches how frozen
        states behave during, for example, a parachute descent.

        Parameters
        ----------
        state_dot : sequence of float
            Time derivative of the phase's raw state, in this schema's order.

        Returns
        -------
        list or sequence of float
            The 13-value canonical derivative. When this schema is already
            the canonical one, ``state_dot`` is returned unchanged.
        """
        if self.is_canonical:
            return state_dot
        result = [0.0] * len(CANONICAL_STATE_NAMES)
        for slot, index in self._own_fills:
            result[slot] = state_dot[index]
        return result

    def reconstruction_inputs(self, state, frozen):
        """Return the state_by_name mapping a reconstruction function reads.

        Holds this phase's own states, plus the value every other canonical
        state had at the start of the phase.

        Parameters
        ----------
        state : sequence of float
            The phase's raw state (without time), in this schema's order.
        frozen : sequence of float or None
            The canonical state captured at the start of the phase.

        Returns
        -------
        dict
            State name to value.
        """
        state_by_name = (
            dict(zip(CANONICAL_STATE_NAMES, frozen)) if frozen is not None else {}
        )
        state_by_name.update(zip(self.names, state))
        return state_by_name

    def subset_from_canonical(self, canonical_state):
        """Pick this schema's states out of a full canonical state.

        Used to seed a new flight phase from the state that ended the previous
        one. Every state in this schema must be a canonical state for the
        default mapping to work; phases with states that are not canonical
        (such as a heading) provide their own seeding rule instead.

        Parameters
        ----------
        canonical_state : sequence of float
            The full 13-value canonical state.

        Returns
        -------
        list of float
            The phase's raw state, in this schema's order.
        """
        if self.is_canonical:
            return list(canonical_state)
        return [canonical_state[CANONICAL_INDEX[name]] for name in self.names]

    def select_atol(self, atol):
        """Map an absolute-tolerance setting onto this schema's states.

        The flight's ``atol`` is either a single number or a 13-value vector in
        canonical order. This returns the tolerance the solver should use for a
        phase that integrates only this schema's states.

        Parameters
        ----------
        atol : float or sequence of float
            A single tolerance for all states, or one tolerance per
            canonical state (13 values).

        Returns
        -------
        float or list of float
            A single number is returned unchanged. A 13-value canonical vector
            is reduced to this schema's states (states that are not
            canonical use the largest supplied tolerance).

        Raises
        ------
        ValueError
            If a per-state tolerance is given whose length is neither 13 nor
            this schema's width.
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
                for name in self.names
            ]
        if len(atol) == self.width:
            return atol
        raise ValueError(
            f"atol vector has length {len(atol)}, which matches neither the "
            f"canonical state (13) nor this flight phase ({self.width})."
        )

    def to_dict(self):
        """Return a serializable description of this schema."""
        return {"state_names": list(self.names)}

    @classmethod
    def from_names(cls, names):
        """Rebuild a schema from a list of state names.

        If the names match a schema RocketPy already knows about (such as the
        canonical or parachute schema), that shared schema — including its
        reconstruction functions — is returned. Otherwise a plain schema with
        the given names is created.
        """
        names = tuple(names)
        registered = _SCHEMA_REGISTRY.get(names)
        if registered is not None:
            return registered
        return cls(names)


CANONICAL_SCHEMA = StateSchema(CANONICAL_STATE_NAMES)
"""The full 13-value flight state schema."""

PARACHUTE_3T_SCHEMA = StateSchema(("x", "y", "z", "vx", "vy", "vz"))
"""Schema for a 3-DOF translational parachute descent (position and velocity)."""

# A saved flight stores only its state names, because reconstruction functions
# are code and cannot be written to a file. Looking the names up here is what
# gives a schema its functions back when the flight is loaded. Neither built-in
# schema has any, so this only matters for a phase that defines its own.
_SCHEMA_REGISTRY = {
    CANONICAL_SCHEMA.names: CANONICAL_SCHEMA,
    PARACHUTE_3T_SCHEMA.names: PARACHUTE_3T_SCHEMA,
}


def register_schema(schema):
    """Register a schema so its reconstruction functions survive a save/load.

    A phase whose schema has no reconstruction functions does not need this.

    Parameters
    ----------
    schema : StateSchema
        The schema to register, keyed by its state names.

    Returns
    -------
    StateSchema
        The schema that was registered, so this can wrap the definition.
    """
    _SCHEMA_REGISTRY[schema.names] = schema
    return schema


class DerivedQuantity:
    """A quantity a flight phase reports alongside the states it integrates.

    Derived quantities are the things a phase computes on its way to the state
    derivative but does not integrate: accelerations, aerodynamic forces and
    moments, net thrust. Registering one tells RocketPy how to label it and,
    more importantly, what to do about the flight phases that do not report it.

    Parameters
    ----------
    name : str
        The quantity's name, used to look it up (for example ``"ax"``).
    label : str, optional
        Axis label used when the quantity is plotted. Defaults to ``name``.
    unit : str, optional
        SI unit shown on the axis label, for example ``"m/s^2"``.
    absent : str, optional
        What to report for the flight phases that do not compute this quantity.
        Either:

        - ``"zero"`` — report zero for the whole phase. Right for a quantity
          that is genuinely zero when a phase does not compute it, such as the
          aerodynamic moments during a parachute descent.
        - ``"omit"`` — report nothing for that phase, leaving a gap in the
          quantity's time history. Right for anything else, since it does not
          invent values that were never computed.

        Default is ``"omit"``, so an unregistered quantity never gets made-up
        numbers mixed into its history.
    interpolation : str, optional
        How values between stored times are computed, for example ``"spline"``
        or ``"linear"``. Default is ``"spline"``.
    extrapolation : str, optional
        How values outside the stored time range are computed, for example
        ``"zero"`` or ``"constant"``. Default is ``"zero"``, which reports zero
        before the first and after the last time the quantity was computed.
    """

    __slots__ = ("name", "label", "unit", "absent", "interpolation", "extrapolation")

    def __init__(
        self,
        name,
        label=None,
        unit=None,
        absent="omit",
        interpolation="spline",
        extrapolation="zero",
    ):
        if absent not in ("zero", "omit"):
            raise ValueError(
                f"absent must be 'zero' or 'omit', not {absent!r}. Use 'zero' "
                f"when a phase that does not compute '{name}' really has it at "
                f"zero, and 'omit' to leave a gap instead."
            )
        self.name = name
        self.label = label or name
        self.unit = unit
        self.absent = absent
        self.interpolation = interpolation
        self.extrapolation = extrapolation

    @property
    def axis_label(self):
        """The quantity's label with its unit, for plotting."""
        return f"{self.label} ({self.unit})" if self.unit else self.label

    def __repr__(self):
        return f"DerivedQuantity({self.name!r}, absent={self.absent!r})"


# Derived quantities RocketPy knows about, by name. A quantity that is not
# registered still works; it just falls back to the "omit" default.
_DERIVED_REGISTRY = {}


def register_derived_quantity(quantity):
    """Register a derived quantity so RocketPy knows how to report it.

    Parameters
    ----------
    quantity : DerivedQuantity
        The quantity to register. Registering the same name twice replaces the
        earlier entry.

    Returns
    -------
    DerivedQuantity
        The quantity that was registered.
    """
    _DERIVED_REGISTRY[quantity.name] = quantity
    return quantity


def get_derived_quantity(name):
    """Return the registered :class:`DerivedQuantity`, or a default for ``name``.

    An unregistered name gets a default quantity that is omitted from the
    phases that do not report it, rather than being reported as zero.
    """
    quantity = _DERIVED_REGISTRY.get(name)
    if quantity is None:
        return DerivedQuantity(name)
    return quantity


class PhaseSolution:
    """The rows integrated during a single flight phase."""

    def __init__(
        self,
        schema,
        start_canonical,
        t_start=None,
        dynamics=None,
        name=None,
        derived_names=None,
    ):
        """Initialize a solution phase.

        Parameters
        ----------
        schema : StateSchema
            The states this phase integrates.
        start_canonical : sequence of float or None
            The full 13-value canonical state at the start of the phase. It
            supplies every canonical state this phase does not integrate,
            which are then held at that value for the whole phase. Only a
            phase that supplies all of them itself may pass ``None``.
        t_start : float, optional
            The time the phase begins.
        dynamics : optional
            The bound dynamics driving this phase, used to recompute
            accelerations and forces after the flight. A solution loaded from
            a file has none, since that requires a live flight.
        name : str, optional
            The phase name, for reference in output and debugging.
        derived_names : sequence of str, optional
            Ordered names of the quantities this phase reports besides the
            states it integrates, such as ``("ax", "ay", "az")``. The order
            fixes which quantity sits at which position in the phase's recorded
            rows. Taken from ``dynamics`` when it is given; a solution loaded
            from a file supplies it directly, since it has no live dynamics.

        Raises
        ------
        ValueError
            If ``start_canonical`` is ``None`` but this phase needs it to fill
            in the canonical states it does not integrate.
        """
        if start_canonical is None and schema.frozen_names:
            raise ValueError(
                f"This flight phase does not integrate "
                f"{', '.join(schema.frozen_names)}, so it needs the state at the "
                f"start of the phase to hold them at. Pass start_canonical."
            )
        self.schema = schema
        self.rows = []
        self.t_start = t_start
        self.start_canonical = (
            tuple(start_canonical) if start_canonical is not None else None
        )
        self.dynamics = dynamics
        self.name = name
        if derived_names is None:
            derived_names = dynamics.derived_names if dynamics is not None else ()
        self.derived_names = tuple(derived_names)
        self._derived_index = {name: i for i, name in enumerate(self.derived_names)}
        self.derived_rows = []
        self._array = None
        self._array_len = -1
        self._canonical_array = None
        self._canonical_array_len = -1

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.series(key)
        if isinstance(key, (slice, int, np.integer)):
            return self.rows[key]
        raise TypeError(
            "Flight phase indices must be integers, slices, or state names, "
            f"not {type(key).__name__}."
        )

    @property
    def width(self):
        """Number of states integrated in this phase."""
        return self.schema.width

    @property
    def array(self):
        """The phase's rows as a 2-D array ``[t, *state]`` per row."""
        if self._array is None or self._array_len != len(self.rows):
            if self.rows:
                self._array = np.array(self.rows)
            else:
                self._array = np.empty((0, self.width + 1))
            self._array_len = len(self.rows)
        return self._array

    @property
    def time(self):
        """The time column of this phase as a 1-D array."""
        return self.array[:, 0]

    @property
    def canonical_array(self):
        """This phase as a rectangular ``(n, 14)`` canonical table.

        Each row is ``[t, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]``.
        States this phase does not integrate are reconstructed or held at
        their value when the phase began.
        """
        if self._canonical_array is None or self._canonical_array_len != len(self.rows):
            if self.schema.is_canonical:
                self._canonical_array = self.array
            else:
                rows = [[row[0], *self.canonical_state(row[1:])] for row in self.rows]
                width = len(CANONICAL_STATE_NAMES) + 1
                self._canonical_array = np.array(rows) if rows else np.empty((0, width))
            self._canonical_array_len = len(self.rows)
        return self._canonical_array

    def invalidate(self):
        """Drop the cached arrays after an in-place edit to ``rows``."""
        self._array = None
        self._array_len = -1
        self._canonical_array = None
        self._canonical_array_len = -1

    def canonical_state(self, state):
        """Return the full canonical state for a raw state of this phase."""
        return self.schema.canonicalize(state, self.start_canonical)

    def state_dict(self, state):
        """Return a raw state of this phase as a state name to value map.

        Covers the 13 canonical states (reconstructed or frozen where this
        phase does not integrate them) plus any state this phase integrates
        that is not one of them.
        """
        values = dict(zip(CANONICAL_STATE_NAMES, self.canonical_state(state)))
        values.update(zip(self.schema.names, state))
        return values

    def series(self, name):
        """Return this phase's ``[t, value]`` history for one state.

        Parameters
        ----------
        name : str
            The state name (for example ``"vz"``).

        Returns
        -------
        numpy.ndarray
            An ``(n, 2)`` array whose columns are time and value.

        Raises
        ------
        KeyError
            If this phase does not define the state, or has no stored
            states yet.
        """
        result = self._series_or_none(name)
        if result is None:
            if not self.rows:
                raise KeyError("This flight phase has no stored states yet.")
            raise KeyError(
                f"State '{name}' is not defined in this flight phase. "
                f"It integrates {', '.join(self.schema.names)}."
            )
        return result

    def _series_or_none(self, name):
        """Return the ``[t, value]`` history, or ``None`` if undefined here.

        Lets :class:`Solution` skip the phases that do not define a state
        while gathering its history across the whole flight.
        """
        if not self.rows:
            return None
        array = self.array
        # A state this phase integrates is a plain column of its rows;
        # anything else is reconstructed or held at its start-of-phase value.
        index = self.schema.index.get(name)
        if index is not None:
            return array[:, [0, index + 1]]
        if name in self.schema.reconstructors:
            reconstruct = self.schema.reconstructors[name]
            values = [
                reconstruct(
                    self.schema.reconstruction_inputs(row[1:], self.start_canonical)
                )
                for row in self.rows
            ]
            return np.column_stack([array[:, 0], values])
        if name in CANONICAL_INDEX and self.start_canonical is not None:
            constant = self.start_canonical[CANONICAL_INDEX[name]]
            return np.column_stack([array[:, 0], np.full(len(self.rows), constant)])
        return None

    def at(self, t, atol=1e-3):
        """Return this phase's state at the stored time nearest ``t``.

        Parameters
        ----------
        t : float
            Time in seconds.
        atol : float, optional
            If the nearest stored time differs from ``t`` by more than this, a
            warning is raised. Default is ``1e-3``.

        Returns
        -------
        dict
            State name to value, as described in :meth:`state_dict`.
        """
        index = _nearest_time_index(self.time, t, atol, "this flight phase")
        return self.at_index(index)

    def at_index(self, index):
        """Return the state stored at row ``index`` as a name to value map.

        Reads a single row, so it stays cheap no matter how long the phase is.
        Use it instead of ``phase["vz"][index, 1]`` when only one row is
        needed, since that builds the state's whole history first.
        """
        return self.state_dict(self.rows[index][1:])

    def canonical_row(self, index):
        """Return ``[t, *canonical_state]`` for the raw row at ``index``."""
        row = self.rows[index]
        return [row[0], *self.canonical_state(row[1:])]

    def canonical_states(self, stop=None):
        """Return canonical states (without time) for every stored row.

        Parameters
        ----------
        stop : int, optional
            If given, only the rows up to ``stop`` (Python slice semantics) are
            returned.
        """
        rows = self.rows if stop is None else self.rows[:stop]
        return [self.canonical_state(row[1:]) for row in rows]

    def derived_index_of(self, name):
        """Return the position of ``name`` in this phase's derived rows.

        Raises
        ------
        KeyError
            If this phase does not report the quantity.
        """
        try:
            return self._derived_index[name]
        except KeyError as error:
            raise KeyError(
                f"Derived quantity '{name}' is not reported by this flight "
                f"phase. It reports: {', '.join(self.derived_names)}."
            ) from error

    def record_derived(self, t, values):
        """Store this phase's derived quantities at time ``t``.

        Parameters
        ----------
        t : float
            The time the values were computed at, in seconds.
        values : sequence of float or dict
            Either the quantity values in this phase's ``derived_names`` order,
            or a dictionary of quantity name to value. A dictionary may leave
            out quantities, which are then recorded as zero.

        Raises
        ------
        ValueError
            If a sequence is given whose length is not the number of quantities
            this phase reports.
        KeyError
            If a dictionary names a quantity this phase does not report.
        """
        width = len(self.derived_names)
        if isinstance(values, dict):
            row = [t] + [0.0] * width
            for name, value in values.items():
                row[self.derived_index_of(name) + 1] = value
        else:
            values = list(values)
            if len(values) != width:
                raise ValueError(
                    f"This flight phase reports {width} derived quantities "
                    f"({', '.join(self.derived_names)}), but {len(values)} "
                    f"values were recorded. The values must be in that same "
                    f"order."
                )
            row = [t, *values]
        self.derived_rows.append(row)

    def to_dict(self):
        """Return a serializable description of this phase."""
        return {
            "name": self.name,
            "dynamics": self.dynamics.key if self.dynamics is not None else None,
            "state_names": list(self.schema.names),
            "t_start": self.t_start,
            "start_canonical": (
                list(self.start_canonical) if self.start_canonical is not None else None
            ),
            "rows": self.rows,
            # Derived quantities are stored with the phase so a loaded flight
            # reports them without having to re-run the simulation.
            "derived_names": list(self.derived_names),
            "derived_rows": self.derived_rows,
        }

    @classmethod
    def from_dict(cls, data):
        """Rebuild a phase from its serialized description."""
        schema = StateSchema.from_names(data["state_names"])
        phase = cls(
            schema,
            data.get("start_canonical"),
            t_start=data.get("t_start"),
            name=data.get("name"),
            derived_names=data.get("derived_names", ()),
        )
        phase.rows = [list(row) for row in data.get("rows", [])]
        phase.derived_rows = [list(row) for row in data.get("derived_rows", [])]
        return phase


class Solution:
    """The state history of a whole flight.

    A ``Solution`` is a list of rows: it supports ``len()``, iteration, integer
    indexing (``solution[-1]``), slicing, ``append``, ``insert`` and ``pop``.
    Every row is ``[t, *state]``, and its length may differ between flight
    phases that integrate different states.

    It also answers queries by state name. ``solution["vz"]`` returns the
    ``[t, value]`` history of the vertical velocity across the whole flight;
    ``solution.at(t)`` returns the whole state at the nearest stored time as a
    name-to-value dictionary.
    """

    def __init__(self, phases=None):
        """Initialize a solution.

        Parameters
        ----------
        phases : sequence of PhaseSolution, optional
            Pre-built phases, in flight order. A new flight starts empty and
            adds one as each of its flight phases begins.
        """
        self._phases = list(phases) if phases else []
        self._length = sum(len(phase.rows) for phase in self._phases)
        self._version = 0
        self._series_cache = {}
        self._canonical_cache = None
        self._canonical_version = -1
        self._time_cache = None
        self._time_version = -1
        # Canonical states of the leading rows, built up as rows are appended.
        # Its length is how many rows are still known good: edits to a row drop
        # this list back to that row, so only the tail is ever rebuilt.
        self._canonical_states_cache = []

    # -- Phase management ------------------------------------------------

    def start_phase(
        self,
        schema,
        start_canonical,
        t_start=None,
        dynamics=None,
        name=None,
    ):
        """Begin a new flight phase and return its (empty) PhaseSolution.

        See :class:`PhaseSolution` for what each argument means.
        """
        phase = PhaseSolution(
            schema,
            start_canonical,
            t_start=t_start,
            dynamics=dynamics,
            name=name,
        )
        self._phases.append(phase)
        self._version += 1
        return phase

    @property
    def phases(self):
        """This flight's phases, in the order they were flown."""
        return tuple(self._phases)

    @property
    def tail(self):
        """The current (most recent) phase."""
        return self._phases[-1]

    # -- List-like behaviour ----------------------------------------------

    def _locate(self, index):
        """Return the ``(phase, local_index)`` owning a global row index."""
        total = self._length
        if index < 0:
            index += total
        if index < 0 or index >= total:
            raise IndexError("solution row index out of range")
        # Walk in from whichever end is closer. The simulation reads the last
        # rows constantly (``solution[-1]``, ``solution[-2]``), so starting at
        # the front would rescan every earlier phase each time.
        if 2 * index >= total:
            remaining = total - index
            for phase in reversed(self._phases):
                count = len(phase.rows)
                if remaining <= count:
                    return phase, count - remaining
                remaining -= count
        else:
            for phase in self._phases:
                count = len(phase.rows)
                if index < count:
                    return phase, index
                index -= count
        raise IndexError("solution row index out of range")

    def _invalidate_canonical_states_from(self, index):
        """Drop cached canonical states from row ``index`` onward.

        Rows before the edit are unaffected, so they stay cached.
        """
        if index < 0:
            index += self._length
        if index < len(self._canonical_states_cache):
            del self._canonical_states_cache[max(index, 0) :]

    def _iter_rows(self):
        for phase in self._phases:
            yield from phase.rows

    def __len__(self):
        return self._length

    def __iter__(self):
        return self._iter_rows()

    def append(self, row):
        """Append a raw state row ``[t, *state]`` to the current phase."""
        phase = self._phases[-1]
        if len(row) != phase.width + 1:
            raise ValueError(
                f"State row of length {len(row)} does not match the current "
                f"flight phase (expected {phase.width + 1} values: time plus "
                f"{phase.width} states)."
            )
        phase.rows.append(row)
        self._length += 1
        self._version += 1

    def __iadd__(self, rows):
        for row in rows:
            self.append(row)
        return self

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.series(key)
        if isinstance(key, slice):
            return list(self._iter_rows())[key]
        if isinstance(key, (int, np.integer)):
            phase, local = self._locate(int(key))
            return phase.rows[local]
        raise TypeError(
            "Solution indices must be integers, slices, or state names, "
            f"not {type(key).__name__}."
        )

    def __setitem__(self, index, row):
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Solution only supports integer row assignment.")
        phase, local = self._locate(int(index))
        if len(row) != phase.width + 1:
            raise ValueError(
                f"State row of length {len(row)} does not match the flight "
                f"phase it is being written to (expected {phase.width + 1})."
            )
        phase.rows[local] = row
        phase.invalidate()
        self._invalidate_canonical_states_from(int(index))
        self._version += 1

    def insert(self, index, row):
        """Insert a raw state row before the row at ``index``."""
        phase, local = self._locate(int(index))
        if len(row) != phase.width + 1:
            raise ValueError(
                f"State row of length {len(row)} does not match the flight "
                f"phase it is being inserted into (expected {phase.width + 1})."
            )
        phase.rows.insert(local, row)
        phase.invalidate()
        self._invalidate_canonical_states_from(int(index))
        self._length += 1
        self._version += 1

    def pop(self, index=-1):
        """Remove and return the raw state row at ``index``."""
        phase, local = self._locate(int(index))
        row = phase.rows.pop(local)
        phase.invalidate()
        self._invalidate_canonical_states_from(int(index))
        self._length -= 1
        self._version += 1
        return row

    def __array__(self, dtype=None, copy=None):  # pylint: disable=unused-argument
        # `copy` is part of the NumPy array protocol; the array is always freshly
        # built here, so it is accepted for compatibility and not acted on.
        widths = {phase.width for phase in self._phases if phase.rows}
        if len(widths) > 1:
            raise TypeError(
                "This flight has phases that integrate different state "
                "states, so its solution cannot be turned into a single "
                "rectangular array. Use flight.solution['name'] for one "
                "state's time history, flight.solution.canonical_array for "
                "the full 14-column table, or flight.solution.phases[i].array "
                "for a single phase."
            )
        rows = list(self._iter_rows())
        return np.array(rows, dtype=dtype)

    # -- Queries by name ---------------------------------------------------

    def series(self, name):
        """Return the ``[t, value]`` history of one state over the flight.

        The history spans only the phases that define the state. States
        in the canonical state are defined in every phase (reconstructed or
        frozen where a phase does not integrate them), so their history covers
        the whole flight. A phase-specific state's history covers only the
        phases that integrate it.

        Parameters
        ----------
        name : str
            The state name (for example ``"vz"``).

        Returns
        -------
        numpy.ndarray
            An ``(n, 2)`` array whose columns are time and value.

        Raises
        ------
        KeyError
            If no phase defines the state.
        """
        cached = self._series_cache.get(name)
        if cached is not None and cached[0] == self._version:
            return cached[1]
        parts = []
        for phase in self._phases:
            part = phase._series_or_none(name)  # noqa: SLF001
            if part is not None and len(part):
                parts.append(part)
        if not parts:
            raise KeyError(
                f"State '{name}' is not defined in any flight phase of this solution."
            )
        result = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        self._series_cache[name] = (self._version, result)
        return result

    def at(self, t, atol=1e-3):
        """Return the flight state at the stored time nearest ``t``.

        Parameters
        ----------
        t : float
            Time in seconds.
        atol : float, optional
            If the nearest stored time differs from ``t`` by more than this, a
            warning is raised. Default is ``1e-3``.

        Returns
        -------
        dict
            State name to value, for example ``state["vz"]``. Holds the 13
            canonical states (reconstructed or held at their value when the
            phase began, where the phase does not integrate them) plus any
            state the phase integrates that is not one of them.
        """
        index = _nearest_time_index(self.time, t, atol, "solution")
        return self.at_index(index)

    def at_index(self, index):
        """Return the state stored at row ``index`` as a name to value map.

        Reads a single row, so it stays cheap no matter how long the flight is.
        Use it instead of ``solution["vz"][index, 1]`` when only one row is
        needed, since that builds the state's whole history first.
        """
        phase, local = self._locate(int(index))
        return phase.at_index(local)

    def canonical_row(self, index):
        """Return ``[t, *canonical_state]`` for the raw row at ``index``."""
        phase, local = self._locate(int(index))
        return phase.canonical_row(local)

    def canonical_states(self, stop=None):
        """Return canonical states (without time) for every stored row.

        Parameters
        ----------
        stop : int, optional
            If given, only the rows up to ``stop`` (Python slice semantics) are
            returned. Useful for building a history that excludes the most
            recent rows.
        """
        cache = self._canonical_states_cache
        # Rows already converted by an earlier call are reused; only rows added
        # since then are converted now. Editing a row drops the cache back to
        # it, so anything stale has already been removed.
        if len(cache) > self._length:
            del cache[self._length :]
        if len(cache) < self._length:
            pending = len(cache)
            for phase in self._phases:
                count = len(phase.rows)
                if pending >= count:
                    pending -= count
                    continue
                for row in phase.rows[pending:]:
                    cache.append(phase.canonical_state(row[1:]))
                pending = 0
        if stop is not None:
            return cache[:stop]
        return list(cache)

    @property
    def time(self):
        """The time column of the whole flight as a 1-D array."""
        # Each phase keeps its own time column, so only the phase that just
        # gained rows rebuilds one; the rest are reused as they are.
        if self._time_cache is None or self._time_version != self._version:
            parts = [phase.time for phase in self._phases if phase.rows]
            self._time_cache = np.concatenate(parts) if parts else np.empty(0)
            self._time_version = self._version
        return self._time_cache

    @property
    def canonical_array(self):
        """The whole flight as a rectangular ``(n, 14)`` canonical table.

        Each row is ``[t, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]``.
        States a phase does not integrate are reconstructed or frozen.
        """
        if self._canonical_cache is None or self._canonical_version != self._version:
            parts = [phase.canonical_array for phase in self._phases if phase.rows]
            width = len(CANONICAL_STATE_NAMES) + 1
            self._canonical_cache = np.vstack(parts) if parts else np.empty((0, width))
            self._canonical_version = self._version
        return self._canonical_cache

    # -- Serialization -----------------------------------------------------

    def to_dict(self):
        """Return a serializable description of the whole solution."""
        return {
            "format": "rocketpy/solution",
            "version": 1,
            "phases": [phase.to_dict() for phase in self._phases],
        }

    @classmethod
    def from_dict(cls, data):
        """Rebuild a solution from its serialized (phase-based) form."""
        phases = [PhaseSolution.from_dict(phase) for phase in data["phases"]]
        return cls(phases)

    @classmethod
    def from_legacy_list(cls, rows):
        """Rebuild a solution from the old flat list of canonical rows.

        Older saved flights stored the solution as a single list of
        14-value rows. This wraps that list as one canonical phase so old
        files keep loading.
        """
        rows = [list(row) for row in rows]
        phase = PhaseSolution(
            CANONICAL_SCHEMA,
            tuple(rows[0][1:]) if rows else None,
            t_start=rows[0][0] if rows else None,
        )
        phase.rows = rows
        return cls([phase])
