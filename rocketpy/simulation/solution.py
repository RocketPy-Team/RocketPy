"""Container for the state history produced by a flight simulation.

A flight is integrated in phases (rail, powered ascent, coast, parachute
descent, ...). Historically every phase advanced the same 13 state variables
``[x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]``, so the whole flight
could be stored as a single rectangular table of rows ``[t, *state]``.

Some flight phases only need a subset of those variables. A parachute descent,
for example, integrates only position and velocity; its attitude is held
fixed. The :class:`Solution` container lets each phase store exactly the
variables it integrates while still presenting the familiar, list-like view of
the full flight to the rest of the library and to the user.

The pieces are:

- :class:`StateSchema` — the ordered names of the variables a phase
  integrates, plus how to rebuild the full canonical state from them.
- :class:`SolutionSegment` — the rows integrated during one flight phase,
  together with that phase's schema.
- :class:`StateView` — a lightweight, read-by-name wrapper over a single raw
  state row.
- :class:`Solution` — the whole flight: a sequence of segments that behaves
  like the old list of rows and also answers queries by variable name.
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
"""Names of the 13 variables in RocketPy's full flight state, in order."""

CANONICAL_INDEX = {name: index for index, name in enumerate(CANONICAL_STATE_NAMES)}


class StateSchema:
    """Describes the state variables a flight phase integrates.

    A schema is an ordered list of variable names. Position ``i`` in the name
    list is the position of that variable inside every raw state row of the
    phase. The canonical schema (the full 13-variable state) is the reference
    everything else is compared against.

    A phase that integrates fewer variables can still report the full
    canonical state for plots, outputs and user callbacks. For each canonical
    variable the phase does not integrate, the value is filled in one of two
    ways:

    - a *reconstruction* function, if one is provided for that variable, that
      computes it from the phase's own variables (for example, a future
      parafoil phase could compute the attitude quaternion from a heading it
      integrates); or
    - the variable's value at the start of the phase, held constant (the
      "freeze" fallback). This reproduces the behaviour of phases whose
      unintegrated variables simply do not change.

    Parameters
    ----------
    names : sequence of str
        Ordered names of the variables the phase integrates.
    reconstructors : dict, optional
        Maps a canonical variable name to a function ``f(state_view) -> float``
        that rebuilds it from the phase's own variables. Any canonical variable
        without a reconstruction function uses the freeze fallback.
    """

    def __init__(self, names, reconstructors=None):
        self.names = tuple(names)
        self.index = {name: i for i, name in enumerate(self.names)}
        self.reconstructors = dict(reconstructors) if reconstructors else {}
        self.width = len(self.names)
        self.is_canonical = self.names == CANONICAL_STATE_NAMES

        # Resolve, once, how to fill each canonical slot: from one of this
        # schema's own variables, from a reconstruction function, or by
        # freezing the value captured at the start of the phase.
        self._canonical_plan = []
        for slot, canonical_name in enumerate(CANONICAL_STATE_NAMES):
            if canonical_name in self.index:
                self._canonical_plan.append(("own", self.index[canonical_name]))
            elif canonical_name in self.reconstructors:
                self._canonical_plan.append(
                    ("hook", self.reconstructors[canonical_name])
                )
            else:
                self._canonical_plan.append(("freeze", slot))

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
            If ``name`` is not one of the variables this phase integrates.
        """
        try:
            return self.index[name]
        except KeyError as error:
            raise KeyError(
                f"State '{name}' is not integrated by this flight phase. "
                f"Available states: {self.names}."
            ) from error

    def canonicalize(self, state, frozen):
        """Return the full 13-variable canonical state for a raw state row.

        Parameters
        ----------
        state : sequence of float
            The phase's raw state (without time), in this schema's order.
        frozen : sequence of float or None
            The canonical state captured at the start of the phase, used to
            fill in any variable this phase does not integrate.

        Returns
        -------
        list or sequence of float
            The 13-variable canonical state. When this schema is already the
            canonical one, ``state`` is returned unchanged (no copy).
        """
        if self.is_canonical:
            return state
        result = [0.0] * len(CANONICAL_STATE_NAMES)
        view = None
        for slot, (kind, ref) in enumerate(self._canonical_plan):
            if kind == "own":
                result[slot] = state[ref]
            elif kind == "freeze":
                result[slot] = frozen[slot]
            else:  # reconstruction function
                if view is None:
                    view = StateView(state, self, frozen)
                result[slot] = ref(view)
        return result

    def canonicalize_derivative(self, state_dot):
        """Return the canonical-state derivative for a raw state derivative.

        Variables this phase does not integrate are treated as unchanging, so
        their canonical time derivative is zero. This matches how frozen
        variables behave during, for example, a parachute descent.

        Parameters
        ----------
        state_dot : sequence of float
            Time derivative of the phase's raw state, in this schema's order.

        Returns
        -------
        list or sequence of float
            The 13-variable canonical derivative. When this schema is already
            the canonical one, ``state_dot`` is returned unchanged.
        """
        if self.is_canonical:
            return state_dot
        result = [0.0] * len(CANONICAL_STATE_NAMES)
        for slot, (kind, ref) in enumerate(self._canonical_plan):
            if kind == "own":
                result[slot] = state_dot[ref]
        return result

    def subset_from_canonical(self, canonical_state):
        """Pick this schema's variables out of a full canonical state.

        Used to seed a new flight phase from the state that ended the previous
        one. Every variable in this schema must be a canonical variable for the
        default mapping to work; phases with variables that are not canonical
        (such as a heading) provide their own seeding rule instead.

        Parameters
        ----------
        canonical_state : sequence of float
            The full 13-variable canonical state.

        Returns
        -------
        list of float
            The phase's raw state, in this schema's order.
        """
        if self.is_canonical:
            return list(canonical_state)
        return [canonical_state[CANONICAL_INDEX[name]] for name in self.names]

    def select_atol(self, atol):
        """Map an absolute-tolerance setting onto this schema's variables.

        The flight's ``atol`` is either a single number or a 13-value vector in
        canonical order. This returns the tolerance the solver should use for a
        phase that integrates only this schema's variables.

        Parameters
        ----------
        atol : float or sequence of float
            A single tolerance for all variables, or one tolerance per
            canonical variable (13 values).

        Returns
        -------
        float or list of float
            A single number is returned unchanged. A 13-value canonical vector
            is reduced to this schema's variables (variables that are not
            canonical use the largest supplied tolerance).

        Raises
        ------
        ValueError
            If a per-variable tolerance is given whose length is neither 13 nor
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
        """Rebuild a schema from a list of variable names.

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


class StateView:
    """Read-by-name access to a single raw state row.

    Wraps one phase's raw state (without time) together with the schema that
    describes it, so callers can read variables by name (``view["vz"]`` or
    ``view.vz``) instead of by position. Variables the phase does not integrate
    are reconstructed or frozen, exactly as :meth:`StateSchema.canonicalize`
    does. Integer indexing (``view[3]``) returns the raw value at that
    position.
    """

    __slots__ = ("raw", "schema", "frozen", "_canonical")

    def __init__(self, raw, schema, frozen=None):
        self.raw = raw
        self.schema = schema
        self.frozen = frozen
        self._canonical = None

    def __getitem__(self, key):
        if isinstance(key, str):
            index = self.schema.index.get(key)
            if index is not None:
                return self.raw[index]
            if key in self.schema.reconstructors:
                return self.schema.reconstructors[key](self)
            if key in CANONICAL_INDEX and self.frozen is not None:
                return self.frozen[CANONICAL_INDEX[key]]
            raise KeyError(f"State '{key}' is not available in this flight phase.")
        return self.raw[key]

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    @property
    def canonical(self):
        """The full 13-variable canonical state for this row."""
        if self._canonical is None:
            self._canonical = self.schema.canonicalize(self.raw, self.frozen)
        return self._canonical


class SolutionSegment:
    """The rows integrated during a single flight phase.

    Parameters
    ----------
    schema : StateSchema
        The variables this phase integrates.
    t_start : float, optional
        The time the phase begins.
    start_canonical : sequence of float, optional
        The full canonical state at the start of the phase, used to freeze the
        variables this phase does not integrate.
    dynamics : optional
        The bound dynamics driving this phase. Kept for post-processing replay;
        not required once the simulation is complete.
    name : str, optional
        The phase name, for reference in output and debugging.
    parachute : optional
        The active parachute during this phase, if any.
    """

    def __init__(
        self,
        schema,
        t_start=None,
        start_canonical=None,
        dynamics=None,
        name=None,
        parachute=None,
    ):
        self.schema = schema
        self.rows = []
        self.t_start = t_start
        self.start_canonical = (
            tuple(start_canonical) if start_canonical is not None else None
        )
        self.dynamics = dynamics
        self.parachute = parachute
        self.name = name
        self.derived_rows = []
        self._array = None
        self._array_len = -1

    @property
    def width(self):
        """Number of state variables integrated in this phase."""
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

    def invalidate(self):
        """Drop the cached array after an in-place edit to ``rows``."""
        self._array = None
        self._array_len = -1

    def canonical_state(self, state):
        """Return the full canonical state for a raw state of this phase."""
        return self.schema.canonicalize(state, self.start_canonical)

    def view(self, state):
        """Return a :class:`StateView` over a raw state of this phase."""
        return StateView(state, self.schema, self.start_canonical)

    def series(self, name):
        """Return this phase's ``[t, value]`` history for one variable.

        Returns ``None`` when the variable is neither integrated nor
        reconstructable nor freezable in this phase, so the caller can skip
        phases that do not define it.
        """
        if not self.rows:
            return None
        array = self.array
        index = self.schema.index.get(name)
        if index is not None:
            return array[:, [0, index + 1]]
        if name in self.schema.reconstructors:
            reconstruct = self.schema.reconstructors[name]
            values = [reconstruct(self.view(row[1:])) for row in self.rows]
            return np.column_stack([array[:, 0], values])
        if name in CANONICAL_INDEX and self.start_canonical is not None:
            constant = self.start_canonical[CANONICAL_INDEX[name]]
            return np.column_stack([array[:, 0], np.full(len(self.rows), constant)])
        return None

    def to_dict(self):
        """Return a serializable description of this segment."""
        return {
            "name": self.name,
            "dynamics": self.dynamics.key if self.dynamics is not None else None,
            "state_names": list(self.schema.names),
            "t_start": self.t_start,
            "start_canonical": (
                list(self.start_canonical) if self.start_canonical is not None else None
            ),
            "rows": self.rows,
        }

    @classmethod
    def from_dict(cls, data):
        """Rebuild a segment from its serialized description."""
        schema = StateSchema.from_names(data["state_names"])
        segment = cls(
            schema,
            t_start=data.get("t_start"),
            start_canonical=data.get("start_canonical"),
            name=data.get("name"),
        )
        segment.rows = [list(row) for row in data.get("rows", [])]
        return segment


class Solution:
    """The state history of a whole flight.

    A ``Solution`` behaves like the list of rows RocketPy has always used: it
    supports ``len()``, iteration, integer indexing (``solution[-1]``),
    slicing, ``append``, ``insert`` and ``pop``. Every row is
    ``[t, *state]``, and its length may differ between flight phases that
    integrate different variables.

    On top of that it answers queries by variable name. ``solution["vz"]``
    returns the ``[t, value]`` history of the vertical velocity across the
    whole flight; ``solution.at(t)`` returns a :class:`StateView` at the
    nearest stored time.

    Parameters
    ----------
    segments : sequence of SolutionSegment, optional
        Pre-built segments. New flights start empty and add segments as phases
        begin.
    """

    def __init__(self, segments=None):
        self._segments = list(segments) if segments else []
        self._length = sum(len(segment.rows) for segment in self._segments)
        self._version = 0
        self._series_cache = {}
        self._canonical_cache = None
        self._canonical_version = -1
        self._time_cache = None
        self._time_version = -1

    # -- Segment management ------------------------------------------------

    def start_segment(
        self,
        schema,
        t_start=None,
        start_canonical=None,
        dynamics=None,
        name=None,
        parachute=None,
    ):
        """Begin a new flight phase and return its (empty) segment."""
        segment = SolutionSegment(
            schema,
            t_start=t_start,
            start_canonical=start_canonical,
            dynamics=dynamics,
            name=name,
            parachute=parachute,
        )
        self._segments.append(segment)
        self._version += 1
        return segment

    @property
    def segments(self):
        """The flight phases' segments, in order."""
        return tuple(self._segments)

    @property
    def tail(self):
        """The current (most recent) segment."""
        return self._segments[-1]

    # -- List-like behaviour ----------------------------------------------

    def _locate(self, index):
        """Return the ``(segment, local_index)`` owning a global row index."""
        total = self._length
        if index < 0:
            index += total
        if index < 0 or index >= total:
            raise IndexError("solution row index out of range")
        for segment in self._segments:
            count = len(segment.rows)
            if index < count:
                return segment, index
            index -= count
        raise IndexError("solution row index out of range")

    def _iter_rows(self):
        for segment in self._segments:
            yield from segment.rows

    def __len__(self):
        return self._length

    def __iter__(self):
        return self._iter_rows()

    def append(self, row):
        """Append a raw state row ``[t, *state]`` to the current phase."""
        segment = self._segments[-1]
        if len(row) != segment.width + 1:
            raise ValueError(
                f"State row of length {len(row)} does not match the current "
                f"flight phase (expected {segment.width + 1} values: time plus "
                f"{segment.width} states)."
            )
        segment.rows.append(row)
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
            segment, local = self._locate(int(key))
            return segment.rows[local]
        raise TypeError(
            "Solution indices must be integers, slices, or state names, "
            f"not {type(key).__name__}."
        )

    def __setitem__(self, index, row):
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Solution only supports integer row assignment.")
        segment, local = self._locate(int(index))
        if len(row) != segment.width + 1:
            raise ValueError(
                f"State row of length {len(row)} does not match the flight "
                f"phase it is being written to (expected {segment.width + 1})."
            )
        segment.rows[local] = row
        segment.invalidate()
        self._version += 1

    def insert(self, index, row):
        """Insert a raw state row before the row at ``index``."""
        segment, local = self._locate(int(index))
        if len(row) != segment.width + 1:
            raise ValueError(
                f"State row of length {len(row)} does not match the flight "
                f"phase it is being inserted into (expected {segment.width + 1})."
            )
        segment.rows.insert(local, row)
        segment.invalidate()
        self._length += 1
        self._version += 1

    def pop(self, index=-1):
        """Remove and return the raw state row at ``index``."""
        segment, local = self._locate(int(index))
        row = segment.rows.pop(local)
        segment.invalidate()
        self._length -= 1
        self._version += 1
        return row

    def __array__(self, dtype=None, copy=None):  # pylint: disable=unused-argument
        # `copy` is part of the NumPy array protocol; the array is always freshly
        # built here, so it is accepted for compatibility and not acted on.
        widths = {segment.width for segment in self._segments if segment.rows}
        if len(widths) > 1:
            raise TypeError(
                "This flight has phases that integrate different state "
                "variables, so its solution cannot be turned into a single "
                "rectangular array. Use flight.solution['name'] for one "
                "variable's time history, flight.solution.canonical_array for "
                "the full 14-column table, or flight.solution.segments[i].array "
                "for a single phase."
            )
        rows = list(self._iter_rows())
        return np.array(rows, dtype=dtype)

    # -- Queries by name ---------------------------------------------------

    def series(self, name):
        """Return the ``[t, value]`` history of one variable over the flight.

        The history spans only the phases that define the variable. Variables
        in the canonical state are defined in every phase (reconstructed or
        frozen where a phase does not integrate them), so their history covers
        the whole flight. A phase-specific variable's history covers only the
        phases that integrate it.

        Parameters
        ----------
        name : str
            The variable name (for example ``"vz"``).

        Returns
        -------
        numpy.ndarray
            An ``(n, 2)`` array whose columns are time and value.

        Raises
        ------
        KeyError
            If no phase defines the variable.
        """
        cached = self._series_cache.get(name)
        if cached is not None and cached[0] == self._version:
            return cached[1]
        parts = []
        for segment in self._segments:
            part = segment.series(name)
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
        """Return a :class:`StateView` at the stored time nearest ``t``.

        Parameters
        ----------
        t : float
            Time in seconds.
        atol : float, optional
            If the nearest stored time differs from ``t`` by more than this, a
            warning is raised. Default is ``1e-3``.
        """
        times = self.time
        index = int(np.argmin(np.abs(times - t)))
        if abs(times[index] - t) > atol:
            warnings.warn(
                f"Time {t} not found in solution. Closest time is "
                f"{times[index]}. Using closest time.",
                UserWarning,
            )
        segment, local = self._locate(index)
        return segment.view(segment.rows[local][1:])

    def view(self, index):
        """Return ``(time, StateView)`` for the raw row at ``index``."""
        segment, local = self._locate(int(index))
        row = segment.rows[local]
        return row[0], segment.view(row[1:])

    def canonical_row(self, index):
        """Return ``[t, *canonical_state]`` for the raw row at ``index``."""
        segment, local = self._locate(int(index))
        row = segment.rows[local]
        return [row[0], *segment.canonical_state(row[1:])]

    def canonical_states(self, stop=None):
        """Return canonical states (without time) for every stored row.

        Parameters
        ----------
        stop : int, optional
            If given, only the rows up to ``stop`` (Python slice semantics) are
            returned. Useful for building a history that excludes the most
            recent rows.
        """
        result = []
        for segment in self._segments:
            for row in segment.rows:
                result.append(segment.canonical_state(row[1:]))
        if stop is not None:
            result = result[:stop]
        return result

    @property
    def time(self):
        """The time column of the whole flight as a 1-D array."""
        if self._time_cache is None or self._time_version != self._version:
            self._time_cache = np.array([row[0] for row in self._iter_rows()])
            self._time_version = self._version
        return self._time_cache

    @property
    def canonical_array(self):
        """The whole flight as a rectangular ``(n, 14)`` canonical table.

        Each row is ``[t, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]``.
        Variables a phase does not integrate are reconstructed or frozen.
        """
        if self._canonical_cache is None or self._canonical_version != self._version:
            rows = []
            for segment in self._segments:
                for row in segment.rows:
                    rows.append([row[0], *segment.canonical_state(row[1:])])
            width = len(CANONICAL_STATE_NAMES) + 1
            self._canonical_cache = np.array(rows) if rows else np.empty((0, width))
            self._canonical_version = self._version
        return self._canonical_cache

    # -- Serialization -----------------------------------------------------

    def to_dict(self):
        """Return a serializable description of the whole solution."""
        return {
            "format": "rocketpy/solution",
            "version": 1,
            "segments": [segment.to_dict() for segment in self._segments],
        }

    @classmethod
    def from_dict(cls, data):
        """Rebuild a solution from its serialized (segment-based) form."""
        segments = [SolutionSegment.from_dict(segment) for segment in data["segments"]]
        return cls(segments)

    @classmethod
    def from_legacy_list(cls, rows):
        """Rebuild a solution from the old flat list of canonical rows.

        Older saved flights stored the solution as a single list of
        14-value rows. This wraps that list as one canonical segment so old
        files keep loading.
        """
        rows = [list(row) for row in rows]
        segment = SolutionSegment(
            CANONICAL_SCHEMA,
            t_start=rows[0][0] if rows else None,
            start_canonical=tuple(rows[0][1:]) if rows else None,
        )
        segment.rows = rows
        return cls([segment])


CANONICAL_SCHEMA = StateSchema(CANONICAL_STATE_NAMES)
"""The full 13-variable flight state schema."""

PARACHUTE_3T_SCHEMA = StateSchema(("x", "y", "z", "vx", "vy", "vz"))
"""Schema for a 3-DOF translational parachute descent (position and velocity)."""

# Schemas RocketPy knows about, keyed by their name tuple, so that schemas
# rebuilt from saved files recover their reconstruction functions.
_SCHEMA_REGISTRY = {
    CANONICAL_SCHEMA.names: CANONICAL_SCHEMA,
    PARACHUTE_3T_SCHEMA.names: PARACHUTE_3T_SCHEMA,
}


def register_schema(schema):
    """Register a schema so it is recovered when loading saved flights."""
    _SCHEMA_REGISTRY[schema.names] = schema
    return schema
