import warnings

import numpy as np

from ..tools import snake_case_key
from .helpers.dynamics import (
    BUILT_IN_DYNAMICS,
    CANONICAL_INDEX,
    CANONICAL_STATE_NAMES,
    SIX_DOF_DYNAMICS,
    _BoundDynamics,
    _PhaseDynamics,
)

__all__ = [
    "CANONICAL_INDEX",
    "CANONICAL_STATE_NAMES",
    "Solution",
]


def _nearest_time_index(times, t, atol, where):
    """Return the position of the stored time closest to ``t``.

    Parameters
    ----------
    times : numpy.ndarray
        The stored times, in seconds, in increasing order.
    t : float
        The time being looked for, in seconds.
    atol : float
        How far, in seconds, the closest stored time may be from ``t`` before
        a warning is raised.
    where : str
        Name of the thing being searched, used in the warning message (for
        example ``"solution"``).

    Returns
    -------
    int
        Position in ``times`` of the closest stored time.
    """
    index = int(np.argmin(np.abs(times - t)))
    if abs(times[index] - t) > atol:
        warnings.warn(
            f"Time {t} not found in {where}. Closest time is "
            f"{times[index]}. Using closest time.",
            UserWarning,
        )
    return index


class _ResultsDict(dict):
    """A dictionary of results by name, also read as attributes in snake_case.

    ``results["Canards"]["deflection"]`` is the value as a plain dictionary
    gives it; ``results.canards.deflection`` is the same value, by each name
    in snake_case, and so is ``results["canards"]``: a name is found by its
    snake_case form when it is not there as given. Everything else is the
    dictionary: ``list(results)`` gives the names as given, ``results.items()``
    the pairs.

    A name that is also a method of a dictionary (``items``, ``keys``,
    ``values``, ``get``, ``copy``, ``update``, ``pop``, ``clear``) is read with
    brackets only: ``results["values"]``, since ``results.values`` is the
    method.
    """

    def __init__(self, data):
        # A value that is a dictionary is a group of its own
        super().__init__(
            (name, _ResultsDict(value) if isinstance(value, dict) else value)
            for name, value in data.items()
        )

    def __missing__(self, key):
        wanted = snake_case_key(key)
        for name in self:
            if snake_case_key(name) == wanted:
                return self[name]
        raise KeyError(
            f"Nothing here is named {key!r}. The names are: "
            f"{', '.join(self) or 'none'}."
        )

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(error.args[0]) from None

    def __dir__(self):
        return [*super().__dir__(), *(snake_case_key(name) for name in self)]


class _PhaseSolution:
    """One phase of a flight: what it was flown with, and where it starts.

    A phase describes how to read the rows that belong to it, but does not hold
    them. The rows live in one list on the :class:`Solution`, and
    :attr:`start` is the position of this phase's first row in it;
    :meth:`Solution.phase_span` gives where its rows begin and end.

    Attributes
    ----------
    dynamics : _PhaseDynamics
        The equations of motion this phase was flown with, describing which
        states it integrates and how the rest are rebuilt.
    bound_dynamics : _BoundDynamics or None
        The same dynamics bound to a flight, which lets the phase be
        post-processed. ``None`` if the phase was given unbound dynamics.
    start_canonical : tuple of float or None
        The 13-value canonical state at the moment the phase began. Supplies
        every canonical state this phase does not integrate. ``None`` only for
        a phase that integrates all of them itself.
    t_start : float or None
        The time the phase begins, in seconds.
    name : str or None
        The phase name, for reference in output and debugging.
    start : int
        Position of this phase's first row in the solution's row list. A phase
        that has just begun has no rows yet, and its ``start`` is then the
        number of rows the flight has so far. The solution keeps this up to
        date; treat it as read-only.
    """

    __slots__ = (
        "bound_dynamics",
        "dynamics",
        "name",
        "start",
        "start_canonical",
        "t_start",
    )

    def __init__(self, dynamics, start_canonical, t_start=None, name=None, start=0):
        """Initialize a solution phase.

        Parameters
        ----------
        dynamics : _PhaseDynamics or _BoundDynamics
            The dynamics this phase was flown with, describing the states it
            integrates. Dynamics bound to a flight also let the phase be
            post-processed.
        start_canonical : sequence of float or None
            The full 13-value canonical state at the start of the phase. It
            supplies every canonical state this phase does not integrate,
            which are then held at that value for the whole phase. Only a
            phase that supplies all of them itself may pass ``None``.
        t_start : float, optional
            The time the phase begins, in seconds. Default is ``None``.
        name : str, optional
            The phase name, for reference in output and debugging. Default is
            ``None``.
        start : int, optional
            Position of this phase's first row in the solution's row list.
            Default is ``0``.

        Raises
        ------
        ValueError
            If ``start_canonical`` is ``None`` but this phase needs it to fill
            in the canonical states it does not integrate.
        """
        if isinstance(dynamics, _BoundDynamics):
            self.bound_dynamics = dynamics
            self.dynamics = dynamics.dynamics
        else:
            self.bound_dynamics = None
            self.dynamics = dynamics
        if start_canonical is None and not self.dynamics.is_canonical:
            raise ValueError(
                "This flight phase does not integrate every canonical state, so "
                "it needs the state at the start of the phase to fill in the "
                "ones it does not. Pass start_canonical."
            )
        self.t_start = t_start
        self.start_canonical = (
            tuple(start_canonical) if start_canonical is not None else None
        )
        self.name = name
        self.start = int(start)

    def __repr__(self):
        """Return the phase's name, its states and where its rows begin."""
        return (
            f"_PhaseSolution(name={self.name!r}, "
            f"states={self.dynamics.states!r}, start={self.start})"
        )

    def canonical_state(self, state):
        """Return the full canonical state for a raw state of this phase.

        Parameters
        ----------
        state : sequence of float
            One of this phase's states, without time, in the order the phase
            integrates them.

        Returns
        -------
        sequence of float
            The 13 canonical states. Those this phase does not integrate are
            reconstructed or held at their value when the phase began.
        """
        return self.dynamics.canonicalize(state, self.start_canonical)

    def canonical_derivative(self, state_dot, state=None):
        """Return the full canonical derivative for a derivative of this phase.

        A state this phase holds at its start-of-phase value does not change, so
        its canonical time derivative is zero. A phase with a
        ``to_canonical_dot`` function computes it instead, which needs
        ``state``, the phase's state at the same instant.

        Parameters
        ----------
        state_dot : sequence of float
            Time derivative of one of this phase's states, in the order the
            phase integrates them.
        state : sequence of float, optional
            The phase's state at the same instant. Only needed by a phase with
            a ``to_canonical_dot`` function. Default is ``None``.

        Returns
        -------
        sequence of float
            The 13 canonical time derivatives.

        Raises
        ------
        ValueError
            If this phase has a ``to_canonical_dot`` function but ``state`` was
            not given.
        """
        return self.dynamics.canonicalize_derivative(
            state_dot, state, self.start_canonical
        )

    def state_dict(self, state):
        """Return a raw state of this phase as a state name to value map.

        Parameters
        ----------
        state : sequence of float
            One of this phase's states, without time, in the order the phase
            integrates them.

        Returns
        -------
        dict
            State name to value, for example ``state["vz"]``. Covers the 13
            canonical states (rebuilt or held where this phase does not
            integrate them) plus any state this phase integrates that is not
            one of them.
        """
        values = dict(zip(CANONICAL_STATE_NAMES, self.canonical_state(state)))
        values.update(zip(self.dynamics.states, state))
        return values

    def to_dict(self):
        """Return a serializable description of this phase.

        Returns
        -------
        dict
            The phase's name, dynamics name, state names, start time,
            start-of-phase canonical state and the position of its first row.
            Pass it to :meth:`from_dict` to rebuild the phase.
        """
        return {
            "name": self.name,
            # The dynamics is stored by name, since a derivative is code; the
            # state names are stored too, so a phase whose name is not recognized
            # can still be read back.
            "dynamics": self.dynamics.name,
            "state_names": list(self.dynamics.states),
            "t_start": self.t_start,
            "start_canonical": (
                list(self.start_canonical) if self.start_canonical is not None else None
            ),
            "start": self.start,
        }

    @classmethod
    def from_dict(cls, data):
        """Rebuild a phase from its serialized description.

        Parameters
        ----------
        data : dict
            A description produced by :meth:`to_dict`.

        Returns
        -------
        _PhaseSolution
            The rebuilt phase. Its ``bound_dynamics`` is ``None``, since tying
            the dynamics to a flight needs a running simulation, so the phase
            cannot be post-processed again.
        """
        # The dynamics is code, so it was saved by name. A built-in one is
        # used again, so the canonical states it rebuilds are rebuilt here too.
        dynamics = BUILT_IN_DYNAMICS.get(data.get("dynamics"))
        names = list(data["state_names"])
        if dynamics is not None and names[: dynamics.width] == list(dynamics.states):
            if len(names) > dynamics.width:
                # Flown with the states of the rocket's controllables after its own
                dynamics = _PhaseDynamics(
                    dynamics.name,
                    None,
                    names,
                    dynamics.post_process_vars,
                    dynamics.to_canonical,
                    dynamics.to_canonical_dot,
                )
        else:
            # Unknown, or changed since the flight was saved: its own states
            # can still be read by name, and every other canonical state is
            # held at its start-of-phase value.
            dynamics = _PhaseDynamics(
                data.get("dynamics") or "unknown", None, data["state_names"]
            )
        return cls(
            dynamics,
            data.get("start_canonical"),
            t_start=data.get("t_start"),
            name=data.get("name"),
            start=data.get("start", 0),
        )


class Solution:
    """Everything a flight produced over time, as raw numbers.

    A ``Solution`` answers queries by name. ``solution.vz`` or
    ``solution["vz"]`` is the ``[t, value]`` history of the vertical velocity
    across the whole flight; so are ``solution.ax`` for a variable the flight
    worked out without integrating it (accelerations, forces, moments, thrust)
    and ``solution.controls.canards.deflection`` for a control input.
    :attr:`names` lists them all; ``solution.at(t)`` returns every one of them
    at the nearest stored time as a name-to-value dictionary.
    ``solution.sensors.barometer.pressure`` is what a sensor read, at its own
    sampling times.

    ``solution.phases`` gives the individual flight phases, each a
    :class:`_PhaseSolution`; :meth:`phase_span` says which rows belong to one,
    so ``solution[start:stop]`` reads them.

    It also reads like a plain list of rows: ``len(solution)``, iteration,
    ``solution[-1]``, slicing and ``numpy.array(solution)`` all give 14-value
    canonical rows ``[t, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]``,
    whatever the phase they came from actually integrated.

    Writing is done by the simulation alone, through the private methods
    below. They take the raw states of the phase being flown, since that is
    what the integrator produces.
    """

    def __init__(self, phases=None, rows=None):
        """Initialize a solution.

        Parameters
        ----------
        phases : sequence of _PhaseSolution, optional
            Pre-built phases, in flight order, each with its ``start`` already
            set. Default is ``None``, an empty solution.
        rows : sequence of sequence of float, optional
            The flight's rows, each ``[t, *state]``, in flight order. Default
            is ``None``, no rows.

        Raises
        ------
        ValueError
            If the phases and rows do not agree: phases out of order, a phase
            starting past the end of the rows, or a row whose width does not
            match the phase it falls in.
        """
        # The flight's phases, in the order they were flown.
        self.phases = list(phases) if phases else []
        self._rows = list(rows) if rows else []
        # The control inputs in force from each row's time on (a dictionary of
        # name to value), one entry per row, kept beside the rows the
        # same way. A row worked out again after the flight is given its own.
        # ``None`` where nothing was stored. Saved with the rows.
        self._controls = [None] * len(self._rows)
        # What the sensors read, at their own sampling times: sensor position
        # name to channel name to ``[(time, reading), ...]``. Saved with the
        # rows, and cut with them when the flight goes back in time.
        self._sensors = {}
        # Bumped by every change to the rows. The canonical table and the
        # histories of non-canonical states are cached against it.
        self._version = 0
        # Bumped whenever a row's control inputs change, so the post-process
        # values and the control histories can tell they are stale without
        # the row caches having to be thrown away too.
        self._post_version = 0
        self._canonical_cache = None
        self._canonical_version = -1
        self._series_cache = {}
        self._control_cache = {}
        self._control_names_cache = (None, ())
        # The post-process values of every phase, worked out on the first read
        # and kept until the rows or their control inputs change
        self._post_tables = None
        self._post_key = None
        self._validate_starts()

    def __repr__(self):
        """Return how many phases and how many rows this solution holds."""
        return f"Solution(phases={len(self.phases)}, rows={len(self._rows)})"

    def _validate_starts(self):
        """Raise if the phases do not line up with the rows.

        Only runs when a solution is built from stored data; the simulation
        keeps the two in step as it goes.

        Raises
        ------
        ValueError
            If a phase starts before the one before it, if the first phase does
            not start at the first row, if a phase starts past the end of the
            rows, or if a row does not have the width of the phase it falls in.
        """
        total = len(self._rows)
        if not self.phases:
            if total:
                raise ValueError(
                    "This solution has rows but no flight phases, so there is "
                    "nothing to say what its states mean."
                )
            return
        starts = [phase.start for phase in self.phases]
        if starts[0] != 0:
            raise ValueError(
                f"The first flight phase must start at the first row, but it "
                f"starts at row {starts[0]}."
            )
        for earlier, later in zip(starts, starts[1:]):
            if later < earlier:
                raise ValueError(
                    "Flight phases must be stored in the order they were flown."
                )
        # A phase starting exactly at the end is a phase that has only just
        # begun and holds no rows yet, which is normal mid-flight.
        if starts[-1] > total:
            raise ValueError(
                f"A flight phase starts at row {starts[-1]}, but this "
                f"solution only has {total} rows."
            )
        for _, phase, start, stop in self._spans():
            expected = phase.dynamics.width + 1
            for row in self._rows[start:stop]:
                if len(row) != expected:
                    raise ValueError(
                        f"A row of {len(row)} values is stored in flight phase "
                        f"'{phase.name}', which stores {expected} (time plus "
                        f"{phase.dynamics.width} states)."
                    )

    # -- Phase management ------------------------------------------------

    def _start_phase(self, dynamics, start_canonical, t_start=None, name=None):
        """Begin a new flight phase and return its (empty) _PhaseSolution.

        Parameters
        ----------
        dynamics : _PhaseDynamics or _BoundDynamics
            The dynamics the new phase is flown with, describing the states it
            integrates. Dynamics bound to a flight also let the phase be
            post-processed.
        start_canonical : sequence of float or None
            The full 13-value canonical state at the start of the phase. It
            supplies every canonical state the phase does not integrate, which
            are then held at that value for the whole phase. Only a phase that
            supplies all of them itself may pass ``None``.
        t_start : float, optional
            The time the phase begins, in seconds. Default is ``None``.
        name : str, optional
            The phase name, for reference in output and debugging. Default is
            ``None``.

        Returns
        -------
        _PhaseSolution
            The new phase, with no rows yet. Rows reach it through
            :meth:`_append`.

        Raises
        ------
        ValueError
            If ``start_canonical`` is ``None`` but the phase needs it to fill
            in the canonical states it does not integrate.
        """
        phase = _PhaseSolution(
            dynamics,
            start_canonical,
            t_start=t_start,
            name=name,
            start=len(self._rows),
        )
        self.phases.append(phase)
        self._version += 1
        return phase

    # -- Row access --------------------------------------------------------

    def _normalize(self, index):
        """Return ``index`` counted from the start, checking it is in range.

        Parameters
        ----------
        index : int
            Row position across the whole flight. Negative values count from
            the end.

        Returns
        -------
        int
            The same position counted from the first row.

        Raises
        ------
        IndexError
            If ``index`` is outside the flight's rows.
        """
        total = len(self._rows)
        if index < 0:
            index += total
        if index < 0 or index >= total:
            raise IndexError("solution row index out of range")
        return index

    def _phase_at(self, index):
        """Return the phase owning row ``index``, counted from the start.

        Parameters
        ----------
        index : int
            Row position across the whole flight, already checked to be in
            range and counted from the first row.

        Returns
        -------
        _PhaseSolution
            The phase that row belongs to.
        """
        return self.phases[self._phase_index_of(index)]

    def _phase_index_of(self, position):
        """Return the position in ``phases`` of the phase owning a row.

        The last phase starting at or before the row owns it. A phase that
        holds no rows shares its successor's start, and is passed over. A
        flight has a handful of phases, so a scan back from the end is fine.
        """
        phases = self.phases
        index = len(phases) - 1
        while index > 0 and phases[index].start > position:
            index -= 1
        return index

    def _spans(self):
        """Yield ``(index, phase, start, stop)`` for every phase holding rows.

        ``start`` and ``stop`` bound the phase's rows in the flight's row list,
        as a Python slice would. Phases with no rows are skipped.
        """
        phases, total = self.phases, len(self._rows)
        for index, phase in enumerate(phases):
            start = phase.start
            stop = phases[index + 1].start if index + 1 < len(phases) else total
            if start < stop:
                yield index, phase, start, stop

    def phase_index_at(self, index):
        """Return the position in :attr:`phases` of the phase owning a row.

        Parameters
        ----------
        index : int
            Row position across the whole flight. Negative values count from
            the end.

        Returns
        -------
        int
            Position of the owning phase in :attr:`phases`.

        Raises
        ------
        IndexError
            If ``index`` is outside the flight's rows.
        """
        return self._phase_index_of(self._normalize(int(index)))

    def phase_span(self, phase_index):
        """Return the ``(start, stop)`` row positions of one phase.

        Parameters
        ----------
        phase_index : int
            Position of the phase in :attr:`phases`. Negative values count
            from the end.

        Returns
        -------
        tuple of (int, int)
            First row of the phase, and one past its last row, as a Python
            slice would bound them. They are equal for a phase with no rows.

        Raises
        ------
        IndexError
            If ``phase_index`` is outside this flight's phases.
        """
        count = len(self.phases)
        if phase_index < 0:
            phase_index += count
        if phase_index < 0 or phase_index >= count:
            raise IndexError("solution phase index out of range")
        start = self.phases[phase_index].start
        stop = (
            self.phases[phase_index + 1].start
            if phase_index + 1 < count
            else len(self._rows)
        )
        return start, stop

    def raw_row(self, index):
        """Return row ``index`` in its own phase's states, ``[t, *state]``.

        Unlike ``solution[index]``, which always reports the 14-value canonical
        row, this gives the row exactly as the phase integrated it. A phase that
        follows only position and velocity gives 7 values here.

        Parameters
        ----------
        index : int
            Row position across the whole flight. Negative values count from
            the end.

        Returns
        -------
        list of float
            The row as stored, ``[t, *state]``.

        Raises
        ------
        IndexError
            If ``index`` is outside the flight's rows.
        """
        return self._rows[int(index)]

    def canonical_row(self, index):
        """Return ``[t, *canonical_state]`` for the raw row at ``index``.

        Parameters
        ----------
        index : int
            Row position across the whole flight. Negative values count from
            the end.

        Returns
        -------
        list of float
            The 14 values ``[t, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2,
            w3]``.

        Raises
        ------
        IndexError
            If ``index`` is outside the flight's rows.
        """
        position = self._normalize(int(index))
        row = self._rows[position]
        phase = self._phase_at(position)
        if phase.dynamics.is_canonical:
            return list(row)
        return [row[0], *phase.canonical_state(row[1:])]

    # -- Mutation ----------------------------------------------------------

    def _shift_starts_after(self, position, delta):
        """Move every phase starting after ``position`` by ``delta`` rows.

        Called whenever a row is added or removed, so the phases keep pointing
        at the right rows.

        Parameters
        ----------
        position : int
            Row position, counted from the start, where the change happened.
        delta : int
            How many rows were added (positive) or removed (negative).
        """
        # Walking back from the end stops at the first phase that begins at or
        # before the change, so a change in the last phase touches nothing.
        for phase in reversed(self.phases):
            if phase.start <= position:
                break
            phase.start += delta

    def _append(self, row):
        """Append a raw state row ``[t, *state]`` to the current phase.

        Parameters
        ----------
        row : sequence of float
            Time followed by the states the current phase integrates, in the
            order that phase integrates them. A phase must have been started
            first (see :meth:`_start_phase`): this is not checked.
        """
        self._rows.append(row)
        self._controls.append(None)
        self._version += 1

    def _set_controls(self, index, controls):
        """Store the control inputs in force from one row's time on.

        Parameters
        ----------
        index : int
            Row position across the whole flight. Negative values count from
            the end.
        controls : dict
            Name to value, a copy of the flight's control inputs.

        Raises
        ------
        IndexError
            If ``index`` is outside the flight's rows.
        """
        self._controls[self._normalize(int(index))] = controls
        self._post_version += 1

    def _replace_last(self, row):
        """Overwrite the most recent row with a raw state row ``[t, *state]``.

        Parameters
        ----------
        row : sequence of float
            Time followed by the states the row's phase integrates, in the
            order that phase integrates them.

        Raises
        ------
        IndexError
            If the flight has no rows yet.
        """
        self._rows[-1] = row
        self._controls[-1] = None
        self._version += 1

    def _insert_before_last(self, row):
        """Insert a raw state row ``[t, *state]`` just before the most recent one.

        Used to record the exact time of an event without discarding the step
        the solver already took past it.

        Parameters
        ----------
        row : sequence of float
            Time followed by the states the row's phase integrates, in the
            order that phase integrates them.

        Raises
        ------
        IndexError
            If the flight has no rows yet.
        """
        position = len(self._rows) - 1
        self._rows.insert(position, row)
        self._controls.insert(position, None)
        self._shift_starts_after(position, 1)
        self._version += 1

    def _drop_last(self):
        """Remove and return the most recent row, in its own phase's states.

        Returns
        -------
        list of float
            The removed row, ``[t, *state]``, as it was stored.

        Raises
        ------
        IndexError
            If the flight has no rows yet.
        """
        if not self._rows:
            raise IndexError("this solution has no stored rows")
        row = self._rows.pop()
        self._controls.pop()
        self._shift_starts_after(len(self._rows), -1)
        self._version += 1
        return row

    # -- List-like reads (canonical rows) ----------------------------------

    def __len__(self):
        """Return how many rows the whole flight holds."""
        return len(self._rows)

    def __iter__(self):
        """Iterate over the flight's 14-value canonical rows."""
        for _, phase, start, stop in self._spans():
            rows = self._rows[start:stop]
            if phase.dynamics.is_canonical:
                for row in rows:
                    yield list(row)
            else:
                canonical_state = phase.canonical_state
                for row in rows:
                    yield [row[0], *canonical_state(row[1:])]

    def __getitem__(self, key):
        """Return a canonical row, a list of them, or a state's history.

        A state name gives that state's ``[t, value]`` history (the same as
        :meth:`series`); an integer gives one 14-value canonical row; a slice
        gives a list of them.

        Parameters
        ----------
        key : str, int or slice
            The state name, row position, or range of rows wanted.

        Returns
        -------
        numpy.ndarray or list of float or list of list of float
            The state's history for a name, one canonical row for an integer,
            a list of canonical rows for a slice.

        Raises
        ------
        KeyError
            If no phase defines the named state.
        IndexError
            If an integer position is outside the flight's rows.
        TypeError
            If ``key`` is not a state name, integer or slice.
        """
        if isinstance(key, str):
            return self.series(key)
        if isinstance(key, slice):
            return [
                self.canonical_row(position)
                for position in range(*key.indices(len(self)))
            ]
        if isinstance(key, (int, np.integer)):
            return self.canonical_row(int(key))
        raise TypeError(
            "Solution indices must be integers, slices, or state names, "
            f"not {type(key).__name__}."
        )

    def __array__(self, dtype=None, copy=None):  # pylint: disable=unused-argument
        """Return the whole flight as the ``(n, 14)`` canonical table.

        Lets ``numpy.array(solution)`` work on any flight, including one whose
        phases integrate different states.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            Element type of the returned array. Default is ``None``, keeping
            the table's own type.
        copy : bool, optional
            Whether the caller needs an array of its own. ``False`` asks for
            the flight's own table without copying it, which only works when
            no type change is wanted. Default is ``None``, which copies.

        Returns
        -------
        numpy.ndarray
            The ``(n, 14)`` canonical table.

        Raises
        ------
        ValueError
            If ``copy`` is ``False`` but the requested ``dtype`` would force
            a copy.
        """
        array = self.canonical_array
        if copy is False:
            if dtype is not None and np.dtype(dtype) != array.dtype:
                raise ValueError(
                    "The canonical table cannot be returned as "
                    f"{np.dtype(dtype)} without copying it."
                )
            return array
        # Anything the caller can write to has to be its own array: the table
        # above is the flight's cache, shared by every later read of it.
        return array.astype(dtype) if dtype is not None else array.copy()

    @property
    def time(self):
        """The time column of the whole flight, in seconds, as a 1-D array."""
        return self.canonical_array[:, 0]

    @property
    def canonical_array(self):
        """The whole flight as a rectangular ``(n, 14)`` canonical table.

        Each row is ``[t, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]``.
        States a phase does not integrate are rebuilt or held.

        The table belongs to the flight and cannot be written to. For a table
        of your own to edit, use ``numpy.array(solution)``.
        """
        if self._canonical_cache is None or self._canonical_version != self._version:
            spans = list(self._spans())
            if not spans:
                result = np.empty((0, len(CANONICAL_STATE_NAMES) + 1))
            elif all(phase.dynamics.is_canonical for _, phase, _, _ in spans):
                # Every phase already stores the 14 canonical values, so the
                # stored rows are the table. Guarded on all of them, since a
                # phase storing fewer values would make the rows uneven.
                result = np.array(self._rows)
            else:
                result = np.array(list(self))
            # Read-only so that a caller writing to the table it was handed
            # is told, rather than silently rewriting every later read of it.
            result.flags.writeable = False
            self._canonical_cache = result
            self._canonical_version = self._version
        return self._canonical_cache

    # -- Queries by name ---------------------------------------------------

    @property
    def names(self):
        """Every quantity this flight holds per row, by name.

        The 13 canonical states first, then the states only some phases
        integrate (a controllable's own states, for example), then the
        post-process variables (accelerations, forces, moments, thrust) and
        the control inputs. Each is read as ``solution.<name>`` or
        ``solution["<name>"]``.

        Returns
        -------
        tuple of str
        """
        names = list(CANONICAL_STATE_NAMES)
        for phase in self.phases:
            for name in phase.dynamics.states:
                if name not in names:
                    names.append(name)
        for name in self.post_process_names:
            if name not in names:
                names.append(name)
        names.extend(self.control_names)
        return tuple(names)

    @property
    def post_process_names(self):
        """Names of every post-process variable this flight computes.

        The phases of a flight need not all compute the same variables: a
        parachute descent reports no moments and no thrust. This lists every
        name any phase computes, in the order the phases first introduce them.
        A phase that does not compute one of them reports zero for it, which is
        the physically right answer for the variables RocketPy defines.

        Returns
        -------
        tuple of str
        """
        names = []
        for phase in self.phases:
            for name in phase.dynamics.post_process_vars:
                # What a continuous controller's law wrote to each control
                # input is kept with the post-process values, but it is read
                # as the control input itself
                if name not in names and not name.startswith("controls."):
                    names.append(name)
        return tuple(names)

    @property
    def control_names(self):
        """Names of the control inputs stored with the rows, in order.

        Returns
        -------
        tuple of str
        """
        # Kept until the rows or their control inputs change: looking for the
        # first row with controls is a pass over the rows
        key = (self._version, self._post_version)
        if self._control_names_cache[0] != key:
            names = next(
                (
                    tuple(controls)
                    for controls in self._controls
                    if controls is not None
                ),
                (),
            )
            self._control_names_cache = (key, names)
        return self._control_names_cache[1]

    def __contains__(self, name):
        """Return whether ``name`` is one of :attr:`names`."""
        return isinstance(name, str) and name in self.names

    def __getattr__(self, name):
        """Read a quantity by name: ``solution.vz``, ``solution.ax``."""
        # Only names that are not attributes get here. Dunder and private
        # names are asked for by copying and pickling before __init__ ran
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self.series(name)
        except KeyError as error:
            raise AttributeError(error.args[0]) from None

    def __dir__(self):
        return [*super().__dir__(), *self.names]

    def series(self, name):
        """Return the ``[t, value]`` history of one quantity over the flight.

        The same as ``solution[name]`` and ``solution.<name>``. A canonical
        state is defined in every phase (rebuilt or held where a phase does not
        integrate it), so its history covers the whole flight. A state only
        some phases integrate covers those phases, and a warning names the
        ones left out. A post-process variable covers the phases that can be
        worked out: all of them on a live flight, none on a flight read back
        from a file. A control input covers every row, with the value in
        force from that row's time on, or what a continuous controller's law
        left at that row.

        Parameters
        ----------
        name : str
            One of :attr:`names`, for example ``"vz"``, ``"ax"`` or
            ``"canards.deflection"``.

        Returns
        -------
        numpy.ndarray
            An ``(n, 2)`` array whose columns are time and value.

        Raises
        ------
        KeyError
            If the flight has nothing by that name, or a post-process
            variable cannot be worked out.

        Warns
        -----
        UserWarning
            If some phases do not define a state, so its history has a gap.
        """
        slot = CANONICAL_INDEX.get(name)
        if slot is not None:
            # Every phase reports the canonical states, so the history is two
            # columns of the canonical table.
            return self.canonical_array[:, [0, slot + 1]]
        if name in self.control_names:
            return self._control_history(name)[0]
        if name in self.post_process_names:
            return self._post_series(name)
        return self._state_series(name)

    def _state_series(self, name):
        """The history of a state only some phases integrate."""
        cached = self._series_cache.get(name)
        if cached is not None and cached[0] == self._version:
            return cached[1]
        parts = []
        missing = []
        for index, phase, start, stop in self._spans():
            # A state the phase integrates is a column of its rows
            column = phase.dynamics.state_index.get(name)
            if column is None:
                missing.append(phase.name or f"phase {index}")
                continue
            rows = self._rows[start:stop]
            parts.append(
                np.array([(row[0], row[column + 1]) for row in rows], dtype=float)
            )
        if not parts:
            raise KeyError(
                f"This flight has nothing named '{name}'. It has: "
                f"{', '.join(self.names)}."
            )
        if missing:
            warnings.warn(
                f"State '{name}' is not defined during {', '.join(missing)}, so "
                f"its history skips those parts of the flight and is not "
                f"continuous in time.",
                UserWarning,
            )
        result = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        self._series_cache[name] = (self._version, result)
        return result

    def _control_history(self, name):
        """The history of one control input, and whether a continuous
        controller's law wrote it.

        The history is the value in force at each row (``nan`` for a row
        stored without controls) or, where a law wrote it, the value the law
        left at each row. A law's value changes between the rows; a sampled
        value holds until the next write."""
        key = (self._version, self._post_version)
        cached = self._control_cache.get(name)
        if cached is not None and cached[0] == key:
            return cached[1]
        values = [
            np.nan if controls is None else controls[name]
            for controls in self._controls
        ]
        result = np.column_stack([self.time, values])
        written = self._post_series_or_none(f"controls.{name}")
        by_law = written is not None and not np.array_equal(written[:, 1], result[:, 1])
        history = (written if by_law else result, by_law)
        self._control_cache[name] = (key, history)
        return history

    @property
    def controls(self):
        """The control inputs over the flight, by controllable.

        ``solution.controls.canards.deflection`` is an ``(n, 2)`` array of time
        and value, one entry per row: the value in force from that row's time
        on, or what a continuous controller's law left at that row. The same
        array is ``solution["canards.deflection"]``;
        ``flight.controls.canards.deflection`` is it as a Function of time.

        Returns
        -------
        dict
            Controllable to its control inputs. Empty for a flight without
            controls.
        """
        grouped = {}
        for name in self.control_names:
            owner, _, control = name.partition(".")
            grouped.setdefault(owner, {})[control] = self._control_history(name)[0]
        return _ResultsDict(grouped)

    def at(self, t, atol=1e-3):
        """Return everything the flight knows at the stored time nearest ``t``.

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
            Name to value: the states (see :meth:`at_index`), the control
            inputs in force at that row, and the post-process variables the
            phase that time falls in computes, when the flight can work them
            out. Phases do not all compute the same variables, so a parachute
            descent reports fewer names than a powered ascent.
        """
        index = _nearest_time_index(self.time, t, atol, "solution")
        values = self.at_index(index)
        values.update(self._post_at_index(index))
        return values

    def at_index(self, index):
        """Return the state and control inputs stored at row ``index``.

        Reads a single row, so it stays cheap no matter how long the flight is.
        Use it instead of ``solution["vz"][index, 1]`` when only one row is
        needed, since that builds the state's whole history first.

        Parameters
        ----------
        index : int
            Row position across the whole flight. Negative values count from
            the end.

        Returns
        -------
        dict
            Name to value, for example ``values["vz"]``. Holds the 13
            canonical states (reconstructed or held at their value when the
            phase began, where the phase does not integrate them), any state
            the phase integrates that is not one of them, and the control
            inputs in force at that row.

        Raises
        ------
        IndexError
            If ``index`` is outside the flight's rows.
        """
        position = self._normalize(int(index))
        phase = self._phase_at(position)
        values = phase.state_dict(self._rows[position][1:])
        controls = self._controls[position]
        if controls:
            values.update(controls)
        return values

    def value_at(self, index, name):
        """Return one state's value at row ``index``.

        Reads a single value, so it stays cheap no matter how long the flight
        is. Use it instead of ``solution.at_index(index)[name]`` when only one
        state is wanted, since that builds the whole state first.

        Parameters
        ----------
        index : int
            Row position across the whole flight. Negative values count from
            the end.
        name : str
            The state name (for example ``"vz"``).

        Returns
        -------
        float
            The state's value at that row.

        Raises
        ------
        IndexError
            If ``index`` is outside the flight's rows.
        KeyError
            If the phase that row belongs to cannot report the state.
        """
        position = self._normalize(int(index))
        row = self._rows[position]
        phase = self._phase_at(position)
        # A state the phase integrates is a value in the row itself; any other
        # canonical state comes from the phase's canonical view.
        column = phase.dynamics.state_index.get(name)
        if column is not None:
            return row[column + 1]
        slot = CANONICAL_INDEX.get(name)
        if slot is not None:
            return phase.canonical_state(row[1:])[slot]
        raise KeyError(
            f"State '{name}' is not defined in this flight phase. "
            f"It integrates {', '.join(phase.dynamics.states)}."
        )

    # -- Sensor readings ---------------------------------------------------

    @property
    def sensors(self):
        """What the sensors read over the flight, by sensor.

        ``solution.sensors.barometer.pressure`` is an ``(n, 2)`` array of time
        and reading, one entry per sampling time of that sensor, with its
        noise and resolution. A sensor added in several positions is numbered
        from 2 at the positions after the first: ``"Accelerometer"``,
        ``"Accelerometer 2"``, read as ``accelerometer`` and
        ``accelerometer_2``. ``flight.sensor_history.barometer.pressure`` is
        the same reading as a Function of time.

        Returns
        -------
        dict
            Sensor to its channels. Empty for a flight without sensors.
        """
        return _ResultsDict(
            {
                name: {
                    channel: np.array(points, dtype=float)
                    for channel, points in channels.items()
                }
                for name, channels in self._sensors.items()
            }
        )

    def _add_reading(self, name, time, readings):
        """Store what the sensor position ``name`` read at ``time``: channel
        name to value."""
        channels = self._sensors.setdefault(name, {})
        for channel, value in readings.items():
            channels.setdefault(channel, []).append((time, value))

    def _latest_reading(self, name, channel):
        """The last value the sensor position ``name`` read on ``channel``,
        with its time, or ``None`` before its first reading."""
        points = self._sensors.get(name, {}).get(channel)
        return points[-1] if points else None

    def _drop_readings_after(self, time):
        """Forget every reading made after ``time``, when the flight goes back
        to it."""
        for channels in self._sensors.values():
            for points in channels.values():
                while points and points[-1][0] > time:
                    points.pop()

    # -- Post-process variables --------------------------------------------
    # On its way to each state derivative, a flight phase works out quantities
    # it never integrates: the accelerations, the aerodynamic forces and
    # moments, the net thrust. They are worked out again after the flight
    # from the stored rows, each with its control inputs and with the surface
    # switches replayed by time, the first time one is read, and kept until
    # the rows or their control inputs change. They are not saved to file: a
    # flight read back from one cannot report them, since that needs the
    # equations of motion.

    def _phase_table(self, phase, start, stop):
        """Return the post-process values of a phase holding rows ``start`` to
        ``stop``, as an ``(n_rows, n_variables)`` array, or ``None`` if the
        phase cannot be worked out again (a phase read back from a saved
        flight)."""
        if phase.bound_dynamics is None:
            return None
        rows = self._rows[start:stop]
        # A row with no controls of its own was written since the flight last
        # stored them, and is worked out with the flight's controls as they
        # are now
        controls = self._controls[start:stop]
        return np.array(
            [
                phase.dynamics.post_process_values(
                    phase.bound_dynamics.post_process_at(row[0], row[1:], row_controls)
                )
                for row, row_controls in zip(rows, controls)
            ],
            dtype=float,
        )

    def _phase_tables(self):
        """Return every phase's post-process values, worked out once and kept
        until the rows or their control inputs change, so reading a second
        variable costs nothing."""
        key = (self._version, self._post_version)
        if self._post_tables is None or self._post_key != key:
            # ``None`` for a phase with no rows
            tables = [None] * len(self.phases)
            for index, phase, start, stop in self._spans():
                tables[index] = self._phase_table(phase, start, stop)
            self._post_tables = tables
            self._post_key = key
        return self._post_tables

    def phase_values(self, phase_index):
        """Return one phase's post-process values, one row per row of the phase.

        Parameters
        ----------
        phase_index : int
            Position of the phase in the solution's phases. Negative values
            count from the end.

        Returns
        -------
        numpy.ndarray or None
            An array with one row per row of the phase and one column per name
            in that phase's ``post_process_vars``, or ``None`` if the phase
            holds no rows or its values cannot be worked out.

        Raises
        ------
        IndexError
            If ``phase_index`` is outside this flight's phases.
        """
        return self._phase_tables()[phase_index]

    def _post_series_or_none(self, name):
        """The history of one post-process variable over the phases that can
        be worked out, or ``None`` when none can."""
        if not any(name in phase.dynamics.post_process_index for phase in self.phases):
            return None
        tables = self._phase_tables()
        parts = []
        for index, phase, start, stop in self._spans():
            table = tables[index]
            if table is None:
                continue
            times = self.time[start:stop]
            column = phase.dynamics.post_process_index.get(name)
            values = np.zeros(len(times)) if column is None else table[:, column]
            parts.append(np.column_stack([times, values]))
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    def _post_series(self, name):
        """The history of one post-process variable, or a ``KeyError`` when
        the flight cannot work it out."""
        result = self._post_series_or_none(name)
        if result is None:
            raise KeyError(
                f"The post-process variable '{name}' cannot be worked out for "
                f"this flight. Either it has no stored states yet, or it was "
                f"read back from a saved file, which does not carry the "
                f"equations of motion needed to work the variable out again."
            )
        return result

    def _post_at_index(self, index):
        """The post-process variables at row ``index``, worked out for that
        row alone, and the control inputs as a continuous controller's law
        left them there. Empty when the phase computes none or cannot be
        worked out."""
        position = self._normalize(int(index))
        phase = self._phase_at(position)
        if not phase.dynamics.post_process_vars or phase.bound_dynamics is None:
            return {}
        row = self._rows[position]
        values = phase.dynamics.post_process_values(
            phase.bound_dynamics.post_process_at(
                row[0], row[1:], self._controls[position]
            )
        )
        return {
            name.removeprefix("controls."): value
            for name, value in zip(phase.dynamics.post_process_vars, values)
        }

    # -- Serialization -----------------------------------------------------

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        """Return a serializable description of the whole solution.

        Parameters
        ----------
        **kwargs
            The options RocketPy passes to every object it saves, such as
            ``include_outputs``. A solution is plain numbers, so none of them
            change what is written here.

        Returns
        -------
        dict
            The flight's rows, one entry per phase as produced by
            :meth:`_PhaseSolution.to_dict`, for a flight with controls the
            control values stored with each row, and for a flight with
            sensors what they read. Pass it to :meth:`from_dict` to rebuild
            the solution.
        """
        data = {
            "phases": [phase.to_dict() for phase in self.phases],
            "rows": self._rows,
        }
        if any(self._controls):
            # Control input names once, then one list of values per row. A
            # control input a continuous controller's law wrote is saved as
            # the law left it, since that is not worked out again on loading
            names = self.control_names
            histories = {name: self._control_history(name) for name in names}
            data["control_names"] = list(names)
            data["controls"] = [
                None
                if controls is None
                else [
                    float(histories[name][0][position, 1])
                    if histories[name][1]
                    else controls[name]
                    for name in names
                ]
                for position, controls in enumerate(self._controls)
            ]
        if self._sensors:
            data["sensors"] = {
                name: {
                    channel: [[float(t), float(value)] for t, value in points]
                    for channel, points in channels.items()
                }
                for name, channels in self._sensors.items()
            }
        return data

    @classmethod
    def from_dict(cls, data):
        """Rebuild a solution from its serialized form.

        Parameters
        ----------
        data : dict
            A description produced by :meth:`to_dict`.

        Returns
        -------
        Solution
            The rebuilt solution. Its phases cannot be post-processed again,
            since that needs a running simulation.
        """
        phases = [_PhaseSolution.from_dict(entry) for entry in data.get("phases", [])]
        rows = [list(row) for row in data.get("rows", [])]
        solution = cls(phases, rows)
        names = data.get("control_names")
        if names:
            solution._controls = [
                None if values is None else dict(zip(names, values))
                for values in data["controls"]
            ]
            solution._post_version += 1
        solution._sensors = {
            name: {
                channel: [tuple(point) for point in points]
                for channel, points in channels.items()
            }
            for name, channels in data.get("sensors", {}).items()
        }
        return solution

    @classmethod
    def from_legacy_list(cls, rows):
        """Rebuild a solution from the old flat list of canonical rows.

        Older saved flights stored the solution as a single list of
        14-value rows. This wraps that list as one canonical phase so old
        files keep loading.

        Parameters
        ----------
        rows : sequence of sequence of float
            The stored rows, each ``[t, x, y, z, vx, vy, vz, e0, e1, e2, e3,
            w1, w2, w3]``. May be empty.

        Returns
        -------
        Solution
            A solution holding one six-degree-of-freedom phase, anchored at the
            first row's time and state.
        """
        rows = [list(row) for row in rows]
        phase = _PhaseSolution(
            SIX_DOF_DYNAMICS,
            tuple(rows[0][1:]) if rows else None,
            t_start=rows[0][0] if rows else None,
            start=0,
        )
        return cls([phase], rows)
