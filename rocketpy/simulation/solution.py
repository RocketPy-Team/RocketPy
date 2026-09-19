import warnings

import numpy as np

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

# Time plus the 13 canonical states: the width of every canonical row.
CANONICAL_WIDTH = len(CANONICAL_STATE_NAMES) + 1


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
        The same dynamics tied to a live flight, which is what post-processing
        needs. ``None`` for a phase read back from a saved file, since binding
        requires a running simulation.
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
            integrates. Pass the flight-bound one during a simulation, so the
            phase can be post-processed afterwards; a solution loaded from a
            file passes the plain one, since binding requires a live flight.
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
            Default is ``0``. :meth:`Solution._start_phase` fills this in.

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
        if dynamics is None or list(dynamics.states) != data["state_names"]:
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
    """The state history of a whole flight.

    A ``Solution`` answers queries by state name. ``solution["vz"]`` returns the
    ``[t, value]`` history of the vertical velocity across the whole flight;
    ``solution.at(t)`` returns the whole state at the nearest stored time as a
    name-to-value dictionary. ``solution.phases`` gives the individual flight
    phases, each a :class:`_PhaseSolution`; :meth:`phase_span` says which rows
    belong to one, so ``solution[start:stop]`` reads them.

    The variables the flight computed without integrating them, such as the
    accelerations and the aerodynamic forces, are read through
    ``solution.post``: see :class:`PostProcessSolution`.

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
            set. A new flight starts empty and adds one as each of its flight
            phases begins. Default is ``None``, an empty solution.
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
        # Post-process values, one entry per row and in the same order, so a row
        # and its values can never drift apart. They carry no time of their own:
        # the row they sit beside supplies it. ``None`` where nothing was
        # recorded. Not saved to file: a solution read back has no live flight
        # to post-process against.
        self._post_values = [None] * len(self._rows)
        # Set by a flight that records its post-process values as it runs. Such
        # a flight changes the rocket mid-flight, so working the values out
        # again afterwards would read the rocket in the configuration it ended
        # in. Once this is on, recorded values are the only acceptable answer.
        self.records_post_values = False
        self.post = PostProcessSolution(self)
        # Bumped by every change to the rows. The canonical table and the
        # histories of non-canonical states are cached against it.
        self._version = 0
        # Bumped whenever a recorded post-process value changes, so
        # ``post`` can tell its own answers are stale without the row caches
        # having to be thrown away too.
        self._post_version = 0
        self._canonical_cache = None
        self._canonical_version = -1
        self._series_cache = {}
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
            integrates. Pass the flight-bound one during a simulation, so the
            phase can be post-processed afterwards; a solution loaded from a
            file passes the plain one, since binding requires a live flight.
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

    def _last_row_phase(self):
        """Return the phase owning the most recent row.

        A phase that has only just begun holds no rows, so this skips it and
        gives the phase before it. Raises ``IndexError`` if there are no rows.
        """
        if not self._rows:
            raise IndexError("this solution has no stored rows")
        return self._phase_at(len(self._rows) - 1)

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

    # -- Per-phase reads ---------------------------------------------------

    def _phase_series_or_none(self, phase, start, stop, name):
        """Return a phase's ``[t, value]`` history, or ``None`` if undefined.

        Lets :meth:`series` skip the phases that do not define a state while
        gathering its history across the whole flight.

        Parameters
        ----------
        phase : _PhaseSolution
            The phase to read.
        start, stop : int
            The phase's rows in the flight's row list, as a slice would bound
            them.
        name : str
            The state name (for example ``"vz"``).

        Returns
        -------
        numpy.ndarray or None
            An ``(n, 2)`` array whose columns are time and value, or ``None``
            if the phase cannot report the state or holds no rows.
        """
        if start == stop:
            return None
        rows = self._rows[start:stop]
        times = np.array([row[0] for row in rows])
        # A state the phase integrates is a plain column of its rows; any other
        # canonical state comes from the phase's canonical view.
        index = phase.dynamics.state_index.get(name)
        if index is not None:
            return np.column_stack([times, [row[index + 1] for row in rows]])
        slot = CANONICAL_INDEX.get(name)
        if slot is None:
            return None
        canonical_state = phase.canonical_state
        return np.column_stack(
            [times, [canonical_state(row[1:])[slot] for row in rows]]
        )

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
            order that phase integrates them.

        Raises
        ------
        IndexError
            If no phase has been started yet.
        """
        self._rows.append(row)
        self._post_values.append(None)
        self._version += 1

    def _set_post_values(self, index, values):
        """Record the post-process values of one row.

        Called as the simulation runs, right after the row is written, so the
        values are the ones the rocket really had at that state. They are kept
        beside the row and move with it, so a rollback cannot leave the two out
        of step.

        Parameters
        ----------
        index : int
            Row position across the whole flight. Negative values count from
            the end.
        values : sequence of float
            The values in that row's phase ``post_process_vars`` order.

        Raises
        ------
        IndexError
            If ``index`` is outside the flight's rows.
        """
        # Deliberately does not count as a change to the flight: the states are
        # untouched, so the cached times and tables stay valid. Only ``post``
        # needs to know, and it watches its own counter.
        self._post_values[self._normalize(int(index))] = values
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
        self._post_values[-1] = None
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
        self._post_values.insert(position, None)
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
        self._post_values.pop()
        self._shift_starts_after(len(self._rows), -1)
        self._version += 1
        return row

    def _insert(self, index, row):
        """Insert a raw state row before the row at ``index``.

        Parameters
        ----------
        index : int
            Row position across the whole flight to insert before. Negative
            values count from the end.
        row : sequence of float
            Time followed by the states the phase at ``index`` integrates, in
            the order that phase integrates them.

        Raises
        ------
        IndexError
            If ``index`` is outside the flight's rows.
        """
        position = self._normalize(int(index))
        self._rows.insert(position, row)
        self._post_values.insert(position, None)
        self._shift_starts_after(position, 1)
        self._version += 1

    def _pop(self, index=-1):
        """Remove and return the raw state row at ``index``.

        Parameters
        ----------
        index : int, optional
            Row position across the whole flight. Negative values count from
            the end. Default is ``-1``, the most recent row.

        Returns
        -------
        list of float
            The removed row, ``[t, *state]``, as it was stored.

        Raises
        ------
        IndexError
            If ``index`` is outside the flight's rows.
        """
        position = self._normalize(int(index))
        row = self._rows.pop(position)
        self._post_values.pop(position)
        self._shift_starts_after(position, -1)
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

    def _set_row(self, index, row):
        """Overwrite the row at ``index`` with a raw state row ``[t, *state]``.

        Parameters
        ----------
        index : int
            Row position across the whole flight. Negative values count from
            the end.
        row : sequence of float
            Time followed by the states the phase at ``index`` integrates, in
            the order that phase integrates them.

        Raises
        ------
        IndexError
            If ``index`` is outside the flight's rows.
        """
        position = self._normalize(int(index))
        self._rows[position] = row
        self._post_values[position] = None
        self._version += 1

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

    # -- Queries by name ---------------------------------------------------

    def series(self, name):
        """Return the ``[t, value]`` history of one state over the flight.

        The history spans only the phases that define the state. A canonical
        state is defined in every phase (rebuilt or held where a phase does not
        integrate it), so its history covers the whole flight. A phase-specific
        state's history covers only the phases that integrate it, and a warning
        names the phases left out.

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

        Warns
        -----
        UserWarning
            If some phases do not define the state, so its history has a gap.
        """
        slot = CANONICAL_INDEX.get(name)
        if slot is not None:
            # Every phase reports the canonical states, so the history is two
            # columns of the canonical table.
            return self.canonical_array[:, [0, slot + 1]]
        cached = self._series_cache.get(name)
        if cached is not None and cached[0] == self._version:
            return cached[1]
        parts = []
        missing = []
        for index, phase, start, stop in self._spans():
            part = self._phase_series_or_none(phase, start, stop, name)
            if part is None:
                missing.append(phase.name or f"phase {index}")
            else:
                parts.append(part)
        if not parts:
            raise KeyError(
                f"State '{name}' is not defined in any flight phase of this solution."
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

        Parameters
        ----------
        index : int
            Row position across the whole flight. Negative values count from
            the end.

        Returns
        -------
        dict
            State name to value, as described in :meth:`at`.

        Raises
        ------
        IndexError
            If ``index`` is outside the flight's rows.
        """
        position = self._normalize(int(index))
        phase = self._phase_at(position)
        return phase.state_dict(self._rows[position][1:])

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
                result = np.empty((0, CANONICAL_WIDTH))
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
            A format marker, the flight's rows, and one entry per phase as
            produced by :meth:`_PhaseSolution.to_dict`. Pass it to
            :meth:`from_dict` to rebuild the solution.
        """
        return {
            "format": "rocketpy/solution",
            "phases": [phase.to_dict() for phase in self.phases],
            "rows": self._rows,
        }

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
        return cls(phases, rows)

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


class PostProcessSolution:
    """The variables a flight computes besides the states it integrates.

    On its way to each state derivative, a flight phase works out quantities it
    never integrates: the accelerations, the aerodynamic forces and moments, and
    the net thrust. This class is how you read them. Reach it as
    ``flight.solution.post``.

    Read one variable over the whole flight with ``post["ax"]``, which gives an
    ``(n, 2)`` array of time and value, the same shape ``solution["vz"]`` gives
    for a state. Read every variable at one instant with ``post.at(t)``, which
    gives a name-to-value dictionary. :attr:`names` lists what this flight
    computes.

    Where the values come from depends on the flight. A flight whose controllers
    move an air brake records them as it runs, since replaying the stored states
    afterwards would read the rocket in the configuration it ended the flight
    in. Any other flight works them out from its stored states the first time
    you ask, and keeps the answer. Either way the values are not saved to file,
    so a flight read back from a saved file cannot report them: recomputing them
    needs the equations of motion, which are code and cannot be written out.

    Examples
    --------
    >>> flight.solution.post.names  # doctest: +SKIP
    ('ax', 'ay', 'az', 'alpha1', ..., 'net_thrust')
    >>> flight.solution.post["az"]  # doctest: +SKIP
    array([[0.   , 0.   ],
           [0.001, 9.81 ],
           ...
    >>> flight.solution.post.at(3.0)["net_thrust"]  # doctest: +SKIP
    1834.2
    """

    # This is Solution's own reading layer, so it works with Solution's
    # internals on purpose. The values themselves stay in the Solution, next to
    # the rows they were computed from, so a rollback cannot leave the two out
    # of step.
    # pylint: disable=protected-access

    def __init__(self, solution):
        """Attach a reading layer to ``solution``.

        Parameters
        ----------
        solution : Solution
            The flight whose post-process variables this reads. A ``Solution``
            builds its own, so there is rarely a reason to build one by hand.
        """
        self._solution = solution
        self._tables = None
        self._key = None

    def __repr__(self):
        """Return which variables this flight computes."""
        return f"PostProcessSolution(names={self.names!r})"

    @property
    def names(self):
        """Names of every post-process variable this flight computes.

        The phases of a flight need not all compute the same variables: a
        parachute descent reports no moments and no thrust. This lists every
        name any phase computes, in the order the phases first introduce them.
        A phase that does not compute one of them reports zero for it, which is
        the physically right answer for the variables RocketPy defines.
        """
        names = []
        for phase in self._solution.phases:
            for name in phase.dynamics.post_process_vars:
                if name not in names:
                    names.append(name)
        return tuple(names)

    def __contains__(self, name):
        """Return whether any phase of this flight computes ``name``."""
        return name in self.names

    def __iter__(self):
        """Iterate over :attr:`names`."""
        return iter(self.names)

    def _phase_table(self, phase_index, phase):
        """Return one phase's values as an ``(n_rows, n_variables)`` array.

        Values recorded while the simulation ran are used as they are. A phase
        with none is worked out again from its stored states, which gives the
        same answer for a flight whose rocket did not change mid-flight.

        Parameters
        ----------
        phase_index : int
            Position of the phase in the solution's phases.
        phase : _PhaseSolution
            That same phase.

        Returns
        -------
        numpy.ndarray or None
            One row of values per row of the phase, or ``None`` if the phase
            holds no rows or can neither be read back nor worked out again.
        """
        start, stop = self._solution.phase_span(phase_index)
        rows = self._solution._rows[start:stop]
        if not rows:
            return None
        recorded = self._solution._post_values[start:stop]
        if self._solution.records_post_values:
            # This flight changes the rocket as it flies, so the values it
            # recorded are the only right answer. Working them out again would
            # read the rocket in the configuration it ended the flight in and
            # report, for the whole flight, the air brake deflection it landed
            # with. A row with nothing recorded is a bug, not a reason to fall
            # back on that.
            missing = [index for index, values in enumerate(recorded) if values is None]
            if missing:
                raise ValueError(
                    f"This flight recorded its post-process variables as it "
                    f"ran, because one of its events changes the rocket "
                    f"mid-flight, but row {start + missing[0]} has none. They "
                    f"cannot be worked out now: the rocket is no longer in the "
                    f"configuration it had at that moment. This is a bug in "
                    f"RocketPy, not something a flight can cause."
                )
            return np.array(recorded, dtype=float)
        if phase.bound_dynamics is None:
            # A phase read back from a saved flight cannot be worked out again,
            # since that needs the equations of motion of a live flight.
            return None
        return np.array(
            [
                phase.dynamics.post_process_values(
                    phase.bound_dynamics.post_process_at(row[0], row[1:])
                )
                for row in rows
            ],
            dtype=float,
        )

    def _phase_tables(self):
        """Return every phase's values, working out the ones not recorded.

        The answer is kept until the flight's rows or its recorded values
        change, so asking for a second variable costs nothing.
        """
        key = (self._solution._version, self._solution._post_version)
        if self._tables is None or self._key != key:
            self._tables = [
                self._phase_table(index, phase)
                for index, phase in enumerate(self._solution.phases)
            ]
            self._key = key
        return self._tables

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
        tables = self._phase_tables()
        return tables[phase_index]

    def __getitem__(self, name):
        """Return the ``[t, value]`` history of one variable over the flight.

        Parameters
        ----------
        name : str
            The variable name, for example ``"az"`` or ``"net_thrust"``.

        Returns
        -------
        numpy.ndarray
            An ``(n, 2)`` array whose columns are time and value.

        Raises
        ------
        KeyError
            If no phase of this flight computes the variable, or if the flight
            has no stored rows, or if it was read back from a saved file and so
            cannot compute it.
        """
        names = self.names
        if name not in names:
            raise KeyError(
                f"No flight phase computed the post-process variable '{name}'. "
                f"This flight computes: {', '.join(names) or '(none)'}."
            )
        tables = self._phase_tables()
        parts = []
        for index, phase in enumerate(self._solution.phases):
            table = tables[index]
            if table is None:
                continue
            start, stop = self._solution.phase_span(index)
            times = self._solution.time[start:stop]
            column = phase.dynamics.post_process_index.get(name)
            values = np.zeros(len(times)) if column is None else table[:, column]
            parts.append(np.column_stack([times, values]))
        if not parts:
            raise KeyError(
                f"The post-process variable '{name}' cannot be worked out for "
                f"this flight. Either it has no stored states yet, or it was "
                f"read back from a saved file, which does not carry the "
                f"equations of motion needed to work the variable out again."
            )
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    def at(self, t, atol=1e-3):
        """Return every post-process variable at the stored time nearest ``t``.

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
            Variable name to value, holding the variables the phase that time
            falls in computes. Phases do not all compute the same variables, so
            a parachute descent reports fewer names than a powered ascent.
        """
        index = _nearest_time_index(self._solution.time, t, atol, "solution")
        return self.at_index(index)

    def at_index(self, index):
        """Return every post-process variable at row ``index``.

        Reads a single row, so it stays cheap no matter how long the flight is.
        Use it instead of ``post["az"][index, 1]`` when only one row is needed,
        since that works out the variable's whole history first.

        Parameters
        ----------
        index : int
            Row position across the whole flight. Negative values count from
            the end.

        Returns
        -------
        dict
            Variable name to value, as described in :meth:`at`. Empty for a row
            of a phase that computes no post-process variables.

        Raises
        ------
        IndexError
            If ``index`` is outside the flight's rows.
        KeyError
            If the row's values were not recorded and the phase was read back
            from a saved file, so they cannot be worked out again.
        """
        solution = self._solution
        position = solution._normalize(int(index))
        phase_index = solution.phase_index_at(position)
        phase = solution.phases[phase_index]
        if not phase.dynamics.post_process_vars:
            return {}
        values = solution._post_values[position]
        if values is None:
            if solution.records_post_values:
                raise ValueError(
                    f"This flight recorded its post-process variables as it "
                    f"ran, because one of its events changes the rocket "
                    f"mid-flight, but row {position} has none. They cannot be "
                    f"worked out now: the rocket is no longer in the "
                    f"configuration it had at that moment. This is a bug in "
                    f"RocketPy, not something a flight can cause."
                )
            if phase.bound_dynamics is None:
                raise KeyError(
                    "This flight phase was read back from a saved file, so its "
                    "post-process variables were not recorded and cannot be "
                    "worked out again: that needs the equations of motion, "
                    "which are code and are not saved with a flight."
                )
            row = solution.raw_row(position)
            values = phase.dynamics.post_process_values(
                phase.bound_dynamics.post_process_at(row[0], row[1:])
            )
        return dict(zip(phase.dynamics.post_process_vars, values))
