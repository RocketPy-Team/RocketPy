"""Container for the state history produced by a flight simulation.

A flight is integrated in phases (rail, powered ascent, coast, parachute
descent, ...), and a phase may integrate fewer than the thirteen canonical
states — a phase following only position and velocity would store seven values
per row instead of fourteen. Whatever a phase stores, the whole flight still
reads as one list of 14-value rows and answers queries by state name. All of
RocketPy's own phases integrate the full canonical state, for the reasons given
in :mod:`rocketpy.simulation.helpers.dynamics`.

The full 13-state vector ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]``
is the *canonical* state: every phase can report it, filling in the states it
does not integrate. Plots, outputs and user callbacks all read that.

The pieces are:

- :class:`Solution`: the whole flight. It holds every row in one list, in
  flight order, and the phases the flight was flown in.
- :class:`_PhaseSolution`: one flight phase. It describes the phase (which
  dynamics it was flown with, when it began, where its rows start) but does not
  hold the rows itself.

Keeping the rows in one list means row *i* of the flight is just row *i* of that
list, no matter which phase it belongs to. Each phase records the index of its
first row, so the phase a row belongs to is found by a quick search over those
indices whenever the row's meaning is needed.

What states a phase integrates, and how its canonical state is rebuilt, is
described by its ``_PhaseDynamics``. Post-process variables (accelerations,
aerodynamic forces and moments, net thrust) are kept beside the rows they were
computed from, so the two always share a time. A flight whose controllers move
an air brake records them as it runs, since replaying the stored states
afterwards would read the rocket in its end-of-flight configuration; any other
flight leaves them empty and the
:class:`~rocketpy.simulation.flight.Flight` works them out once the simulation
is over.

Reading a :class:`Solution` as a list always gives 14-value canonical rows, the
same shape older versions of RocketPy stored. Writing to it works in the raw
states of the phase being flown, which is what the integrator produces.
"""

import warnings
from bisect import bisect_right

import numpy as np

from .helpers.dynamics import (
    CANONICAL_INDEX,
    CANONICAL_STATE_NAMES,
    SIX_DOF_DYNAMICS,
    _BoundDynamics,
    dynamics_for_name,
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
    :attr:`start` is the position of this phase's first row in it. Read a
    phase's rows with :meth:`Solution.phase_rows`, its times with
    :meth:`Solution.phase_time`, and one of its states by name with
    :meth:`Solution.phase_series`.

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
            Default is ``0``. :meth:`Solution.start_phase` fills this in.

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
        if start_canonical is None and self.dynamics.frozen_states:
            raise ValueError(
                f"This flight phase does not integrate "
                f"{', '.join(self.dynamics.frozen_states)}, so it needs the state "
                f"at the start of the phase to hold them at. Pass start_canonical."
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
        its canonical time derivative is zero; a reconstructed one gets the value
        its own rule computes, which needs ``state``, the phase's state at the
        same instant.

        Parameters
        ----------
        state_dot : sequence of float
            Time derivative of one of this phase's states, in the order the
            phase integrates them.
        state : sequence of float, optional
            The phase's state at the same instant. Only needed by a phase that
            reconstructs a state, whose rule reads the state values. Default is
            ``None``.

        Returns
        -------
        sequence of float
            The 13 canonical time derivatives.

        Raises
        ------
        ValueError
            If this phase reconstructs a state but ``state`` was not given, so
            the reconstruction rules have nothing to read.
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
            canonical states (reconstructed or frozen where this phase does not
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
            A description produced by :meth:`to_dict`. A description saved by
            an older RocketPy also carries the phase's rows; they are ignored
            here, since :class:`Solution` reads them itself.

        Returns
        -------
        _PhaseSolution
            The rebuilt phase. Its ``bound_dynamics`` is ``None``, since tying
            the dynamics to a flight needs a running simulation, so the phase
            cannot be post-processed again.
        """
        return cls(
            dynamics_for_name(data.get("dynamics"), data["state_names"]),
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
    phases, each a :class:`_PhaseSolution`; read one phase's rows with
    :meth:`phase_rows` and one of its states with :meth:`phase_series`.

    It also behaves like a plain list of rows: ``len(solution)``, iteration,
    ``solution[-1]``, slicing and ``numpy.array(solution)`` all give 14-value
    canonical rows ``[t, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]``,
    whatever the phase they came from actually integrated.

    Writing works the other way round: ``append``, ``insert``, ``pop`` and row
    assignment all take the raw states of the phase being flown, since that is
    what the integrator produces. Those are simulation internals; prefer
    :meth:`replace_last`, :meth:`insert_before_last` and :meth:`drop_last`,
    which say what they do.
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
        self._phases = list(phases) if phases else []
        self._rows = list(rows) if rows else []
        # Mirrors phase.start. Kept alongside the phases so the search for the
        # phase owning a row runs over a plain list of numbers.
        self._starts = [phase.start for phase in self._phases]
        # Post-process values, one entry per row and in the same order, so a row
        # and its values can never drift apart. They carry no time of their own:
        # the row they sit beside supplies it. ``None`` where nothing was
        # recorded. Not saved to file: a solution read back has no live flight
        # to post-process against.
        self._post_values = [None] * len(self._rows)
        self._version = 0
        self._series_cache = {}
        self._canonical_cache = None
        self._canonical_version = -1
        self._time_cache = None
        self._time_version = -1
        self._validate_starts()

    def __repr__(self):
        """Return how many phases and how many rows this solution holds."""
        return f"Solution(phases={len(self._phases)}, rows={len(self._rows)})"

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
        if not self._phases:
            if total:
                raise ValueError(
                    "This solution has rows but no flight phases, so there is "
                    "nothing to say what its states mean."
                )
            return
        if self._starts[0] != 0:
            raise ValueError(
                f"The first flight phase must start at the first row, but it "
                f"starts at row {self._starts[0]}."
            )
        for earlier, later in zip(self._starts, self._starts[1:]):
            if later < earlier:
                raise ValueError(
                    "Flight phases must be stored in the order they were flown."
                )
        # A phase starting exactly at the end is a phase that has only just
        # begun and holds no rows yet, which is normal mid-flight.
        if self._starts[-1] > total:
            raise ValueError(
                f"A flight phase starts at row {self._starts[-1]}, but this "
                f"solution only has {total} rows."
            )
        for _, phase, start, stop in self._spans():
            for row in self._rows[start:stop]:
                self._check_width(phase, row, "stored in")

    # -- Phase management ------------------------------------------------

    def start_phase(self, dynamics, start_canonical, t_start=None, name=None):
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
            :meth:`append`.

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
        self._phases.append(phase)
        self._starts.append(phase.start)
        self._version += 1
        return phase

    @property
    def phases(self):
        """This flight's phases as a tuple, in the order they were flown."""
        return tuple(self._phases)

    @property
    def tail(self):
        """The current (most recent) phase, whether or not it has rows yet.

        This is the phase the simulation is flying now. Right after a new phase
        begins it holds no rows, and the most recent row still belongs to the
        phase before it; use :attr:`last_phase` when you want the phase that
        owns that row.

        Raises an ``IndexError`` if no phase has been started.
        """
        return self._phases[-1]

    @property
    def last_phase(self):
        """The phase that owns the most recent row.

        A phase that has only just begun holds no rows, so this skips it and
        gives the phase before it.

        Raises an ``IndexError`` if the flight has no rows yet.
        """
        if not self._rows:
            raise IndexError("this solution has no stored rows")
        return self._phase_at(len(self._rows) - 1)

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
        # The last phase starting at or before this row owns it. A phase that
        # holds no rows shares its successor's start, and is passed over.
        return self._phases[bisect_right(self._starts, index) - 1]

    def _spans(self):
        """Yield ``(index, phase, start, stop)`` for every phase holding rows.

        ``start`` and ``stop`` bound the phase's rows in the flight's row list,
        as a Python slice would. Phases with no rows are skipped.
        """
        starts, total = self._starts, len(self._rows)
        for index, phase in enumerate(self._phases):
            start = starts[index]
            stop = starts[index + 1] if index + 1 < len(starts) else total
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
        return bisect_right(self._starts, self._normalize(int(index))) - 1

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
        count = len(self._phases)
        if phase_index < 0:
            phase_index += count
        if phase_index < 0 or phase_index >= count:
            raise IndexError("solution phase index out of range")
        start = self._starts[phase_index]
        stop = (
            self._starts[phase_index + 1]
            if phase_index + 1 < count
            else len(self._rows)
        )
        return start, stop

    @property
    def penultimate_raw_time(self):
        """The time of the second-to-last stored row, in seconds.

        This is the time the flight was at one step before the latest one, which
        is what the elapsed step size is measured against. Rows are counted
        across the whole flight, so the answer is right even when the last step
        began in the previous flight phase.

        Returns
        -------
        float or None
            The time of the second-to-last row, or ``None`` if the flight does
            not have two rows yet.
        """
        rows = self._rows
        return rows[-2][0] if len(rows) > 1 else None

    @property
    def last_time(self):
        """The time of the most recent row, in seconds.

        Raises an ``IndexError`` if the flight has no rows yet.
        """
        return self._rows[-1][0]

    @property
    def last_state(self):
        """The most recent state, in its own phase's states, without time.

        Raises an ``IndexError`` if the flight has no rows yet.
        """
        return self._rows[-1][1:]

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

    def phase_rows(self, phase_index):
        """Return one phase's rows, each ``[t, *state]`` as it was stored.

        Parameters
        ----------
        phase_index : int
            Position of the phase in :attr:`phases`. Negative values count
            from the end.

        Returns
        -------
        list of list of float
            The phase's rows, in flight order. Empty for a phase with no rows
            yet. This is a new list, so changing it does not touch the flight.

        Raises
        ------
        IndexError
            If ``phase_index`` is outside this flight's phases.
        """
        start, stop = self.phase_span(phase_index)
        return self._rows[start:stop]

    def phase_time(self, phase_index):
        """Return the time column of one phase, in seconds, as a 1-D array.

        Parameters
        ----------
        phase_index : int
            Position of the phase in :attr:`phases`. Negative values count
            from the end.

        Returns
        -------
        numpy.ndarray
            The phase's times, in flight order. Empty for a phase with no rows.

        Raises
        ------
        IndexError
            If ``phase_index`` is outside this flight's phases.
        """
        start, stop = self.phase_span(phase_index)
        if start == stop:
            return np.empty(0)
        return np.array([row[0] for row in self._rows[start:stop]])

    def phase_canonical_array(self, phase_index):
        """Return one phase as a rectangular ``(n, 14)`` canonical table.

        Each row is ``[t, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]``.
        States the phase does not integrate are reconstructed or held at their
        value when the phase began.

        Parameters
        ----------
        phase_index : int
            Position of the phase in :attr:`phases`. Negative values count
            from the end.

        Returns
        -------
        numpy.ndarray
            The phase's canonical table, ``(0, 14)`` for a phase with no rows.

        Raises
        ------
        IndexError
            If ``phase_index`` is outside this flight's phases.
        """
        start, stop = self.phase_span(phase_index)
        if start == stop:
            return np.empty((0, CANONICAL_WIDTH))
        phase = self._phases[phase_index]
        rows = self._rows[start:stop]
        if phase.dynamics.is_canonical:
            return np.array(rows)
        canonical_state = phase.canonical_state
        return np.array([[row[0], *canonical_state(row[1:])] for row in rows])

    def phase_post_values(self, phase_index):
        """Return the post-process values recorded for one phase's rows.

        Parameters
        ----------
        phase_index : int
            Position of the phase in :attr:`phases`. Negative values count
            from the end.

        Returns
        -------
        list
            One entry per row of the phase, in flight order, each the values in
            that phase's ``post_process_vars`` order. An entry is ``None``
            where nothing was recorded for that row.

        Raises
        ------
        IndexError
            If ``phase_index`` is outside this flight's phases.
        """
        start, stop = self.phase_span(phase_index)
        return self._post_values[start:stop]

    def phase_series(self, phase_index, name):
        """Return one phase's ``[t, value]`` history for a single state.

        Parameters
        ----------
        phase_index : int
            Position of the phase in :attr:`phases`. Negative values count
            from the end.
        name : str
            The state name (for example ``"vz"``).

        Returns
        -------
        numpy.ndarray
            An ``(n, 2)`` array whose columns are time and value.

        Raises
        ------
        IndexError
            If ``phase_index`` is outside this flight's phases.
        KeyError
            If the phase does not define the state, or has no stored rows yet.
        """
        start, stop = self.phase_span(phase_index)
        phase = self._phases[phase_index]
        result = self._phase_series_or_none(phase, start, stop, name)
        if result is None:
            if start == stop:
                raise KeyError("This flight phase has no stored states yet.")
            raise KeyError(
                f"State '{name}' is not defined in this flight phase. "
                f"It integrates {', '.join(phase.dynamics.states)}."
            )
        return result

    def _phase_series_or_none(self, phase, start, stop, name):
        """Return a phase's ``[t, value]`` history, or ``None`` if undefined.

        The forgiving version of :meth:`phase_series`, so :meth:`series` can
        skip the phases that do not define a state while gathering its history
        across the whole flight.

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
        dynamics = phase.dynamics
        # A state the phase integrates is a plain column of its rows; anything
        # else is reconstructed or held at its start-of-phase value.
        index = dynamics.state_index.get(name)
        if index is not None:
            return np.column_stack([times, [row[index + 1] for row in rows]])
        position = dynamics.reconstructed_index.get(name)
        if position is not None:
            values = [
                dynamics.reconstruct(
                    dynamics.reconstruction_inputs(row[1:], phase.start_canonical)
                )[position]
                for row in rows
            ]
            return np.column_stack([times, values])
        if name in CANONICAL_INDEX and phase.start_canonical is not None:
            constant = phase.start_canonical[CANONICAL_INDEX[name]]
            return np.column_stack([times, np.full(len(rows), constant)])
        return None

    # -- Mutation ----------------------------------------------------------

    def _check_width(self, phase, row, what):
        """Raise if ``row`` does not have the width ``phase`` stores.

        Parameters
        ----------
        phase : _PhaseSolution
            The phase the row is headed for.
        row : sequence of float
            The row being written, ``[t, *state]``.
        what : str
            How the row is being written, used in the error message, for
            example ``"appended to"`` or ``"inserted into"``.

        Raises
        ------
        ValueError
            If the row's length is not time plus the phase's state count.
        """
        expected = phase.dynamics.width + 1
        if len(row) != expected:
            raise ValueError(
                f"State row of length {len(row)} does not match the flight "
                f"phase it is being {what} (expected {expected} values: time "
                f"plus {phase.dynamics.width} states)."
            )

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
        phases, starts = self._phases, self._starts
        index = len(phases) - 1
        while index >= 0 and starts[index] > position:
            starts[index] += delta
            phases[index].start = starts[index]
            index -= 1

    def append(self, row):
        """Append a raw state row ``[t, *state]`` to the current phase.

        Parameters
        ----------
        row : sequence of float
            Time followed by the states the current phase integrates, in the
            order that phase integrates them.

        Raises
        ------
        ValueError
            If the row's length does not match the current phase.
        IndexError
            If no phase has been started yet.
        """
        phase = self._phases[-1]
        # Compared here rather than in _check_width because this runs on every
        # solver step; _check_width is only reached to build the error.
        if len(row) != phase.dynamics.width + 1:
            self._check_width(phase, row, "appended to")
        self._rows.append(row)
        self._post_values.append(None)
        self._version += 1

    def set_last_post_values(self, values):
        """Record the post-process values of the most recent row.

        Called as the simulation runs, right after the row is appended, so the
        values are the ones the rocket really had at that step. They are kept
        beside the row and move with it, so a rollback cannot leave the two out
        of step.

        Parameters
        ----------
        values : sequence of float
            The values in the current phase's ``post_process_vars`` order.

        Raises
        ------
        IndexError
            If the flight has no rows yet.
        """
        if not self._post_values:
            raise IndexError("this solution has no stored rows")
        # Deliberately does not count as a change to the flight: the states are
        # untouched, so the cached times and tables stay valid.
        self._post_values[-1] = values

    def replace_last(self, row):
        """Overwrite the most recent row with a raw state row ``[t, *state]``.

        Parameters
        ----------
        row : sequence of float
            Time followed by the states the row's phase integrates, in the
            order that phase integrates them.

        Raises
        ------
        ValueError
            If the row's length does not match the phase it lands in.
        IndexError
            If the flight has no rows yet.
        """
        self._check_width(self.last_phase, row, "written to")
        self._rows[-1] = row
        self._post_values[-1] = None
        self._version += 1

    def insert_before_last(self, row):
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
        ValueError
            If the row's length does not match the phase it lands in.
        IndexError
            If the flight has no rows yet.
        """
        self._check_width(self.last_phase, row, "inserted into")
        position = len(self._rows) - 1
        self._rows.insert(position, row)
        self._post_values.insert(position, None)
        self._shift_starts_after(position, 1)
        self._version += 1

    def drop_last(self):
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

    def insert(self, index, row):
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
        ValueError
            If the row's length does not match the phase it lands in.
        """
        position = self._normalize(int(index))
        self._check_width(self._phase_at(position), row, "inserted into")
        self._rows.insert(position, row)
        self._post_values.insert(position, None)
        self._shift_starts_after(position, 1)
        self._version += 1

    def pop(self, index=-1):
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

    # -- List-like behaviour (canonical rows) ------------------------------

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
            return list(self)[key]
        if isinstance(key, (int, np.integer)):
            return self.canonical_row(int(key))
        raise TypeError(
            "Solution indices must be integers, slices, or state names, "
            f"not {type(key).__name__}."
        )

    def __setitem__(self, index, row):
        """Overwrite the row at ``index`` with a raw state row ``[t, *state]``.

        Unlike reading, which always reports canonical rows, writing takes the
        raw states of the phase the row lands in.

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
        TypeError
            If ``index`` is not an integer.
        IndexError
            If ``index`` is outside the flight's rows.
        ValueError
            If the row's length does not match the phase it lands in.
        """
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Solution only supports integer row assignment.")
        position = self._normalize(int(index))
        self._check_width(self._phase_at(position), row, "written to")
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
            Part of the NumPy array protocol, accepted and not acted on.

        Returns
        -------
        numpy.ndarray
            The ``(n, 14)`` canonical table.
        """
        # With no dtype asked for, this hands back the cached canonical table
        # rather than rebuilding it. `numpy.array` copies by default, so the
        # cache is safe from the usual call.
        array = self.canonical_array
        return array.astype(dtype) if dtype is not None else array

    # -- Queries by name ---------------------------------------------------

    def series(self, name):
        """Return the ``[t, value]`` history of one state over the flight.

        The history spans only the phases that define the state. States
        in the canonical state are defined in every phase (reconstructed or
        frozen where a phase does not integrate them), so their history covers
        the whole flight. A phase-specific state's history covers only the
        phases that integrate it, and a warning names the phases left out.

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
        dynamics = phase.dynamics
        # A state the phase integrates is a value in the row itself; anything
        # else is reconstructed or held at its start-of-phase value.
        column = dynamics.state_index.get(name)
        if column is not None:
            return row[column + 1]
        slot = dynamics.reconstructed_index.get(name)
        if slot is not None:
            return dynamics.reconstruct(
                dynamics.reconstruction_inputs(row[1:], phase.start_canonical)
            )[slot]
        if name in CANONICAL_INDEX and phase.start_canonical is not None:
            return phase.start_canonical[CANONICAL_INDEX[name]]
        raise KeyError(
            f"State '{name}' is not defined in this flight phase. "
            f"It integrates {', '.join(dynamics.states)}."
        )

    @property
    def time(self):
        """The time column of the whole flight, in seconds, as a 1-D array."""
        if self._time_cache is None or self._time_version != self._version:
            rows = self._rows
            self._time_cache = np.fromiter(
                (row[0] for row in rows), dtype=float, count=len(rows)
            )
            self._time_version = self._version
        return self._time_cache

    @property
    def canonical_array(self):
        """The whole flight as a rectangular ``(n, 14)`` canonical table.

        Each row is ``[t, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]``.
        States a phase does not integrate are reconstructed or frozen.
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
                result = np.vstack(
                    [self.phase_canonical_array(index) for index, _, _, _ in spans]
                )
            self._canonical_cache = result
            self._canonical_version = self._version
        return self._canonical_cache

    # -- Serialization -----------------------------------------------------

    def to_dict(self):
        """Return a serializable description of the whole solution.

        Returns
        -------
        dict
            A format marker, a version number, the flight's rows, and one
            entry per phase as produced by :meth:`_PhaseSolution.to_dict`. Pass
            it to :meth:`from_dict` to rebuild the solution.
        """
        return {
            "format": "rocketpy/solution",
            "version": 2,
            "phases": [phase.to_dict() for phase in self._phases],
            "rows": self._rows,
        }

    @classmethod
    def from_dict(cls, data):
        """Rebuild a solution from its serialized form.

        Reads both the current layout, where the rows are stored once for the
        whole flight, and the earlier one, where each phase carried its own
        rows.

        Parameters
        ----------
        data : dict
            A description produced by :meth:`to_dict`, or by an older
            RocketPy.

        Returns
        -------
        Solution
            The rebuilt solution. Its phases cannot be post-processed again,
            since that needs a running simulation.
        """
        if data.get("version", 1) >= 2 or "rows" in data:
            return cls._from_flat_dict(data)
        return cls._from_nested_dict(data)

    @classmethod
    def _from_flat_dict(cls, data):
        """Rebuild a solution whose rows are stored once for the whole flight.

        Parameters
        ----------
        data : dict
            A description produced by :meth:`to_dict`.

        Returns
        -------
        Solution
            The rebuilt solution.
        """
        phases = [_PhaseSolution.from_dict(entry) for entry in data.get("phases", [])]
        rows = [list(row) for row in data.get("rows", [])]
        return cls(phases, rows)

    @classmethod
    def _from_nested_dict(cls, data):
        """Rebuild a solution saved with each phase carrying its own rows.

        Parameters
        ----------
        data : dict
            A description saved by an older RocketPy, whose phase entries each
            hold a ``"rows"`` list.

        Returns
        -------
        Solution
            The rebuilt solution, with the rows joined into one list.
        """
        phases = []
        rows = []
        for entry in data.get("phases", []):
            phase = _PhaseSolution.from_dict(entry)
            # Any stored start belongs to the phase's own rows, so it is
            # replaced by where those rows land in the joined list.
            phase.start = len(rows)
            phases.append(phase)
            rows.extend(list(row) for row in entry.get("rows", []))
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
