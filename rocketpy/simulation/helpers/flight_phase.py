import math


class _FlightPhases:
    """Class to handle flight phases. It is used to store the derivatives
    and events for each flight phase. It is also used to handle the
    insertion of flight phases in the correct order, according to their
    initial time.

    Attributes
    ----------
    list : list
        A list of _FlightPhase objects. See _FlightPhase subclass.
    """

    def __init__(
        self,
        t_initial,
        initial_derivative,
        max_time,
        verbose=False,
    ):
        self.list = []

        self.add_phase(
            t_initial,
            initial_derivative,
            name="initial_phase",
            clear=True,
        )
        self.add_phase(max_time, name="max_time_stop")

        self.verbose = verbose

    def __getitem__(self, index):
        return self.list[index]

    def __len__(self):
        return len(self.list)

    def __iter__(self):
        """Iterate as (index, phase) pairs excluding the last phase. The
        last phase is serves as a sentinel for phase transitions and should
        not be processed as a normal phase."""
        index = 0
        while index < len(self.list) - 1:
            yield index, self.list[index]
            index += 1

    def __repr__(self):
        return str(self.list)

    def display_warning(self, instant, order):  # pragma: no cover
        """Print a contextual warning when phases conflict.

        Parameters
        ----------
        instant : str
            Human-readable description of when the collision occurs
            (for example: "together", "before", "after").
        order : str
            Human-readable positional hint (for example: "preceding",
            "proceeding").

        Notes
        -----
        This method only prints when `self.verbose` is True. It exists to
        centralize the exact warning wording used across insertion helpers.
        """
        if self.verbose:
            message = (
                f"Trying to add flight phase starting {instant} with the "
                f"one {order} it. This may be caused by multiple events "
                "being triggered simultaneously."
            )
            print("WARNING:", message)

    def _try_append_to_tail(self, flight_phase):
        """Attempt to append `flight_phase` at the timeline tail.

        Behavior matches original implementation exactly:
        - If the new phase time is strictly greater than the last phase,
          append and fix time bounds.
        - If the new phase time equals the last phase, emit a warning,
          nudge the new phase forward by 1e-7 and append.
        - Otherwise, do not modify the list and return False.

        Returns
        -------
        bool
            True if the phase was appended (possibly after a nudge), False
            otherwise.
        """
        tail = self.list[-1]
        if flight_phase.t > tail.t:
            self.list.append(flight_phase)
            self._link_to_next(len(self.list) - 2)
            self._link_to_next(len(self.list) - 1)
            return True

        if flight_phase.t == tail.t:
            self.display_warning("together", "preceding")
            flight_phase.t += 1e-7
            self.list.append(flight_phase)
            self._link_to_next(len(self.list) - 2)
            self._link_to_next(len(self.list) - 1)
            return True

        return False

    def _link_to_next(self, index):
        """Ensure `self.list[index].time_bound` matches the next phase time.

        If `index` refers to the last (tail) phase, `time_bound` is set to
        ``None``. For interior phases it is set to the next phase's start
        time. Invalid indexes are ignored (no exception raised).
        """
        if 0 <= index < len(self.list) - 1:
            self.list[index].time_bound = self.list[index + 1].t
        elif 0 <= index < len(self.list):
            # Tail phase: no upper bound.
            self.list[index].time_bound = None

    def _normalize_insertion_index(self, index):
        """Normalize an insertion hint into a valid index in [0, len].

        - Negative indexes are interpreted relative to the current list
          (Python-style), e.g. -1 becomes len(self.list) - 1; then the
          result is clamped into the inclusive range [0, len(self.list)].
        """
        if index < 0:
            index += len(self.list)
        # Clamp to valid insertion range: 0..len(self.list)
        return min(max(index, 0), len(self.list))

    def _get_neighbor_phases(self, index):
        """Return (previous_phase, next_phase) for a candidate insertion index.

        If there is no previous or next phase (edges), the corresponding
        value is ``None``. This mirrors the original behavior used by the
        insertion loop.
        """
        previous_phase = self.list[index - 1] if index > 0 else None
        next_phase = self.list[index] if index < len(self.list) else None
        return previous_phase, next_phase

    def _resolve_time_collisions(self, flight_phase, previous_phase, next_phase):
        """Detect equal-time collisions and nudge the new phase forward.

        Returns True when a nudge was performed and the caller should retry
        the insertion checks (because the candidate time changed). The
        nudge amount is the same small constant (1e-7) used originally.
        """
        if previous_phase is not None and flight_phase.t == previous_phase.t:
            self.display_warning("together", "preceding")
            flight_phase.t += 1e-7
            return True

        if next_phase is not None and flight_phase.t == next_phase.t:
            self.display_warning("together", "proceeding")
            flight_phase.t += 1e-7
            return True

        return False

    def _adjust_index_direction(self, flight_phase, index, previous_phase, next_phase):
        """If the candidate index is not correctly ordered, shift it.

        Returns the adjusted index (can be index-1, index+1, or the same
        index). This preserves the original behavior and warnings.
        """
        if previous_phase is not None and flight_phase.t < previous_phase.t:
            self.display_warning("before", "preceding")
            return index - 1

        if next_phase is not None and flight_phase.t > next_phase.t:
            self.display_warning("after", "proceeding")
            return index + 1

        return index

    def _find_insertion_index(self, flight_phase, start_index):
        """Find the correct insertion index for `flight_phase`.

        The loop here encapsulates the original while-loop logic and returns
        a stable index where the new phase can be inserted. The method
        mutates `flight_phase.t` when necessary (to nudge equal-times) to
        preserve the exact semantics of the original implementation.
        """
        index = self._normalize_insertion_index(start_index)
        while True:
            previous_phase, next_phase = self._get_neighbor_phases(index)

            # If we had an equal-time collision, _resolve_time_collisions
            # will nudge `flight_phase.t` and request another iteration.
            if self._resolve_time_collisions(flight_phase, previous_phase, next_phase):
                continue

            # If ordering is incorrect for this index, shift and retry.
            new_index = self._adjust_index_direction(
                flight_phase, index, previous_phase, next_phase
            )
            if new_index != index:
                index = new_index
                continue

            # Index is valid: previous.t < flight_phase.t < next.t (or edge)
            return index

    def add(self, flight_phase, index=None):
        """Insert `flight_phase` in chronological order preserving behavior.

        This method keeps the exact behavior of the previous implementation
        but delegates the searching logic to `_find_insertion_index` for
        clarity. Side effects (warnings, nudges, and time_bound updates)
        remain identical.
        """
        phases = self.list

        # Fast path: empty list -- the first phase becomes the start.
        if not phases:
            phases.append(flight_phase)
            self._link_to_next(0)
            return None

        # If caller provided no hint, prefer appending when possible.
        if index is None:
            if self._try_append_to_tail(flight_phase):
                return None
            # Start searching from the tail when append didn't apply.
            index = len(phases)

        # Find the final insertion position (this may nudge flight_phase.t).
        final_index = self._find_insertion_index(flight_phase, index)

        # Perform insertion and update only the local time bounds that
        # are affected, matching the original code's link updates.
        phases.insert(final_index, flight_phase)
        self._link_to_next(final_index - 1)
        self._link_to_next(final_index)

        return None

    def add_phase(
        self,
        t,
        derivatives=None,
        event=None,
        index=None,
        name=None,
        **kwargs,
    ):
        """Add a new flight phase to the list, with the specified
        characteristics. This method creates a new _FlightPhase instance and
        adds it to the flight phases list, either at the specified index
        position or appended to the end. See _FlightPhases.add() for more
        information.

        Parameters
        ----------
        t : float
            The initial time of the new flight phase.
        derivatives : function, optional
            A function representing the derivatives of the flight phase.
            Default is None.
        event : list of functions, optional
            A list of events to be executed during the flight
            phase. Default is None. You can also pass an empty list.
        index : int, optional
            The index at which the new flight phase should be inserted.
            If not provided, the flight phase will be appended
            to the end of the list. Default is None.
        name : str, optional
            A descriptive name to identify the phase in logs and
            debug output. Default is None.

        Returns
        -------
        None
        """
        self.add(
            _FlightPhase(
                t,
                derivative=derivatives,
                events=event,
                name=name,
                **kwargs,
            ),
            index,
        )

    def flush_after(self, index):
        """This function deletes all flight phases after a given index.

        Parameters
        ----------
        index : int
            The index of the last flight phase to be kept.

        Returns
        -------
        None
        """
        del self.list[index + 1 :]

        # After flush, only the retained tail changes terminal state.
        if self.list:
            self._link_to_next(len(self.list) - 1)


class _FlightPhase:
    """Store a single flight phase with its time, derivatives, and events.

    This class encapsulates one discrete interval in the flight timeline.
    The `time_bound` is managed by _FlightPhases and marks the end time
    (upper boundary) of this phase in the overall simulation.

    Attributes
    ----------
    t : float
        Start time (in seconds) of this flight phase.
    derivative : callable, optional
        Function computing state derivatives during this phase.
    events : list of callable, optional
        Events to be evaluated or triggered during this phase.
    name : str, optional
        Descriptive label for logging and debug output.
    time_bound : float, optional
        Upper time boundary of this phase (managed by _FlightPhases).
    parachute : optional
        Parachute reference used only in post-processing.
    """

    def __init__(
        self,
        t,
        derivative=None,
        events=None,
        name=None,
        clear=False,
        **kwargs,
    ):
        """Initialize a flight phase.

        Parameters
        ----------
        t : float
            Start time of the phase.
        derivative : callable, optional
            State derivative function for this phase.
        events : list of callable, optional
            Event hooks to evaluate during this phase.
        name : str, optional
            Human-readable phase identifier.
        clear : bool, optional
            If True, clear events from the first time node of this phase.
            Useful for avoiding event checks at t=0 when state is incomplete.
            Default is False.
        **kwargs
            Additional attributes (e.g., parachute) used in post-processing.
        """
        self.t = t
        self.derivative = derivative
        self.events = events
        self.name = name
        self.clear = clear
        self.time_bound = None
        self.parachute = kwargs.get("parachute", None)

    def __repr__(self):
        """Return compact machine-readable representation."""
        derivative_name = getattr(self.derivative, "__name__", self.derivative.__class__.__name__)
        return (
            "_FlightPhase("
            f"t={self.t!r}, "
            f"name={self.name!r}, "
            f"derivative={derivative_name!r}, "
            f"time_bound={self.time_bound!r}"
            ")"
        )

    def __str__(self):
        """Return human-readable multi-line representation."""
        return self.__repr__()


class _TimeNodes:
    """_TimeNodes is a class that stores all the time nodes of a simulation.
    It is meant to work like a python list, but it has some additional
    methods that are useful for the simulation. Them items stored in are
    _TimeNodes object are instances of the _TimeNode class.
    """

    def __init__(self, init_list=None):
        self.list = [] if not init_list else init_list[:]
        self.continuous_events = []

    def __getitem__(self, index):
        return self.list[index]

    def __delitem__(self, key):
        del self.list[key]

    def __len__(self):
        return len(self.list)

    def __iter__(self):
        """Iterate as (index, node) pairs excluding the last node.

        This iterator checks list length dynamically at each step so items
        appended during iteration can still be visited in the same pass.
        """
        index = 0
        while index < len(self.list) - 1:
            node = self.list[index]
            yield index, node
            if index < len(self.list) and self.list[index] is node:
                index += 1

    def __repr__(self):
        return str(self.list)

    def add(self, time_node):
        """Append a pre-built time node to the list.

        Parameters
        ----------
        time_node : _TimeNode
            The time node to append.
        """
        self.list.append(time_node)

    def add_node(self, t, events):
        """Create and append a new time node.

        Parameters
        ----------
        t : float
            Time value for the node.
        events : list of callable
            Events to attach to this node.
        """
        self.list.append(_TimeNode(t, events))

    def add_continuous_events(self, events):
        """Register continuous events to be checked at every time step.

        Continuous events (without discrete sampling) are stored separately
        and evaluated at each time node in the simulation.

        Parameters
        ----------
        events : list of callable
            Continuous event hooks to add.
        """
        self.continuous_events += events

    def add_discrete_events(self, events, t_init, t_end):
        """Create and register time nodes for events with sampling rates.

        For each enabled event with a defined sampling_rate, this method
        creates time nodes at fixed intervals (1/sampling_rate) and adds the
        event to each node. Only nodes within [t_init, t_end] are kept.

        Parameters
        ----------
        events : list of callable
            Events with sampling_rate attributes to discretize.
        t_init : float
            Simulation start time.
        t_end : float
            Simulation end time.
        """
        for event in events:
            # Allow disabled events to be registered only if they provide an
            # `enable_on` predicate. Events that are disabled and have no
            # `enable_on` cannot become enabled again and are therefore
            # skipped here to avoid scheduling unnecessary checks.
            if (
                (not event.enabled and event.enable_on is None)
                or event.sampling_rate is None
            ):
                continue

            sampling_interval = event.sampling_interval
            # TODO: this is the inefficient part of the code
            # we are creating a list of time nodes for each event and then we 
            # are merging them later
            # A smarter way to do this is to only create the next time node
            # when needed. Too complex though
            node_list = [
                _TimeNode(
                    i * sampling_interval,
                    [event],
                )
                for i in range(
                    math.ceil(t_init / sampling_interval),
                    math.floor(t_end / sampling_interval) + 1,
                )
            ]
            self.list += node_list

    def add_event(self, event, t_init, t_end):
        """Register an event as continuous or discrete based on its properties.

        Parameters
        ----------
        event : callable
            Event object with optional sampling_rate attribute.
        t_init : float
            Simulation start time.
        t_end : float
            Simulation end time.
        """
        # Register the event only when it's present and either enabled or
        # able to re-enable itself via `enable_on`.
        if not event.enabled and event.enable_on is None:
            return

        if event.sampling_rate is None:
            self.add_continuous_events([event])
        else:
            self.add_discrete_events([event], t_init, t_end)

    def add_event_list(self, events, t_init, t_end):
        """Register multiple events as continuous or discrete.

        Parameters
        ----------
        events : list of callable
            Event objects to register.
        t_init : float
            Simulation start time.
        t_end : float
            Simulation end time.
        """
        for event in events:
            self.add_event(event, t_init, t_end)

    def disable_event(self, node_index, event):
        """Remove an event from the timeline starting at a given node index.

        - For continuous events, removes from the continuous_events list.
        - For discrete events, removes from nodes after node_index and drops
        any now-empty future nodes.

        Parameters
        ----------
        node_index : int
            Index of the first node to consider for removal.
        event : callable
            Event to disable.
        """
        if event.sampling_rate is None:
            self.continuous_events.remove(event)
        else:
            empty_node_indices = []
            for node_list_index, node in enumerate(
                self.list[node_index + 1 :], start=node_index + 1
            ):
                # Remove all occurrences of the event in this node
                while event in node.events:
                    node.events.remove(event)
                # Keep the final sentinel node even if empty.
                if len(node.events) == 0 and node_list_index != len(self.list) - 1:
                    empty_node_indices.append(node_list_index)

            # Delete now-empty nodes starting from the end to keep indices valid
            for node_list_index in reversed(empty_node_indices):
                del self.list[node_list_index]

    def enable_event(self, node_index, event):
        """Add an event to the timeline starting at a given node index.

        - For continuous events, adds to the continuous_events list.
        - For discrete events, creates and adds time nodes at sampling intervals
          from node_index to t_end, then sorts and merges to consolidate.

        Parameters
        ----------
        node_index : int
            Index of the first node to enable the event for.
        event : callable
            Event to enable.
        t_end : float, optional
            End time for discrete event nodes. If not provided, uses the last
            node's time. Default is None.
        """
        t_end = self.list[-1].t

        if event.sampling_rate is None:
            # Continuous event: add to continuous_events if not already present
            if event not in self.continuous_events:
                self.continuous_events.append(event)
        else:
            # Discrete event: create time nodes at sampling intervals
            t_init = self.list[node_index].t if node_index < len(self.list) else 0
            self.add_discrete_events([event], t_init, t_end)
            # Consolidate: sort and merge to handle overlapping times
            self.sort()
            self.merge()

    def sort(self):
        """Sort time nodes by their time values in ascending order."""
        self.list.sort()

    def merge(self):
        """Consolidate nodes that have the same rounded time value.

        This prevents duplicate time node evaluations by merging events from
        nodes with the same time (rounded to 7 decimals). The result order
        is not guaranteed, so sort() should be called before or after if
        a specific order is needed.
        """
        merged_nodes = {}
        for node in self.list:
            rounded_time = round(node.t, 7)
            if rounded_time in merged_nodes:
                # Merge events from duplicate time nodes.
                merged_nodes[rounded_time].events += node.events
            else:
                # First occurrence of this time value.
                merged_nodes[rounded_time] = node
        self.list = list(merged_nodes.values())

    def flush_after(self, index):
        """Delete all time nodes after a given index.

        Parameters
        ----------
        index : int
            Nodes at index+1 and beyond are removed.
        """
        del self.list[index + 1 :]


class _TimeNode:
    """A snapshot of events to evaluate at a specific simulation time.

    This class pairs a time value with the events that should be evaluated
    (or triggered) at that moment. It is designed to work exclusively within
    the _TimeNodes collection and should not be used independently.

    Attributes
    ----------
    t : float
        Time value (in seconds) at which events occur.
    events : list of callable
        Event hooks to evaluate at time t.
    """

    def __init__(self, t, events):
        """Initialize a time node.

        Parameters
        ----------
        t : float
            Time value for this node.
        events : list of callable, optional
            Event hooks. A shallow copy is made to avoid external mutation.
        """
        self.t = t
        # Make a shallow copy of events to prevent external mutation.
        self.events = events[:] if events is not None else []

    def __repr__(self):
        return f"<_TimeNode(t: {self.t}, events: {len(self.events)})>"

    def __lt__(self, other):
        """Allows the comparison of two _TimeNode objects based on their
        initial time. This is particularly useful for sorting a list of
        _TimeNode objects.

        Parameters
        ----------
        other : _TimeNode
            Another _TimeNode object to compare with.

        Returns
        -------
        bool
            True if the initial time of the current _TimeNode is less
            than the initial time of the other _TimeNode, False
            otherwise.
        """
        return self.t < other.t
