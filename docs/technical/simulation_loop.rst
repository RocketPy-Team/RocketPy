Simulation Loop
===============

This page describes how a ``Flight`` advances the simulation: how the flight is
split into phases, how the solver is driven inside each phase, when events are
checked, and what happens when an event changes the course of the flight. It
documents the loop in ``rocketpy/simulation/flight.py`` and its helpers in
``rocketpy/simulation/helpers/``. For how to write an event, see
:ref:`eventusage`.

Overview
--------

A flight is integrated one **phase** at a time. A phase is a stretch of the
flight flown with one set of equations of motion (on the rail, powered ascent,
free flight, parachute descent, ...). Inside a phase, the timeline is split at
**time nodes**: the moments where the solver must stop so that events can be
checked exactly there. Between two nodes the solver takes as many steps as it
likes, and events are checked after every step.

Events observe the flight and, when they fire, queue **commands**: enable or
disable events, add events, switch the equations of motion, start a new phase,
or end the flight. The loop applies those commands and adjusts the phase list
and node schedule accordingly.

.. mermaid::

   flowchart TD
       A["Start a phase: create the solver and its time nodes"] --> B["Advance the solver one step and store the state"]
       B --> C["Check events, apply their commands"]
       C --> D{"What did the commands ask for?"}
       D -- "nothing" --> B
       D -- "roll back" --> E["Restore the state at the crossing and restart the solver"]
       E --> B
       D -- "new phase or dynamics" --> A
       D -- "end the flight" --> F["Finalize and cache the results"]

The structure is three nested loops::

    for phase in flight_phases:           # one solver per phase
        for node in phase.time_nodes:     # solver runs up to the next node
            check the node's events
            while solver.status == "running":
                solver.step()
                check events (sampled ones on the overshot part of the step,
                              then the continuous ones at the step's end)
                record post-process variables

Building blocks
---------------

``Flight``
    Owns everything: builds the phase list and the event set, runs the loops
    above, and stores the result in ``flight.solution``.

``_FlightPhases`` / ``_FlightPhase`` (``helpers/flight_phase.py``)
    The ordered list of phases and one phase. A phase has a start time ``t``,
    its ``dynamics``, a ``time_bound`` (the start of the next phase), and, once
    running, its ``solver`` and ``time_nodes``. The list always ends with a
    sentinel phase at ``max_time`` that is never flown; it only closes the
    last real phase. Iterating over ``_FlightPhases`` skips that sentinel.

``_PhaseDynamics`` (``helpers/dynamics.py``)
    The equations of motion of a phase and the list of states they integrate.
    Every phase can report the full 13-state canonical vector, so events and
    outputs never need to know which states a phase actually integrates.
    Bound to a flight (``dynamics.bind(flight)``) it is callable as ``(t, u)``,
    which is what the SciPy solver expects.

``_TimeNodes`` / ``_TimeNode``
    The node schedule of one phase. A node is a time and the list of events
    to check there. Continuous events are kept aside in
    ``time_nodes.continuous_events`` and checked after every step instead.

``Event`` and ``Commands`` (``events/``)
    An event is a trigger and a callback. Calling it evaluates the trigger and,
    if it fires, runs the callback, which queues commands on ``event.commands``.
    The loop reads those commands right after the call.

``Solution`` (``solution.py``)
    The stored rows ``[t, *state]`` of the flight, grouped by phase, plus the
    post-process values recorded beside them.

Setting up
----------

``Flight.__init__`` collects the events: the three core events (rail exit,
apogee, impact), the parachute events, the controller events, the sensor events,
and the user's ``custom_events``. Each is sorted into one of two groups:

- **non-overshootable**: continuous events (``sampling_rate=None``) and sampled
  events with ``time_overshootable=False``. These get time nodes, so the solver
  stops exactly when they must be checked.
- **overshootable**: sampled events with ``time_overshootable=True`` (the
  default). These get no nodes. The solver steps past their sampling times and
  they are checked on interpolated states afterwards (see
  :ref:`time_overshoot_processing`).

The phase list starts with the initial phase (the rail, or whatever
``initial_dynamics`` the flight was built with) and the sentinel at
``max_time``. Later phases are added by events while the flight runs.

Running a phase
---------------

``__simulate_phase`` does the following for each phase:

1. **Seed the state.** For every phase after the first, the canonical state that
   ended the previous phase is turned into the new phase's own state through
   ``dynamics.initial_state``, and a new phase is opened in the solution.
2. **Create the solver.** A fresh SciPy solver (LSODA by default) is built for
   the phase's dynamics, starting at the phase's ``t`` and bounded by its
   ``time_bound``.
3. **Build the time nodes.** A node at the phase start, a node at the phase end,
   and one node per sampling time of every non-overshootable sampled event in
   between. Nodes are sorted and nodes at the same time are merged. A phase
   created with ``clear=True`` (the initial phase) has the events removed from
   its first node, so nothing is checked at ``t = 0`` on an incomplete state.
4. **Walk the nodes.** For each node, the solver's ``t_bound`` is set to the
   next node's time, the node's own events are checked, and then the solver is
   stepped until it reaches that bound. If checking the node's events changed
   the schedule (an event was enabled, disabled or added), the bound is
   synchronized again before stepping.

After every solver step (``__execute_solver_step``), the new row
``[solver.t, *solver.y]`` is appended to the solution and ``flight.t`` /
``flight.y_sol`` are updated. Then ``__process_events`` runs, and finally
``__post_process_step`` records the post-process variables of the rows the
step wrote, if the flight records them (see below).

Checking events after a step
----------------------------

``__process_events`` works in this order:

1. **Overshoot processing.** The overshootable events that had a sampling time
   inside the step just taken are checked on the solver's dense output at
   those times, earliest first. This may roll the flight back (see the next
   section) and yields the time and state at which the remaining events are
   checked: either the end of the step, or the rollback point.
2. **Continuous events.** The phase's continuous events are added to the list
   of events to check, and the whole list is sorted by ``priority`` (lower
   first).
3. **Evaluation.** One ``EventContext`` is built for that moment and every
   event in the list is called with it. After each event that fires, its
   commands are applied (``apply_event_commands``).
4. **Restart the solver.** If the flight was rolled back, or a command moved
   the end of the phase, and the phase is still being flown, the solver is
   rebuilt from the flight's current state and runs on to the next node.

.. _time_overshoot_processing:

Time-overshoot processing
-------------------------

A sampled event with ``time_overshootable=True`` does not force the solver to
stop at its sampling times. Instead, after each step, ``__process_overshootable_nodes``
builds the list of sampling times that fell inside the step and, for each one,
interpolates the state there with the solver's dense output and evaluates the
event's trigger on it (``trigger_only=True``). Then:

- If the trigger is false, nothing happens.
- If it is true and the event does **not** declare ``changes_dynamics``, its
  callback runs right there on the interpolated state. If the commands it
  queued change the trajectory (a new phase, new dynamics, or termination), the
  flight is first rolled back to the interpolated point so the commands apply
  from the crossing rather than from the overshot step end.
- If it is true and the event declares ``changes_dynamics=True``, the flight
  is rolled back to the interpolated point and the event is queued to be
  called for real, together with the continuous events, at that rolled-back
  moment. This guarantees the callback's changes to the rocket take effect from
  the crossing on, and that the overshot part of the step is discarded.

A rollback (``apply_rollback_command``) sets ``flight.t`` and ``flight.y_sol``
to the interpolated point and replaces the last stored row with it. Sampling
times later in the same step are not checked: they will be reached again after
the restart.

This is why ``time_overshootable=True`` is the default and much faster: the
solver keeps its natural step size, and only the rare step that contains a
trigger is redone.

Applying commands
-----------------

``apply_event_commands`` (``helpers/event_commands.py``) reads the commands an
event queued and applies them, in this order:

1. **Exact time.** If the event solved its exact time, the exact row is written
   into the solution. When the event changes the trajectory, the exact row
   replaces the end of the step and the flight continues from it; otherwise it
   is inserted just before the step's end and the step stands as flown.
2. **New phase or dynamics.** ``set_dynamics`` and ``start_flight_phase`` add a
   phase to the list right after the current one, starting at the trigger time
   (plus ``lag``, if any) with the requested dynamics or, if none was given,
   the current ones, and the flight is rolled back to the trigger. Without a
   lag the current phase ends there: its remaining nodes are dropped and its
   solver is marked finished, so the outer loop picks up the new phase. With a
   lag the current phase keeps flying its own equations until the new phase
   begins: its schedule is cut at that time and its solver is restarted from
   the trigger (a parachute keeps falling freely while it opens).
3. **Termination.** ``terminate_flight`` truncates the phase list after the
   current phase and inserts a final phase at the trigger time, so the outer
   loop ends there.
4. **Event list changes.** ``add_event`` registers a new event with the flight
   and, if it is non-overshootable, adds its nodes to the current phase.
   ``enable``/``disable`` commands, including ``trigger_only_once`` and the
   ``enable_on``/``disable_on`` gates, update the event and add or remove its
   nodes from the current phase's schedule.

After the events of a node are checked, the solver bound is set again, since
step 4 may have changed the node schedule.

Post-process variables
----------------------

The accelerations, aerodynamic forces and moments, and net thrust are not
integrated; they are computed from a stored row by evaluating the equations of
motion again with ``post_processing=True``. For most flights that is done once,
after the simulation, over all stored rows.

A flight with an event that declares ``changes_dynamics=True`` cannot do that:
a controller may have moved an air brake mid-flight, and a replay afterwards
would read the rocket as it ended the flight. Such a flight records the values
step by step during the simulation (``__post_process_step``), one extra
evaluation of the equations of motion per stored row. Whether the flight records
is decided in ``__init__`` from the events it was built with, which is why an
event added later with ``add_event`` cannot turn recording on for the rows
already flown.

Where to look
-------------

- ``Flight.__simulate``, ``__simulate_phase``, ``__simulate_phase_nodes``,
  ``__run_node_solver_loop``, ``__process_events``,
  ``__process_overshootable_nodes``, ``__restart_phase_solver``.
- ``helpers/event_calling.py``: ``EventContext``, ``build_event_kwargs``,
  ``call_events``, ``process_overshootable_event``.
- ``helpers/event_commands.py``: ``apply_event_commands`` and the
  ``apply_*`` functions it calls.
- ``helpers/flight_phase.py``: ``_FlightPhases``, ``_FlightPhase``,
  ``_TimeNodes``.
- ``helpers/dynamics.py``: ``_PhaseDynamics`` and the built-in phases.
