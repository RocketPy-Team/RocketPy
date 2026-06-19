Simulation Loop
===============

Overview
--------
This page gives a short overview of how the simulation loop is organized. A
``Flight`` is split into phases, each phase is advanced by its own solver, and
events are evaluated at selected time nodes along the way. The main purpose of
this structure is to keep the numerical integration, event handling, and phase
changes coordinated without making the loop hard to extend.

Key concepts
------------
- **Flight**: the top-level simulation object. It builds the phases, runs the
  solver loop, and stores the final results.
- **Phase**: one segment of flight with its own solver and derivative
  function.
- **Time node**: a scheduled time where the solver state is checked and events
  can be evaluated.
- **Event**: a trigger/callback pair that can request simulation changes.
- **Commands**: a small result container used by events to request actions such
  as enabling, disabling, adding, or redirecting simulation behavior.

Classes
-------

- ``Flight``: orchestrates the whole run.
- ``_FlightPhase``: stores the solver, derivative, and timing for one phase.
- ``_FlightPhases``: keeps the ordered list of phases.
- ``_TimeNode`` and ``_TimeNodes``: store the node schedule for a phase.
- ``Event``: evaluates trigger logic, runs callbacks, and optionally refines an
  event time.
- ``Commands``: stores any actions requested by an event callback.

Simulation structure and loop
-----------------------------
At a high level the loop works like this:

1. ``Flight`` builds the phase list and the initial event set.
2. Each phase creates its solver and its ordered time nodes.
3. The solver advances to each node, then events are checked at that time.
4. Event callbacks can request changes through ``Commands``.
5. If a callback changes the phase layout, the loop updates the node schedule.
6. If an overshootable event forces a rollback, the solver is rebuilt from the
   rolled-back state.

In practice, ``Flight`` owns the orchestration, ``Event`` owns the trigger and
callback logic, ``Commands`` stores requested actions, and ``_TimeNodes`` keeps
the schedule that ties everything together.

.. _time_overshoot_processing:

Time-overshoot processing (implementation detail)
-----------------------------------------------
``time_overshootable`` is only used for sampled events. RocketPy checks those
events on interpolated states between solver steps, so it can detect a trigger
without forcing a tiny step size everywhere. If a sampled event triggers in the
overshoot path, the simulation rolls back to that interpolated point and then
restarts the solver from there.

.. important::

  Events that change dynamics should not rely on overshoot handling. In
  those cases, the callback should request an explicit phase or solver change
  instead of trying to move the trigger later.


Post-processing and derivative updates
--------------------------------------
Some callbacks mutate simulation state, such as enabling a controller or
changing a parachute condition. ``Flight`` applies those requests after the
event fires, then refreshes the derivative or phase schedule if needed so the
next solver step uses the updated setup.
