Solution Classes
----------------

These classes hold the state history produced by a :class:`rocketpy.Flight`
simulation. They are not part of RocketPy's top-level namespace; a flight's
solution is reached through :attr:`Flight.solution <rocketpy.Flight.solution>`.

The ``Solution`` holds every row of the flight in one list, in the order they
were flown. The flight phases sit alongside it, each saying where its own rows
begin. So a row is read from the solution, and a phase tells you what that row
means::

    for index, phase in enumerate(flight.solution.phases):
        print(phase.name, len(flight.solution.phase_rows(index)))

Read one phase with :meth:`Solution.phase_rows`, :meth:`Solution.phase_time`,
:meth:`Solution.phase_series` and :meth:`Solution.phase_canonical_array`, each
taking the phase's position in :attr:`Solution.phases`.

.. autoclass:: rocketpy.simulation.solution.Solution
   :members:

.. autoclass:: rocketpy.simulation.solution.PhaseSolution
   :members:

Each phase records which states it integrated, described by the dynamics that
phase was flown with. Those are internal to RocketPy for now: read a phase's
``dynamics.states`` to see what it integrated, and ``dynamics.name`` for the
kind of phase it was.
