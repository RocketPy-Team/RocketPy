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

The phase objects themselves are internal to RocketPy for now, so their shape
may change between releases. These are the values worth reading off one:

``name``
    The phase's name, such as ``"rail"`` or ``"free_flight"``.
``t_start``
    The time the phase began, in seconds.
``start``
    Where the phase's first row sits in the solution's rows.
``dynamics.states``
    The states the phase integrated, in the order it stored them.
``dynamics.name``
    The kind of phase it was.
