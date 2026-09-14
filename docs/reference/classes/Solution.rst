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
        start, stop = flight.solution.phase_span(index)
        print(phase.name, stop - start)

:meth:`Solution.phase_span` gives where a phase's rows begin and end, so
``flight.solution[start:stop]``, ``flight.solution["vz"][start:stop]`` and
``flight.solution.canonical_array[start:stop]`` read just that phase.

.. autoclass:: rocketpy.simulation.solution.Solution
   :members:

Post-process variables
~~~~~~~~~~~~~~~~~~~~~~

On its way to each state derivative, a flight phase works out quantities it
never integrates: the accelerations, the aerodynamic forces and moments, and
the net thrust. Read them through ``flight.solution.post``::

    post = flight.solution.post
    post.names             # every variable this flight computes
    post["az"]             # its [t, value] history over the whole flight
    post.at(3.0)           # every variable at the nearest stored time

The same values back the flight's own :attr:`Flight.az <rocketpy.Flight.az>`
and its companions, which wrap them as a
:class:`Function <rocketpy.Function>` so they can be plotted and evaluated at
any time. Reach for ``post`` when you want the values exactly as the simulation
stored them, without interpolation.

.. autoclass:: rocketpy.simulation.solution.PostProcessSolution
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
