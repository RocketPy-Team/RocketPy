Solution Classes
----------------

These classes hold everything a :class:`rocketpy.Flight` simulation produced
over time, as raw numbers. They are not part of RocketPy's top-level
namespace; a flight's solution is reached through
:attr:`Flight.solution <rocketpy.Flight.solution>`.

Every quantity the flight knows at each stored row is read by name, as an
attribute or as a key, and comes back as an ``(n, 2)`` array of time and
value:

.. code-block:: python

    flight.solution.vz                       # an integrated state
    flight.solution.az                       # a variable worked out from the states
    flight.solution.controls.canards.fin_1   # a control input, by controllable
    flight.solution["vz"], flight.solution["canards.fin_1"]   # the same, by name
    flight.solution.names                    # everything it holds
    flight.solution.at(3.0)                  # all of it at the nearest stored time

The states are the 13 the flight integrates (position, velocity, attitude
and angular rates) plus any state a controllable adds. The variables worked
out from them are the ones a flight phase computes on its way to each state
derivative without integrating them: the accelerations, the aerodynamic
forces and moments, the net thrust. They are worked out again after the
flight the first time one is read, from the stored rows and their control
inputs, and kept; a flight read back from a file cannot report them, since
that needs the equations of motion.

The flight's own :attr:`Flight.az <rocketpy.Flight.az>` and its companions,
and ``flight.controls.canards.fin_1``, wrap these arrays as a
:class:`Function <rocketpy.Function>`, which can be plotted and evaluated at
any time. Reach for the solution when you want the values exactly as the
simulation stored them.

What the sensors read sits in the solution too, at the sensors' own sampling
times rather than the solver's rows: ``flight.solution.sensors.barometer.pressure``
is the ``(n, 2)`` array of readings, and ``flight.sensor_history`` the same as
Functions. What events recorded with ``context.record`` is the user's own
data, kept exactly as given in ``flight.records``, not in the solution.

The ``Solution`` holds every row of the flight in one list, in the order they
were flown. The flight phases sit alongside it, each saying where its own rows
begin. So a row is read from the solution, and a phase tells you what that row
means. For example, this lists the phases of a flight and how many rows each
one holds:

.. code-block:: python

    for index, phase in enumerate(flight.solution.phases):
        start, stop = flight.solution.phase_span(index)
        print(phase.name, stop - start)

For a rocket with a drogue and a main parachute it prints something like:

.. code-block:: text

    initial_phase 50
    free_flight 363
    Drogue_parachute_descent 76
    Main_parachute_descent 59

:meth:`Solution.phase_span` gives where a phase's rows begin and end, so
``flight.solution[start:stop]``, ``flight.solution.vz[start:stop]`` and
``flight.solution.canonical_array[start:stop]`` read just that phase.

.. autoclass:: rocketpy.simulation.solution.Solution
   :members:

The phase objects themselves are internal to RocketPy for now, so their shape
may change between releases. These are the values worth reading off one:

``name``
    The phase's name, such as ``"initial_phase"`` or ``"free_flight"``.
``t_start``
    The time the phase began, in seconds.
``start``
    Where the phase's first row sits in the solution's rows.
``dynamics.states``
    The states the phase integrated, in the order it stored them.
``dynamics.name``
    The kind of phase it was, such as ``"rail"``, ``"six_dof"`` or
    ``"parachute"``.
