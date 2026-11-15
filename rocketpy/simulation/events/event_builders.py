from functools import partial

from ..events.event import Event


def out_of_rail_trigger(flight, context):
    """Check whether the rocket has left the launch rail.

    Parameters
    ----------
    flight : Flight
        The flight the event belongs to.
    context : EventContext
        Event context containing ``state``.

    Returns
    -------
    bool
        ``True`` when the rail-exit condition is satisfied.
    """
    state = context["canonical_state"]
    return len(flight.out_of_rail_state) == 1 and (
        state[0] ** 2 + state[1] ** 2 + (state[2] - flight.env.elevation) ** 2
        >= flight.effective_1rl**2
    )


def out_of_rail_callback(flight, context):
    """Handle the rail-exit event.

    The callback records the rail-exit state and time, switches the derivative
    function to :func:`flight.u_dot_generalized`, and starts the free-flight phase.

    Parameters
    ----------
    flight : Flight
        The flight the event belongs to.
    context : EventContext
        Event context containing ``event``.

    Returns
    -------
    None
        This callback performs in-place updates on ``flight`` and ``event``.
    """
    event = context["event"]

    flight.out_of_rail_time = context["time"]
    flight.out_of_rail_time_index = len(flight.solution) - 1
    flight.out_of_rail_state = context["canonical_state"]

    event.commands.set_dynamics(flight.u_dot_generalized)
    event.commands.start_flight_phase("free_flight")


def out_of_rail_exact_time_function(flight, context):
    """Evaluate the rail-exit event function.

    Parameters
    ----------
    flight : Flight
        The flight the event belongs to.
    context : EventContext
        Event context containing ``state``.

    Returns
    -------
    float
        Value of the event function. The root corresponds to the effective rail
        length being reached.
    """
    state = context["canonical_state"]
    return (
        state[0] ** 2
        + state[1] ** 2
        + (state[2] - flight.env.elevation) ** 2
        - flight.effective_1rl**2
    )


def out_of_rail_exact_time_derivative(flight, context):
    """Evaluate the time derivative of the rail-exit event function.

    Parameters
    ----------
    flight : Flight
        The flight the event belongs to.
    context : EventContext
        Event context containing ``state``.

    Returns
    -------
    float
        Time derivative of :func:`out_of_rail_exact_time_function`.
    """
    state = context["canonical_state"]
    return 2.0 * (
        state[0] * state[3]
        + state[1] * state[4]
        + (state[2] - flight.env.elevation) * state[5]
    )


def apogee_trigger(flight, context):
    """Check whether the apogee event should fire.

    Parameters
    ----------
    flight : Flight
        The flight the event belongs to.
    context : EventContext
        Event context containing ``state``.

    Returns
    -------
    bool
        ``True`` when the vertical velocity crosses from positive to non-positive
        after the apogee state has not yet been recorded.
    """
    state = context["canonical_state"]
    if len(flight.apogee_state) != 1 or len(flight.solution) < 2:
        return False

    previous_vz = flight.solution.value_at(-2, "vz")
    current_vz = state[5]
    return previous_vz > 0 >= current_vz


def apogee_callback(flight, context):
    """Handle the apogee event.

    The callback records the apogee state and time, updates the apogee position
    attributes, and optionally terminates the flight.

    Parameters
    ----------
    flight : Flight
        The flight the event belongs to.
    context : EventContext
        Event context containing ``event``.

    Returns
    -------
    bool
        Always returns ``False`` so the event remains non-blocking.
    """
    event = context["event"]
    flight.apogee_state = context["canonical_state"]
    flight.apogee_time = context["time"]
    flight.apogee_x = flight.apogee_state[0]
    flight.apogee_y = flight.apogee_state[1]
    flight.apogee = flight.apogee_state[2]

    if flight.terminate_on_apogee:
        event.commands.terminate_flight()

    return False


def apogee_event_exact_time_function(context):
    """Evaluate the apogee event function.

    Parameters
    ----------
    context : EventContext
        Event context containing ``state``.

    Returns
    -------
    float
        Vertical velocity component. The root corresponds to ``vz = 0``.
    """
    return context["canonical_state"][5]


def impact_trigger(context):
    """Check whether the impact event should fire.

    Parameters
    ----------
    context : EventContext
        Event context containing ``height_agl``.

    Returns
    -------
    bool
        ``True`` when altitude is below the environment elevation.
    """
    return context["height_agl"] < 0


def impact_callback(flight, context):
    """Handle the impact event.

    The callback stores the impact state and velocity and terminates the flight
    using the impact termination phase.

    Parameters
    ----------
    flight : Flight
        The flight the event belongs to.
    context : EventContext
        Event context containing ``event``.

    Returns
    -------
    None
        This callback performs in-place updates on ``flight`` and ``event``.
    """
    event = context["event"]

    flight.impact_state = context["canonical_state"]
    flight.x_impact = flight.impact_state[0]
    flight.y_impact = flight.impact_state[1]
    flight.z_impact = flight.impact_state[2]
    flight.impact_velocity = flight.impact_state[5]
    flight.impact_time = context["time"]

    event.commands.terminate_flight()


def impact_event_exact_time_function(context):
    """Evaluate the impact event function.

    Parameters
    ----------
    context : EventContext
        Event context containing ``height_agl``.

    Returns
    -------
    float
        Altitude above ground level. The root corresponds to ground impact.
    """
    return context["height_agl"]


def impact_event_exact_time_derivative(context):
    """Evaluate the time derivative of the impact event function.

    Parameters
    ----------
    context : EventContext
        Event context containing ``state``.

    Returns
    -------
    float
        Time derivative of :func:`impact_event_exact_time_function`.
    """
    return context["canonical_state"][5]


def build_core_events(flight):
    """Build the default core flight events of ``flight``.

    The events are the flight's own: they hold the flight and write its
    results (rail exit, apogee, impact) on it, so the flight never has to be
    handed to them through the context, which users' functions also see.

    Parameters
    ----------
    flight : Flight
        The flight the events belong to.

    Returns
    -------
    tuple of Event
        ``(out_of_rail_event, apogee_event, impact_event)``.
    """
    out_of_rail_event = Event(
        trigger=partial(out_of_rail_trigger, flight),
        callback=partial(out_of_rail_callback, flight),
        name="Out Of Rail",
        memory=None,
        exact_time_function=partial(out_of_rail_exact_time_function, flight),
        exact_time_config={
            "solver": "cubic_hermite",
            "derivative_function": partial(out_of_rail_exact_time_derivative, flight),
        },
        sampling_rate=None,
        trigger_only_once=True,
        time_overshootable=False,
        priority=0,
    )

    apogee_event = Event(
        trigger=partial(apogee_trigger, flight),
        callback=partial(apogee_callback, flight),
        name="Apogee",
        memory=None,
        exact_time_function=apogee_event_exact_time_function,
        sampling_rate=None,
        trigger_only_once=True,
        time_overshootable=False,
        priority=0,
    )

    impact_event = Event(
        trigger=impact_trigger,
        callback=partial(impact_callback, flight),
        name="Impact",
        memory=None,
        exact_time_function=impact_event_exact_time_function,
        exact_time_config={
            "solver": "cubic_hermite",
            "derivative_function": impact_event_exact_time_derivative,
        },
        sampling_rate=None,
        trigger_only_once=True,
        time_overshootable=False,
        priority=0,
    )

    return out_of_rail_event, apogee_event, impact_event
