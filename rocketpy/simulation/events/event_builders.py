from ..events.event import Event


def out_of_rail_trigger(**kwargs):
    """Check whether the rocket has left the launch rail.

    Parameters
    ----------
    **kwargs : dict
        Event callback arguments containing ``flight`` and ``state``.

    Returns
    -------
    bool
        ``True`` when the rail-exit condition is satisfied.
    """
    flight = kwargs["flight"]
    state = kwargs["state"]
    return len(flight.out_of_rail_state) == 1 and (
        state[0] ** 2 + state[1] ** 2 + (state[2] - flight.env.elevation) ** 2
        >= flight.effective_1rl**2
    )


def out_of_rail_callback(**kwargs):
    """Handle the rail-exit event.

    The callback records the rail-exit state and time, switches the derivative
    function to :func:`flight.u_dot_generalized`, and starts the free-flight phase.

    Parameters
    ----------
    **kwargs : dict
        Event callback arguments containing ``flight`` and ``event``.

    Returns
    -------
    None
        This callback performs in-place updates on ``flight`` and ``event``.
    """
    flight = kwargs["flight"]
    event = kwargs["event"]

    flight.out_of_rail_time = kwargs.get("time")
    flight.out_of_rail_time_index = len(flight.solution) - 1
    flight.out_of_rail_state = kwargs.get("state")

    event.commands.set_derivative(flight.u_dot_generalized)
    event.commands.start_flight_phase("free_flight")


def out_of_rail_exact_time_function(state, **kwargs):
    """Evaluate the rail-exit event function.

    Parameters
    ----------
    state : array_like
        Interpolated flight state vector without time.
    **kwargs : dict
        Event context containing ``flight``.
        - ``flight``: Flight instance.

    Returns
    -------
    float
        Value of the event function. The root corresponds to the effective rail
        length being reached.
    """
    flight = kwargs["flight"]
    return (
        state[0] ** 2
        + state[1] ** 2
        + (state[2] - flight.env.elevation) ** 2
        - flight.effective_1rl**2
    )


def out_of_rail_exact_time_derivative(state, **kwargs):
    """Evaluate the time derivative of the rail-exit event function.

    Parameters
    ----------
    state : array_like
        Interpolated flight state vector without time.
    **kwargs : dict
        Event context containing ``flight``.
        - ``flight``: Flight instance.

    Returns
    -------
    float
        Time derivative of :func:`out_of_rail_exact_time_function`.
    """
    flight = kwargs["flight"]
    return 2.0 * (
        state[0] * state[3]
        + state[1] * state[4]
        + (state[2] - flight.env.elevation) * state[5]
    )


def apogee_trigger(**kwargs):
    """Check whether the apogee event should fire.

    Parameters
    ----------
    **kwargs : dict
        Event callback arguments containing ``flight`` and ``state``.

    Returns
    -------
    bool
        ``True`` when the vertical velocity crosses from positive to non-positive
        after the apogee state has not yet been recorded.
    """
    flight = kwargs["flight"]
    state = kwargs["state"]
    if len(flight.apogee_state) != 1 or len(flight.solution) < 2:
        return False

    previous_vz = flight.solution[-2][6]
    current_vz = state[5]
    return previous_vz > 0 >= current_vz


def apogee_callback(**kwargs):
    """Handle the apogee event.

    The callback records the apogee state and time, updates the apogee position
    attributes, and optionally terminates the flight.

    Parameters
    ----------
    **kwargs : dict
        Event callback arguments containing ``flight`` and ``event``.

    Returns
    -------
    bool
        Always returns ``False`` so the event remains non-blocking.
    """
    flight = kwargs["flight"]
    event = kwargs["event"]
    flight.apogee_state = kwargs.get("state")
    flight.apogee_time = kwargs.get("time")
    flight.apogee_x = flight.apogee_state[0]
    flight.apogee_y = flight.apogee_state[1]
    flight.apogee = flight.apogee_state[2]

    if flight.terminate_on_apogee:
        event.commands.terminate_flight()

    return False


def apogee_event_exact_time_function(state, **_kwargs):
    """Evaluate the apogee event function.

    Parameters
    ----------
    state : array_like
        Interpolated flight state vector without time.
    **kwargs : dict
        Event context (unused here).

    Returns
    -------
    float
        Vertical velocity component. The root corresponds to ``vz = 0``.
    """
    return state[5]


def impact_trigger(**kwargs):
    """Check whether the impact event should fire.

    Parameters
    ----------
    **kwargs : dict
        Event callback arguments containing ``flight`` and ``state``.

    Returns
    -------
    bool
        ``True`` when altitude is below the environment elevation.
    """
    flight = kwargs["flight"]
    state = kwargs["state"]
    return state[2] < flight.env.elevation


def impact_callback(**kwargs):
    """Handle the impact event.

    The callback stores the impact state and velocity and terminates the flight
    using the impact termination phase.

    Parameters
    ----------
    **kwargs : dict
        Event callback arguments containing ``flight`` and ``event``.

    Returns
    -------
    None
        This callback performs in-place updates on ``flight`` and ``event``.
    """
    flight = kwargs["flight"]
    event = kwargs["event"]

    flight.impact_state = kwargs.get("state")
    flight.x_impact = flight.impact_state[0]
    flight.y_impact = flight.impact_state[1]
    flight.z_impact = flight.impact_state[2]
    flight.impact_velocity = flight.impact_state[5]
    flight.impact_time = kwargs.get("time")

    event.commands.terminate_flight()


def impact_event_exact_time_function(state, **kwargs):
    """Evaluate the impact event function.

    Parameters
    ----------
    state : array_like
        Interpolated flight state vector without time.
    **kwargs : dict
        Event context containing ``flight``.
        - ``flight``: Flight instance.

    Returns
    -------
    float
        Altitude relative to the environment elevation. The root corresponds to
        ground impact.
    """
    flight = kwargs["flight"]
    return state[2] - flight.env.elevation


def impact_event_exact_time_derivative(state, **kwargs):
    """Evaluate the time derivative of the impact event function.

    Parameters
    ----------
    state : array_like
        Interpolated flight state vector without time.
    **kwargs : dict
        Event context (unused here).

    Returns
    -------
    float
        Time derivative of :func:`impact_event_exact_time_function`.
    """
    _ = kwargs
    return state[5]


def impact_step_end_function(*_, **context):
    """Return the step-end estimate for the exact-time solver.

    Returns
    -------
    float
        Step size used as the step-end estimate for the cubic Hermite solver.
    """
    return float(context["step_size"])


def build_core_events():
    """Build the default core flight events.

    Returns
    -------
    tuple of Event
        ``(out_of_rail_event, apogee_event, impact_event)``.
    """
    out_of_rail_event = Event(
        trigger=out_of_rail_trigger,
        callback=out_of_rail_callback,
        name="Out Of Rail",
        context=None,
        exact_time_function=out_of_rail_exact_time_function,
        exact_time_config={
            "solver": "cubic_hermite",
            "derivative_function": out_of_rail_exact_time_derivative,
            "max_abs_imag": 0.001,
        },
        sampling_rate=None,
        trigger_only_once=True,
        time_overshootable=False,
        priority=0,
        needs=frozenset(),
    )

    apogee_event = Event(
        trigger=apogee_trigger,
        callback=apogee_callback,
        name="Apogee",
        context=None,
        exact_time_function=apogee_event_exact_time_function,
        sampling_rate=None,
        trigger_only_once=True,
        time_overshootable=False,
        priority=0,
        needs=frozenset(),
    )

    impact_event = Event(
        trigger=impact_trigger,
        callback=impact_callback,
        name="Impact",
        context=None,
        exact_time_function=impact_event_exact_time_function,
        exact_time_config={
            "solver": "cubic_hermite",
            "derivative_function": impact_event_exact_time_derivative,
            "step_end_function": impact_step_end_function,
            "max_abs_imag": 0.001,
        },
        sampling_rate=None,
        trigger_only_once=True,
        time_overshootable=False,
        priority=0,
        needs=frozenset(),
    )

    return out_of_rail_event, apogee_event, impact_event
