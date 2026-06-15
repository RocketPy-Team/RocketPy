class Commands:
    """Command API exposed to hook trigger/callback callables."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.exact_time = None
        self.exact_state = None
        self._disabled = None  # None=no-op, True=disable, False=enable
        self.rollback = False
        self.new_events = []
        self.enable_events = []
        self.disable_events = []
        self.new_controllers = []
        self.disable_controllers = []
        self.new_derivative = None
        self.new_derivative_set = False
        self.new_flight_phase = None
        self.new_flight_phase_name = None
        self.new_flight_phase_lag = 0
        self.new_flight_phase_parachute = None
        self._terminate = False
        self.terminate_phase_name = None

    def disable(self):
        self._disabled = True

    def enable(self):
        self._disabled = False

    def add_event(self, event):
        self.new_events.append(event)

    def disable_event(self, event):
        self.disable_events.append(event)

    def set_derivative(self, derivative):
        self.new_derivative = derivative
        self.new_derivative_set = True

    def start_flight_phase(self, phase_name=None, lag=0, parachute=None):
        self.new_flight_phase = True
        self.new_flight_phase_name = phase_name
        self.new_flight_phase_lag = lag
        self.new_flight_phase_parachute = parachute

    def terminate_flight(self):
        self._terminate = True

    @property
    def alters_trajectory(self):
        """Whether these commands change what happens at/after the trigger.

        Returns ``True`` when the queued commands start a new flight phase,
        set a new derivative, or terminate the flight. In all three cases the
        trajectory past the trigger is no longer valid, so during time
        overshoot the simulation must be rolled back to the exact trigger
        crossing before the commands are applied. Pure scheduling changes
        (enabling/disabling or adding events) do not require a rollback.
        """
        return (
            self.new_flight_phase is not None
            or self.new_derivative_set
            or self._terminate
        )
