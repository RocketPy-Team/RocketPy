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
        self.new_dynamics = None
        self.new_dynamics_kwargs = {}
        self.new_flight_phase = None
        self.new_flight_phase_name = None
        self.new_flight_phase_lag = 0
        self._terminate = False
        self.terminate_phase_name = None
        self.surface_switches = []

    def disable(self):
        self._disabled = True

    def enable(self):
        self._disabled = False

    def add_event(self, event):
        """Bring a new event into the flight, from here on.

        The event joins the flight at the moment this runs and is checked from
        the next time node onwards. It never sees the part of the flight that
        already happened.

        Use this when the number of events is not known when the flight is
        built. When you do know, prefer giving the event to the ``Flight`` with
        ``enabled=False`` and turning it on later, through ``enable_on`` or the
        ``enable`` command. That way the flight knows about the event from the
        start, which matters for the case below.

        Parameters
        ----------
        event : Event
            The event to add.

        Warns
        -----
        UserWarning
            If the event has ``changes_dynamics=True`` and the flight was not
            already recording its post-process variables. A flight decides
            whether to record the accelerations, aerodynamic forces and moments
            and net thrust before it starts, from the events it was built with.
            An event that changes the rocket and arrives later cannot turn that
            on for the part of the flight already flown, so those variables are
            worked out afterwards from the rocket in the state it ends the
            flight in, and whatever the event changed is reported for the whole
            flight rather than only from the moment it was added. The
            trajectory itself is correct either way. Register such an event up
            front, disabled, to avoid this. When the events cannot be written in
            advance at all, give the flight one disabled placeholder event that
            declares ``changes_dynamics=True``; see "Commands" in the event
            usage guide.
        """
        self.new_events.append(event)

    def disable_event(self, event):
        self.disable_events.append(event)

    def set_dynamics(self, dynamics, **phase_kwargs):
        """Fly the rest of the flight with a different set of equations.

        Parameters
        ----------
        dynamics : _PhaseDynamics or bound dynamics
            The equations of motion to switch to, together with the states they
            integrate. Pass one of the built-in phases (``PARACHUTE_DYNAMICS``,
            ``SIX_DOF_DYNAMICS``, ...) and the flight fills itself in, or pass a
            phase already attached to a flight, such as
            ``flight.u_dot_generalized``, when you want that flight's own choice
            of ascent equations.
        **phase_kwargs
            Values the new equations need that are only known now, passed
            straight through to the derivative on every step. A parachute
            descent uses this to carry which parachute is out::

                commands.set_dynamics(PARACHUTE_DYNAMICS, parachute=main)

        Notes
        -----
        Changing the equations of motion invalidates the trajectory past the
        trigger, so an event that calls this must be created with
        ``changes_dynamics=True``.
        """
        self.new_dynamics = dynamics
        self.new_dynamics_kwargs = phase_kwargs

    def start_flight_phase(self, phase_name=None, lag=0):
        self.new_flight_phase = True
        self.new_flight_phase_name = phase_name
        self.new_flight_phase_lag = lag

    def terminate_flight(self):
        self._terminate = True

    def activate_surface(self, surface):
        """Make an aerodynamic surface produce force from this moment on.

        Use it for a surface that only exists from some point of the flight.
        Build that surface with ``active=False`` so that it starts the flight
        switched off, add it to the rocket as usual, and call this from the
        callback of an event::

            def switch_on(context):
                context.event.commands.activate_surface(my_surface)

        A surface with ``active_during="power_on"`` or ``"power_off"`` still
        only produces force during that motor phase once it is switched on.

        Parameters
        ----------
        surface : GenericSurface
            The surface to switch on. It must already be one of the rocket's
            aerodynamic surfaces.

        See Also
        --------
        deactivate_surface : Switch a surface off.
        """
        self.surface_switches.append((surface, True))

    def deactivate_surface(self, surface):
        """Make an aerodynamic surface stop producing force from this moment on.

        Use it for a surface that stops existing at some point of the flight.
        Call it from the callback of an event::

            def switch_off(context):
                context.event.commands.deactivate_surface(my_surface)

        The change lasts for the rest of this flight only. The surface itself
        is not modified, so another flight of the same rocket starts with it
        switched on again.

        Parameters
        ----------
        surface : GenericSurface
            The surface to switch off. It must be one of the rocket's
            aerodynamic surfaces.

        See Also
        --------
        activate_surface : Switch a surface on.
        """
        self.surface_switches.append((surface, False))

    @property
    def changes_trajectory(self):
        """Whether these commands change what happens at/after the trigger.

        Returns ``True`` when the queued commands start a new flight phase,
        switch the equations of motion, terminate the flight, or switch an
        aerodynamic surface on or off. In all these cases the trajectory past
        the trigger is no longer valid, so during time overshoot the simulation
        must be rolled back to the exact trigger crossing before the commands
        are applied. Pure scheduling changes (enabling/disabling or adding
        events) do not require a rollback.
        """
        return (
            self.new_flight_phase is not None
            or self.new_dynamics is not None
            or self._terminate
            or bool(self.surface_switches)
        )
