from functools import cached_property

from rocketpy.mathutils.vector_matrix import Matrix, Vector

from .state import _State


class FlightConditions:
    """The flight at one moment: where the rocket is and the air around it.

    One of these is given as ``conditions`` to:

    - the methods of :class:`rocketpy.ThrustEffector` and
      :class:`rocketpy.ForceEffector`;
    - the ``actual_controls`` and ``state_derivative`` of every controllable;
    - the law of a :class:`rocketpy.ContinuousController`.

    An event or controller function reads the same values on its context, as
    ``context.pressure`` or ``context.mach``, and the object itself as
    ``context.conditions``.

    The air data are looked up the first time they are read, so reading only
    what is needed keeps the simulation fast. Every flight phase gives the 13
    standard states listed below, so all of these can be read in any phase.

    Attributes
    ----------
    time : float
        Time, in s.
    state : _State
        The rocket's state, read by name, ``conditions.state.w3``, or by
        position, ``conditions.state[12]``. The 13 standard states are
        ``x``, ``y``, ``z`` (m) and ``vx``, ``vy``, ``vz`` (m/s) in the Earth
        frame, the attitude quaternion ``e0`` to ``e3``, and ``w1``, ``w2``,
        ``w3`` (rad/s) in the rocket's body frame. Any other state the
        current phase follows is read by name only, such as
        ``conditions.state.canards_a1`` for a state of a controllable.
    height_agl : float
        Height above the ground, in m.
    pressure : float
        Air pressure at the rocket's altitude, in Pa.
    density : float
        Air density at the rocket's altitude, in kg/m³.
    speed_of_sound : float
        Speed of sound at the rocket's altitude, in m/s.
    dynamic_viscosity : float
        Dynamic viscosity of the air at the rocket's altitude, in Pa s.
    wind_velocity : Vector
        Wind velocity at the rocket's altitude, in m/s, in the Earth frame
        (x East, y North, z up).
    stream_velocity : Vector
        Velocity of the air relative to the rocket, in m/s, in the rocket's
        body frame (z along the rocket's axis, toward the nose).
    stream_speed : float
        Speed of the air relative to the rocket, in m/s.
    mach : float
        Mach number of the air relative to the rocket.
    body_to_earth : Matrix
        Rotation from the rocket's body frame to the Earth frame:
        ``conditions.body_to_earth @ vector`` gives, in the Earth frame, a
        vector given in the body frame.
    earth_to_body : Matrix
        The opposite rotation, from the Earth frame to the body frame.
    rocket : Rocket
        The rocket being flown.
    environment : Environment
        The environment it flies in.
    """

    def __init__(
        self,
        rocket,
        environment,
        time,
        state,
        phase_names=(),
        phase_state=(),
    ):
        """Describe the flight of ``rocket`` in ``environment`` at ``time``.

        Parameters
        ----------
        rocket : Rocket
            The rocket being flown.
        environment : Environment
            The environment it flies in.
        time : float
            Time, in s.
        state : sequence of float
            The rocket's 13 standard states, in their order (see the class).
        phase_names : sequence of str, optional
            The names of the states the flight phase follows, to read the
            ones that are not among the 13 by name, including the states of
            the rocket's controllables. Default is none.
        phase_state : sequence of float, optional
            The value of each state in ``phase_names``, in the same order.
        """
        self.time = time
        self._state = state
        self.rocket = rocket
        self.environment = environment
        self._phase_names = phase_names
        self._phase_state = phase_state

    @cached_property
    def state(self):
        """The rocket's state, by name or by position (see the class)."""
        return _State(self._state, self._phase_names, self._phase_state)

    @cached_property
    def height_agl(self):
        """Height of the rocket above the ground, in m."""
        return self._state[2] - self.environment.elevation

    def states_of(self, controllable):
        """Return the values of ``controllable``'s own states, in the order of
        its ``states``. Empty for a controllable without states.

        Inside an effector's method, ``conditions.states_of(self)`` gives the
        effector its states without it having to know how they are named in
        the flight.

        Parameters
        ----------
        controllable : Controllable
            The controllable whose states are wanted.

        Returns
        -------
        list of float
        """
        names, values = self._phase_names, self._phase_state
        return [
            values[names.index(name)]
            for name in controllable._state_names()  # pylint: disable=protected-access
        ]

    @cached_property
    def pressure(self):
        """Air pressure at the rocket's altitude, in Pa."""
        return self.environment.pressure.get_value_opt(self._state[2])

    @cached_property
    def density(self):
        """Air density at the rocket's altitude, in kg/m³."""
        return self.environment.density.get_value_opt(self._state[2])

    @cached_property
    def speed_of_sound(self):
        """Speed of sound at the rocket's altitude, in m/s."""
        return self.environment.speed_of_sound.get_value_opt(self._state[2])

    @cached_property
    def wind_velocity(self):
        """Wind velocity at the rocket's altitude, in m/s, in the Earth frame
        (x East, y North, z up)."""
        z = self._state[2]
        return Vector(
            [
                self.environment.wind_velocity_x.get_value_opt(z),
                self.environment.wind_velocity_y.get_value_opt(z),
                0,
            ]
        )

    @cached_property
    def dynamic_viscosity(self):
        """Dynamic viscosity of the air at the rocket's altitude, in Pa s."""
        return self.environment.dynamic_viscosity.get_value_opt(self._state[2])

    @cached_property
    def body_to_earth(self):
        """Rotation from the rocket's body frame to the Earth frame, as a
        matrix."""
        return Matrix.transformation(self._state[6:10])

    @cached_property
    def earth_to_body(self):
        """Rotation from the Earth frame to the rocket's body frame, as a
        matrix."""
        return self.body_to_earth.transpose

    @cached_property
    def _stream_velocity_earth(self):
        """Velocity of the air relative to the rocket, in m/s, in the Earth
        frame."""
        return self.wind_velocity - Vector(self._state[3:6])

    @cached_property
    def stream_velocity(self):
        """Velocity of the air relative to the rocket, in m/s, in the rocket's
        body frame (z along the rocket's axis, toward the nose)."""
        return self.earth_to_body @ self._stream_velocity_earth

    @cached_property
    def stream_speed(self):
        """Speed of the air relative to the rocket, in m/s."""
        return abs(self._stream_velocity_earth)

    @cached_property
    def mach(self):
        """Mach number of the air relative to the rocket."""
        return self.stream_speed / self.speed_of_sound
