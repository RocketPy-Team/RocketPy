import numpy as np

from ..helpers.dynamics import CANONICAL_INDEX


class _StateView:
    """Thirteen standard values and a phase's own, read by name or position.

    Shared by :class:`_State` and :class:`_StateDot`, which differ only in the
    names they give the values. A subclass writes out one property per
    standard value, lists their names in ``_standard_names``, and says how a
    phase's own state is named through ``_phase_name``.
    """

    __slots__ = ("values", "_phase_names", "_phase_values")

    def __init__(self, values, phase_names=(), phase_values=()):
        """Wrap the standard values, and optionally a phase's own.

        Parameters
        ----------
        values : sequence of float
            The thirteen standard values, in the standard order.
        phase_names : sequence of str, optional
            Names of the states the flight phase follows. Default is none.
        phase_values : sequence of float, optional
            The value of each name in ``phase_names``, in the same order.
        """
        self.values = np.asarray(values, dtype=float)
        self._phase_names = phase_names
        self._phase_values = phase_values

    _standard_names = ()

    @staticmethod
    def _phase_name(name):
        """Return the name a phase's own state ``name`` is read by."""
        return name

    @property
    def names(self):
        """Every name that can be read, standard ones first."""
        return self._standard_names + tuple(
            self._phase_name(name)
            for name in self._phase_names
            if name not in CANONICAL_INDEX
        )

    def __getattr__(self, name):
        # Only reached for a name that is not one of the standard thirteen,
        # so those pay nothing for it.
        if name.startswith("_"):
            raise AttributeError(name)
        for index, phase_name in enumerate(self._phase_names):
            if (
                phase_name not in CANONICAL_INDEX
                and self._phase_name(phase_name) == name
            ):
                return self._phase_values[index]
        raise AttributeError(
            f"There is no value named {name!r} here. The names available in "
            f"the current flight phase are: {', '.join(self.names)}."
        )

    def __getitem__(self, key):
        return self.values[key]

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __array__(self, dtype=None, copy=None):  # pylint: disable=unused-argument
        return self.values if dtype is None else self.values.astype(dtype)

    def __repr__(self):
        pairs = ", ".join(f"{name}={getattr(self, name):.6g}" for name in self.names)
        return f"{type(self).__name__.lstrip('_')}({pairs})"


class _State(_StateView):
    """The rocket's states at one moment, read by name or by position.

    ``context.state`` is one of these. Read a value by its name, such as
    ``state.vz``, or by its position in the standard list of thirteen, such as
    ``state[5]``. Both give the same number.

    The thirteen standard states are, in order: ``x``, ``y``, ``z``, ``vx``,
    ``vy``, ``vz``, ``e0``, ``e1``, ``e2``, ``e3``, ``w1``, ``w2``, ``w3``.
    Every flight phase reports all of them.

    A flight phase may also follow states of its own. Those are read by name in
    the same way, such as ``state.my_state`` for a state the phase calls
    ``my_state``, while the flight is in that phase. They have no position: ``state[5]``,
    ``len(state)`` and looping over the state cover the thirteen standard
    values only.

    Attributes
    ----------
    values : numpy.ndarray
        The thirteen standard values as a plain array, in the order above. Use
        it for arithmetic on the whole state, such as ``state.values * 2``.
    names : tuple of str
        Every name that can be read from this state: the thirteen standard
        ones followed by any the current flight phase adds.
    """

    __slots__ = ()

    _standard_names = tuple(CANONICAL_INDEX)

    # Written out one by one, rather than generated, so that code editors can
    # list them.
    @property
    def x(self):
        """Position towards the East, in m."""
        return self.values[0]

    @property
    def y(self):
        """Position towards the North, in m."""
        return self.values[1]

    @property
    def z(self):
        """Altitude above sea level, in m."""
        return self.values[2]

    @property
    def vx(self):
        """Velocity towards the East, in m/s."""
        return self.values[3]

    @property
    def vy(self):
        """Velocity towards the North, in m/s."""
        return self.values[4]

    @property
    def vz(self):
        """Velocity upwards, in m/s."""
        return self.values[5]

    @property
    def e0(self):
        """First component of the attitude quaternion."""
        return self.values[6]

    @property
    def e1(self):
        """Second component of the attitude quaternion."""
        return self.values[7]

    @property
    def e2(self):
        """Third component of the attitude quaternion."""
        return self.values[8]

    @property
    def e3(self):
        """Fourth component of the attitude quaternion."""
        return self.values[9]

    @property
    def w1(self):
        """Angular velocity around the rocket's body x axis, in rad/s."""
        return self.values[10]

    @property
    def w2(self):
        """Angular velocity around the rocket's body y axis, in rad/s."""
        return self.values[11]

    @property
    def w3(self):
        """Angular velocity around the rocket's body z axis, in rad/s."""
        return self.values[12]


class _StateDot(_StateView):
    """The derivative of the rocket's states with respect to time.

    ``context.state_dot`` is one of these. Read a value by its name, such as
    ``state_dot.az``, or by its position in the standard list of thirteen, such
    as ``state_dot[5]``. Both give the same number.

    The thirteen standard values are the time derivatives of the thirteen
    standard states, in the same order: ``vx``, ``vy``, ``vz``, ``ax``, ``ay``,
    ``az``, ``e0_dot``, ``e1_dot``, ``e2_dot``, ``e3_dot``, ``alpha1``,
    ``alpha2``, ``alpha3``. The names match those of :class:`rocketpy.Flight`,
    such as ``flight.az`` and ``flight.alpha1``. Every flight phase reports all
    of them.

    The time derivative of a state a flight phase follows on its own is read by
    that state's name followed by ``_dot``. For a state the phase calls
    ``my_state``, that is ``state_dot.my_state_dot``, while the flight is in
    that phase. These have no
    position: ``state_dot[5]``, ``len(state_dot)`` and looping over it cover
    the thirteen standard values only.

    Attributes
    ----------
    values : numpy.ndarray
        The thirteen standard values as a plain array, in the order above.
    names : tuple of str
        Every name that can be read: the thirteen standard ones followed by
        any the current flight phase adds.
    """

    __slots__ = ()

    _standard_names = (
        "vx",
        "vy",
        "vz",
        "ax",
        "ay",
        "az",
        "e0_dot",
        "e1_dot",
        "e2_dot",
        "e3_dot",
        "alpha1",
        "alpha2",
        "alpha3",
    )

    @staticmethod
    def _phase_name(name):
        return f"{name}_dot"

    # Written out one by one, rather than generated, so that code editors can
    # list them.
    @property
    def vx(self):
        """Velocity towards the East, in m/s."""
        return self.values[0]

    @property
    def vy(self):
        """Velocity towards the North, in m/s."""
        return self.values[1]

    @property
    def vz(self):
        """Velocity upwards, in m/s."""
        return self.values[2]

    @property
    def ax(self):
        """Acceleration towards the East, in m/s²."""
        return self.values[3]

    @property
    def ay(self):
        """Acceleration towards the North, in m/s²."""
        return self.values[4]

    @property
    def az(self):
        """Acceleration upwards, in m/s²."""
        return self.values[5]

    @property
    def e0_dot(self):
        """Time derivative of the first component of the attitude quaternion."""
        return self.values[6]

    @property
    def e1_dot(self):
        """Time derivative of the second component of the attitude quaternion."""
        return self.values[7]

    @property
    def e2_dot(self):
        """Time derivative of the third component of the attitude quaternion."""
        return self.values[8]

    @property
    def e3_dot(self):
        """Time derivative of the fourth component of the attitude quaternion."""
        return self.values[9]

    @property
    def alpha1(self):
        """Angular acceleration around the rocket's body x axis, in rad/s²."""
        return self.values[10]

    @property
    def alpha2(self):
        """Angular acceleration around the rocket's body y axis, in rad/s²."""
        return self.values[11]

    @property
    def alpha3(self):
        """Angular acceleration around the rocket's body z axis, in rad/s²."""
        return self.values[12]
