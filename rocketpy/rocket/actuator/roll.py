from rocketpy.prints.roll_actuator_prints import _RollActuatorPrints

from .actuator import Actuator


class RollActuator(Actuator):
    """Roll actuator class as a controllable component. Inherits from Actuator.

    This class represents a roll actuator that applies roll torque around the rocket's Z-axis.
    Magic/hand-of-god roll torque is assumed. The roll torque is positive for counter-clockwise rotation when
    viewed from the nose of the rocket.
    This actuator is typically controlled by a controller function similar to air brakes
    and is used by ``Flight`` to model roll control  system.

    Attributes
    ----------
    name : str
        Name of the roll actuator.
    demand_rate : float
        Demand rate of the roll control in Hz. None indicates a continuous-time actuator.
    actuator_range : float
        Range of the roll control in N·m.
    actuator_rate_limit : float
        Rate limit of the roll control in N·m/s. The roll torque change is limited to this rate.
    clamp : bool, optional
        If True, roll torque is clamped to actuator_range.
        If False, a warning is issued when roll torque exceeds the range.
    actuator_time_constant : float
        Time constant for the roll torque actuator dynamics (first-order IIR filter) in seconds.
    actuator_initial_output : float
        Initial roll torque in N·m.
    roll_torque : float
        Current roll torque output magnitude in N·m (Newton-meters).
        Positive values indicate counter-clockwise rotation when viewed
        from the nose of the rocket.
    """

    def __init__(
        self,
        name="Roll Control",
        demand_rate=100,
        max_roll_torque=0,
        torque_rate_limit=None,
        clamp=True,
        initial_roll_torque=0.0,
        roll_torque_time_constant=None,
    ):
        """Initializes the RollControl class.

        Parameters
        ----------
        name : str, optional
            Name of the roll actuator. Default is "Roll Control".
        demand_rate : int, optional
            Demand rate of the roll actuator in Hz. Default is 100 Hz.
            None indicates a continuous-time actuator.
        max_roll_torque : float, int
            Maximum roll torque magnitude in N·m. Must be non-negative.
            Default is 0 (no roll control).
        torque_rate_limit : float, int
            Maximum roll torque rate in N·m/s. Roll torque is limited to this
            rate. Must be non-negative. Default is None (no limit). demand_rate must be specified if torque_rate_limit is not None.
        clamp : bool, optional
            If True, the simulation will clamp roll torque to the range
            [-max_roll_torque, max_roll_torque] if it exceeds this range.
            If False, the simulation will issue a warning if roll torque
            exceeds the maximum value. Default is True.
        initial_roll_torque : float, optional
            Initial roll torque in N·m. Default is 0.0 (no torque).
        roll_torque_time_constant : float, optional
            Time constant for the roll torque actuator dynamics (first-order IIR
            filter) in seconds. If None, no actuator dynamics are applied.
            Must be non-negative. Default is None. demand_rate must be specified if roll_torque_time_constant is not None.

        Returns
        -------
        None
        """
        super().__init__(
            name=name,
            demand_rate=demand_rate,
            actuator_range=(-max_roll_torque, max_roll_torque),
            actuator_rate_limit=torque_rate_limit,
            clamp=clamp,
            actuator_initial_output=initial_roll_torque,
            actuator_time_constant=roll_torque_time_constant,
        )
        self.prints = _RollActuatorPrints(self)

    @property
    def roll_torque(self):
        """Returns the current roll torque in N·m."""
        return self.actuator_output

    @roll_torque.setter
    def roll_torque(self, value):
        """Sets the roll torque in N·m."""
        self.actuator_output = value

    def info(self):
        """Prints summarized information of the roll control system.

        Returns
        -------
        None
        """
        self.prints.basics()

    def all_info(self):
        """Prints all information of the roll control system.

        Returns
        -------
        None
        """
        self.info()

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        return {
            "name": self.name,
            "demand_rate": self.demand_rate,
            "max_roll_torque": self.actuator_range[1],
            "torque_rate_limit": self.actuator_rate_limit,
            "clamp": self.clamp,
            "initial_roll_torque": self.actuator_initial_output,
            "roll_torque_time_constant": self.actuator_time_constant,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data.get("name"),
            demand_rate=data.get("demand_rate"),
            max_roll_torque=data.get("max_roll_torque"),
            torque_rate_limit=data.get("torque_rate_limit"),
            clamp=data.get("clamp"),
            initial_roll_torque=data.get("initial_roll_torque"),
            roll_torque_time_constant=data.get("roll_torque_time_constant"),
        )
