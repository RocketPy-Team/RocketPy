from rocketpy.prints.throttle_actuator_prints import _ThrottleActuatorPrints

from .actuator import Actuator


class ThrottleActuator(Actuator):
    """Throttle actuator class as a controllable component. Inherits from Actuator.

    This class represents a throttle actuator that applies throttle around the rocket's engine.
    The throttle is typically controlled by a controller function similar to air brakes
    and is used by ``Flight`` to model throttle control system.

    This class represents a throttle actuator that throttles the rocket's engine.
    The throttle is the fraction of the maximum thrust produced by the engine, ranging from 0 (no thrust) to 1 (full thrust).
    This actuator is typically controlled by a controller function similar to air brakes
    and is used by ``Flight`` to model throttle control system.

    Attributes
    ----------
    name : str
        Name of the throttle actuator.
    demand_rate : float
        Demand rate of the throttle actuator in Hz. None indicates a continuous-time actuator.
    actuator_range : float
        Range of the throttle actuator.
    actuator_rate_limit : float
        Rate limit of the throttle actuator in 1/s. The throttle change is limited to this rate.
    clamp : bool, optional
        If True, throttle is clamped to actuator_range.
        If False, a warning is issued when throttle exceeds the range.
    actuator_time_constant : float
        Time constant for the throttle actuator dynamics (first-order IIR filter) in seconds.
    actuator_initial_output : float
        Initial throttle value.
    throttle : float
        Current throttle value. The throttle is the fraction of the maximum thrust produced by the engine
        ranging from 0 (no thrust) to 1 (full thrust).
    """

    def __init__(
        self,
        name="Throttle Control",
        demand_rate=100,
        throttle_range=(0, 1),
        throttle_rate_limit=None,
        clamp=True,
        initial_throttle=1.0,
        throttle_time_constant=None,
    ):
        """Initializes the ThrottleActuator class.

        Parameters
        ----------
        name : str, optional
            Name of the throttle actuator. Default is "Throttle Control".
        demand_rate : int, optional
            Demand rate of the throttle actuator in Hz. Default is 100 Hz.
            None indicates a continuous-time actuator.
        throttle_range : tuple, optional
            A tuple containing the minimum and maximum throttle values. Default is (0.0, 1.0).
        throttle_rate_limit : float, int
            Maximum throttle rate in 1/s. Throttle is limited to this
            rate. Must be non-negative. Default is None (no limit). demand_rate must be specified if throttle_rate_limit is not None.
        clamp : bool, optional
            If True, the simulation will clamp throttle to the range
            [throttle_range[0], throttle_range[1]] if it exceeds this range.
            If False, the simulation will issue a warning if throttle
            exceeds the maximum value. Default is True.
        initial_throttle : float, optional
            Initial throttle value. Default is 1.0 (full thrust).
        throttle_time_constant : float, optional
            Time constant for the throttle actuator dynamics (first-order IIR
            filter) in seconds. If None, no actuator dynamics are applied.
            Must be non-negative. Default is None. demand_rate must be specified if throttle_time_constant is not None.

        Returns
        -------
        None
        """
        super().__init__(
            name=name,
            demand_rate=demand_rate,
            actuator_range=throttle_range,
            actuator_rate_limit=throttle_rate_limit,
            clamp=clamp,
            actuator_initial_output=initial_throttle,
            actuator_time_constant=throttle_time_constant,
        )
        self.prints = _ThrottleActuatorPrints(self)

    @property
    def throttle(self):
        """Returns the current throttle value."""
        return self.actuator_output

    @throttle.setter
    def throttle(self, value):
        """Sets the throttle value."""
        self.actuator_output = value

    def info(self):
        """Prints summarized information of the throttle control system.

        Returns
        -------
        None
        """
        self.prints.basics()

    def all_info(self):
        """Prints all information of the throttle control system.

        Returns
        -------
        None
        """
        self.info()

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        return {
            "name": self.name,
            "demand_rate": self.demand_rate,
            "throttle_range": self.actuator_range,
            "throttle_rate_limit": self.actuator_rate_limit,
            "clamp": self.clamp,
            "initial_throttle": self.actuator_initial_output,
            "throttle_time_constant": self.actuator_time_constant,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data.get("name"),
            demand_rate=data.get("demand_rate"),
            throttle_range=(
                data.get("throttle_range")[0],
                data.get("throttle_range")[1],
            ),
            throttle_rate_limit=data.get("throttle_rate_limit"),
            clamp=data.get("clamp"),
            initial_throttle=data.get("initial_throttle"),
            throttle_time_constant=data.get("throttle_time_constant"),
        )
