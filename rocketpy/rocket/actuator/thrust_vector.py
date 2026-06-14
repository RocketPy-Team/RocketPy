from rocketpy.prints.thrust_vector_actuator_prints import _ThrustVectorActuatorPrints

from .actuator import Actuator


class ThrustVectorActuator(Actuator):
    """Thrust vector actuator class as a controllable component. Inherits from Actuator.

    This class represents a thrust vector actuator that deflects the direction
    of the thrust vector through gimbal angle.
    This actuator is typically controlled by a controller function similar to air brakes
    and is used by ``Flight`` to model thrust vector control (TVC).

    Attributes
    ----------
    name : str
        Name of the thrust vector actuator.
    demand_rate : float
        Demand rate of the thrust vector actuator in Hz. None indicates a continuous-time actuator.
    actuator_range : float
        Range of the thrust vector actuator in deg.
    actuator_rate_limit : float
        Rate limit of the thrust vector actuator in deg/sec. The thrust vector change is limited to this rate.
    clamp : bool, optional
        If True, thrust gimbal angle is clamped to actuator_range.
        If False, a warning is issued when thrust vector exceeds the range.
    actuator_time_constant : float
        Time constant for the thrust vector actuator dynamics (first-order IIR filter) in seconds.
    actuator_initial_output : float
        Initial thrust vector gimbal angle in deg.
    gimbal_angle : float
        Current thrust vector gimbal angle in deg.

    """

    def __init__(
        self,
        name="Thrust Vector Control",
        demand_rate=100,
        max_gimbal_angle=10,
        gimbal_rate_limit=None,
        clamp=True,
        initial_gimbal_angle=0.0,
        gimbal_time_constant=None,
    ):
        """Initializes the thrust vector actuator class.

        Parameters
        ----------
        name : str, optional
            Name of the thrust vector actuator. Default is "Thrust Vector Control".
        demand_rate : int, optional
            Demand rate of the thrust vector actuator in Hz. Default is 100 Hz.
            None indicates a continuous-time actuator.
        max_gimbal_angle : float, int
            Maximum gimbal angle in deg. Must be non-negative.
            Default is 10 deg.
        gimbal_rate_limit : float, int
            Maximum gimbal rate in deg/sec. Must be non-negative.
            Default is None (no limit). demand_rate must be specified if gimbal_rate_limit is not None.
        clamp : bool, optional
            If True, the simulation will clamp gimbal angle to the range
            [-max_gimbal_angle, max_gimbal_angle] if it exceeds this range.
            If False, the simulation will issue a warning if gimbal angle
            exceeds the maximum value. Default is True.
        initial_gimbal_angle : float, optional
            Initial gimbal angle in deg. Default is 0.0 (no gimbal).
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
            actuator_range=(-max_gimbal_angle, max_gimbal_angle),
            actuator_rate_limit=gimbal_rate_limit,
            clamp=clamp,
            actuator_initial_output=initial_gimbal_angle,
            actuator_time_constant=gimbal_time_constant,
        )
        self.prints = _ThrustVectorActuatorPrints(self)

    @property
    def gimbal_angle(self):
        """Returns the current gimbal angle in deg."""
        return self.actuator_output

    @gimbal_angle.setter
    def gimbal_angle(self, value):
        """Sets the gimbal angle in deg."""
        self.actuator_output = value

    def info(self):
        """Prints summarized information of the thrust vector actuator.

        Returns
        -------
        None
        """
        self.prints.basics()

    def all_info(self):
        """Prints all information of the thrust vector actuator.

        Returns
        -------
        None
        """
        self.info()

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        return {
            "name": self.name,
            "demand_rate": self.demand_rate,
            "max_gimbal_angle": self.actuator_range[1],
            "gimbal_rate_limit": self.actuator_rate_limit,
            "clamp": self.clamp,
            "initial_gimbal_angle": self.actuator_initial_output,
            "gimbal_time_constant": self.actuator_time_constant,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data.get("name"),
            demand_rate=data.get("demand_rate"),
            max_gimbal_angle=data.get("max_gimbal_angle"),
            gimbal_rate_limit=data.get("gimbal_rate_limit"),
            clamp=data.get("clamp"),
            initial_gimbal_angle=data.get("initial_gimbal_angle"),
            gimbal_time_constant=data.get("gimbal_time_constant"),
        )


class ThrustVectorActuator2D:
    """Dual-axis thrust vector actuator class as a controllable component.

    This class represents dual-axis thrust vector actuator that deflects the direction
    of the thrust vector through dual-axis gimbal angles.
    This actuator is typically controlled by a controller function similar to air brakes
    and is used by ``Flight`` to model thrust vector control (TVC).

    Attributes
    ----------
    ThrustVectorActuator2D.name : str
        Name of the dual-axis thrust vector actuator.
    ThrustVectorActuator2D.demand_rate : float
        Demand rate of the dual-axis thrust vector actuator in Hz. None indicates a continuous-time actuator.
    ThrustVectorActuator2D.actuator_range : float
        Range of the dual-axis thrust vector actuator in deg.
    ThrustVectorActuator2D.actuator_rate_limit : float
        Rate limit of the dual-axis thrust vector actuator in deg/sec. The thrust vector change is limited to this rate.
    ThrustVectorActuator2D.clamp : bool, optional
        If True, thrust vector gimbal angles are clamped to actuator_range.
        If False, a warning is issued when thrust vector exceeds the max value.
    ThrustVectorActuator2D.actuator_time_constant : float
        Time constant for the thrust vector actuator dynamics (first-order IIR filter) in seconds.
    ThrustVectorActuator2D.actuator_initial_output : float
        Initial thrust  gimbal angles in deg.
    ThrustVectorActuator2D.actuator_output : float
        Current thrust vector gimble angles in deg.

    """

    def __init__(
        self,
        name="Thrust Vector Control",
        demand_rate=100,
        max_gimbal_angle=10,
        gimbal_rate_limit=None,
        clamp=True,
        initial_gimbal_angle=0.0,
        gimbal_time_constant=None,
    ):
        """Initializes the dual-axis thrust vector actuator class.

        Parameters
        ----------
        name : str, optional
            Name of the dual-axis thrust vector actuator. Default is "Thrust Vector Control (X/Y)-axis".
        demand_rate : int, optional
            Demand rate of the dual-axis thrust vector actuator in Hz. Default is 100 Hz.
            None indicates a continuous-time actuator.
        max_gimbal_angle : float, int
            Maximum gimbal angle in deg. Must be non-negative.
            Default is 10 deg.
        gimbal_rate_limit : float, int
            Maximum gimbal rate in deg/sec. Must be non-negative.
            Default is None (no limit). demand_rate must be specified if gimbal_rate_limit is not None.
        clamp : bool, optional
            If True, the simulation will clamp gimbal angles to the range
            [-max_gimbal_angle, max_gimbal_angle] if it exceeds this range.
            If False, the simulation will issue a warning if gimbal angle
            exceeds the maximum value. Default is True.
        initial_gimbal_angle : float, optional
            Initial gimbal angle in deg. Default is 0.0 (no gimbal).
        gimbal_time_constant : float, optional
            Time constant for the gimbal actuator dynamics (first-order IIR
            filter) in seconds. If None, no actuator dynamics are applied.
            Must be non-negative. Default is None. demand_rate must be specified if gimbal_time_constant is not None.


        Returns
        -------
        None
        """
        self.name = name

        self.x = ThrustVectorActuator(
            name=self.name + " X-axis",
            demand_rate=demand_rate,
            max_gimbal_angle=max_gimbal_angle,
            gimbal_rate_limit=gimbal_rate_limit,
            clamp=clamp,
            initial_gimbal_angle=initial_gimbal_angle,
            gimbal_time_constant=gimbal_time_constant,
        )
        self.y = ThrustVectorActuator(
            name=self.name + " Y-axis",
            demand_rate=demand_rate,
            max_gimbal_angle=max_gimbal_angle,
            gimbal_rate_limit=gimbal_rate_limit,
            clamp=clamp,
            initial_gimbal_angle=initial_gimbal_angle,
            gimbal_time_constant=gimbal_time_constant,
        )

    @property
    def gimbal_angle_x(self):
        """Returns the current gimbal angle around the x-axis (pitch)."""
        return self.x.gimbal_angle

    @gimbal_angle_x.setter
    def gimbal_angle_x(self, value):
        """Sets the gimbal angle in deg."""
        self.x.gimbal_angle = value

    @property
    def gimbal_angle_y(self):
        """Returns the current gimbal angle around the y-axis (yaw)."""
        return self.y.gimbal_angle

    @gimbal_angle_y.setter
    def gimbal_angle_y(self, value):
        """Sets the gimbal angle in deg."""
        self.y.gimbal_angle = value

    @property
    def gimbal_angles(self):
        """Returns a tuple of the current gimbal angles (x, y) in degrees."""
        return (self.gimbal_angle_x, self.gimbal_angle_y)

    @gimbal_angles.setter
    def gimbal_angles(self, value):
        """Sets both gimbal angles from a tuple.

        Parameters
        ----------
        value : tuple
            Tuple of (gimbal_angle_x, gimbal_angle_y) in degrees.
        """
        self.gimbal_angle_x = value[0]
        self.gimbal_angle_y = value[1]

    def info(self):
        """Prints summarized information of the thrust vector actuator.

        Returns
        -------
        None
        """
        self.x.info()
        self.y.info()

    def all_info(self):
        """Prints all information of the thrust vector actuator.

        Returns
        -------
        None
        """
        self.info()

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        return {
            "name": self.name,
            "demand_rate": self.x.demand_rate,
            "max_gimbal_angle": self.x.actuator_range[1],
            "gimbal_rate_limit": self.x.actuator_rate_limit,
            "clamp": self.x.clamp,
            "initial_gimbal_angle": self.x.actuator_initial_output,
            "gimbal_time_constant": self.x.actuator_time_constant,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data.get("name"),
            demand_rate=data.get("demand_rate"),
            max_gimbal_angle=data.get("max_gimbal_angle"),
            gimbal_rate_limit=data.get("gimbal_rate_limit"),
            clamp=data.get("clamp"),
            initial_gimbal_angle=data.get("initial_gimbal_angle"),
            gimbal_time_constant=data.get("gimbal_time_constant"),
        )
