from .motor_plots import _MotorPlots


class _LiquidMotorPlots(_MotorPlots):
    """Class that holds plot methods for LiquidMotor class.

    Attributes
    ----------
    _LiquidMotorPlots.liquid_motor : LiquidMotor
        LiquidMotor object that will be used for the plots.

    """

    def __init__(self, liquid_motor):
        """Initializes _MotorClass class.

        Parameters
        ----------
        liquid_motor : LiquidMotor
            Instance of the LiquidMotor class

        Returns
        -------
        None
        """
        super().__init__(liquid_motor)

    def all(self):
        """Prints out all graphs available about the LiquidMotor. It simply calls
        all the other plotter methods in this class.

        Returns
        -------
        None
        """
        self.thrust(*self.motor.burn_time)
        self.mass_flow_rate(*self.motor.burn_time)
        self.exhaust_velocity(*self.motor.burn_time)
        self.total_mass(*self.motor.burn_time)
        self.propellant_mass(*self.motor.burn_time)
        self.center_of_mass(*self.motor.burn_time)
        self.I_11(*self.motor.burn_time)
        self.I_22(*self.motor.burn_time)
        self.I_33(*self.motor.burn_time)
        self.I_12(*self.motor.burn_time)
        self.I_13(*self.motor.burn_time)
        self.I_23(*self.motor.burn_time)
