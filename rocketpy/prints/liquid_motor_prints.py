from .motor_prints import _MotorPrints


class _LiquidMotorPrints(_MotorPrints):
    """Class that holds prints methods for LiquidMotor class.

    Attributes
    ----------
    _LiquidMotorPrints.liquid_motor : liquid_motor
        LiquidMotor object that will be used for the prints.

    """

    def __init__(
        self,
        liquid_motor,
    ):
        """Initializes _LiquidMotorPrints class

        Parameters
        ----------
        liquid_motor: LiquidMotor
            Instance of the LiquidMotor class.

        Returns
        -------
        None
        """
        super().__init__(liquid_motor)
        self.liquid_motor = liquid_motor

    def nozzle_details(self):
        """Prints out all data available about the Nozzle.

        Returns
        -------
        None
        """
        print("Nozzle Details")
        print("Nozzle Radius: " + str(self.liquid_motor.nozzle_radius) + " m\n")

    def all(self):
        """Prints out all data available about the LiquidMotor.

        Returns
        -------
        None
        """
        self.nozzle_details()
        self.motor_details()
