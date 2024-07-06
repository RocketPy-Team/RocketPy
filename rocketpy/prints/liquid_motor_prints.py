class _LiquidMotorPrints:
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
        self.liquid_motor = liquid_motor

    def nozzle_details(self):
        """Prints out all data available about the Nozzle.

        Returns
        -------
        None
        """
        print("Nozzle Details")
        print("Nozzle Radius: " + str(self.liquid_motor.nozzle_radius) + " m\n")

    def motor_details(self):
        """Prints out all data available about the motor.

        Returns
        -------
        None
        """
        print("Motor Details")
        print(f"Total Burning Time: {self.liquid_motor.burn_duration} s")
        print(
            f"Total Propellant Mass: {self.liquid_motor.propellant_initial_mass:.3f} kg"
        )
        avg = self.liquid_motor.exhaust_velocity.average(*self.liquid_motor.burn_time)
        print(f"Average Propellant Exhaust Velocity: {avg:.3f} m/s")
        print(f"Average Thrust: {self.liquid_motor.average_thrust:.3f} N")
        print(
            f"Maximum Thrust: {self.liquid_motor.max_thrust} N at "
            f"{self.liquid_motor.max_thrust_time} s after ignition."
        )
        print(f"Total Impulse: {self.liquid_motor.total_impulse:.3f} Ns\n")

    def all(self):
        """Prints out all data available about the LiquidMotor.

        Returns
        -------
        None
        """
        self.nozzle_details()
        self.motor_details()
