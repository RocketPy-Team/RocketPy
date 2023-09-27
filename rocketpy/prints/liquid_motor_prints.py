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
        return None

    def nozzle_details(self):
        """Prints out all data available about the Nozzle.

        Returns
        -------
        None
        """
        print("Nozzle Details")
        print("Nozzle Radius: " + str(self.liquid_motor.nozzle_radius) + " m\n")
        return None

    def motor_details(self):
        """Prints out all data available about the motor.

        Returns
        -------
        None
        """
        print("Motor Details")
        print("Total Burning Time: " + str(self.liquid_motor.burn_duration) + " s")
        print(
            "Total Propellant Mass: "
            + "{:.3f}".format(self.liquid_motor.propellant_initial_mass)
            + " kg"
        )
        print(
            "Average Propellant Exhaust Velocity: "
            + "{:.3f}".format(
                self.liquid_motor.exhaust_velocity.average(*self.liquid_motor.burn_time)
            )
            + " m/s"
        )
        print(
            "Average Thrust: "
            + "{:.3f}".format(self.liquid_motor.average_thrust)
            + " N"
        )
        print(
            "Maximum Thrust: "
            + str(self.liquid_motor.max_thrust)
            + " N at "
            + str(self.liquid_motor.max_thrust_time)
            + " s after ignition."
        )
        print(
            "Total Impulse: "
            + "{:.3f}".format(self.liquid_motor.total_impulse)
            + " Ns\n"
        )
        return None

    def all(self):
        """Prints out all data available about the LiquidMotor.

        Returns
        -------
        None
        """
        self.nozzle_details()
        self.motor_details()
        return None
