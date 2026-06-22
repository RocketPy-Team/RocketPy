class _MotorPrints:
    """Class that holds prints methods for Motor class.

    Attributes
    ----------
    _MotorPrints.motor : Motor
        Motor object that will be used for the prints.

    """

    def __init__(
        self,
        motor,
    ):
        """Initializes _MotorPrints class

        Parameters
        ----------
        motor: Motor
            Instance of the Motor class.

        Returns
        -------
        None
        """
        self.motor = motor

    def motor_details(self):
        """Print Motor details.

        Returns
        -------
        None
        """
        print("Motor Details")
        print("Total Burning Time: " + str(self.motor.burn_out_time) + " s")
        print(f"Total Propellant Mass: {self.motor.propellant_initial_mass:.3f} kg")
        print(f"Structural Mass Ratio: {self.motor.structural_mass_ratio:.3f}")
        print(
            "Average Propellant Exhaust Velocity: "
            f"{self.motor.exhaust_velocity.average(*self.motor.burn_time):.3f} m/s"
        )
        print(f"Average Thrust: {self.motor.average_thrust:.3f} N")
        print(
            "Maximum Thrust: "
            + str(self.motor.max_thrust)
            + " N at "
            + str(self.motor.max_thrust_time)
            + " s after ignition."
        )
        print(f"Total Impulse: {self.motor.total_impulse:.3f} Ns\n")

    def all(self):
        """Prints out all data available about the Motor.

        Returns
        -------
        None
        """
        self.motor_details()
