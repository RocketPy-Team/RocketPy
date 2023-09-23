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
        return None

    def motor_details(self):
        """Print Motor details.

        Returns
        -------
        None
        """
        print("Motor Details")
        print("Total Burning Time: " + str(self.motor.burn_out_time) + " s")
        print(
            "Total Propellant Mass: "
            + "{:.3f}".format(self.motor.propellant_initial_mass)
            + " kg"
        )
        print(
            "Average Propellant Exhaust Velocity: "
            + "{:.3f}".format(
                self.motor.exhaust_velocity.average(*self.motor.burn_time)
            )
            + " m/s"
        )
        print("Average Thrust: " + "{:.3f}".format(self.motor.average_thrust) + " N")
        print(
            "Maximum Thrust: "
            + str(self.motor.max_thrust)
            + " N at "
            + str(self.motor.max_thrust_time)
            + " s after ignition."
        )
        print("Total Impulse: " + "{:.3f}".format(self.motor.total_impulse) + " Ns\n")
        return None

    def all(self):
        """Prints out all data available about the Motor.

        Returns
        -------
        None
        """
        self.motor_details()
        return None
